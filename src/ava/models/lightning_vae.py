"""
PyTorch Lightning training utilities for the VAE.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
	import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - only hit when optional dep missing
	raise ImportError(
		"PyTorch Lightning is required for ava.models.lightning_vae. "
		"Install with `pip install pytorch-lightning`."
	) from exc

from ava.models.vae import VAE


class VAELightningModule(pl.LightningModule):
	"""LightningModule wrapper for the VAE."""

	def __init__(self, vae: Optional[VAE] = None, save_dir: str = "",
		lr: float = 1e-3, z_dim: int = 32, model_precision: float = 10.0,
		input_shape: Optional[tuple[int, int]] = None,
		posterior_type: str = "diag"):
		super().__init__()
		if vae is None:
			vae_kwargs = dict(
				save_dir=save_dir,
				lr=lr,
				z_dim=z_dim,
				model_precision=model_precision,
				device_name="cpu",
				posterior_type=posterior_type,
			)
			if input_shape is not None:
				vae_kwargs["input_shape"] = input_shape
			vae = VAE(**vae_kwargs)
		elif save_dir:
			vae.save_dir = save_dir
		self.vae = vae
		self.save_dir = self.vae.save_dir
		self.save_hyperparameters(ignore=["vae"])
		self._train_loss_sum = 0.0
		self._train_loss_count = 0
		self._val_loss_sum = 0.0
		self._val_loss_count = 0

	def forward(self, x, return_latent_rec: bool = False):
		return self.vae(x, return_latent_rec=return_latent_rec)

	def _compute_loss_and_stats(self, batch):
		mu, u, d = self.vae.encode(batch)
		latent_dist = self.vae._posterior_distribution(mu, u, d)
		z = latent_dist.rsample()
		x_rec = self.vae.decode(z)
		x_flat = batch.view(batch.shape[0], -1)
		elbo = -0.5 * (torch.sum(z ** 2) + self.vae.z_dim * math.log(2 * math.pi))
		pxz_term = -0.5 * x_flat.shape[1] * math.log(
			2 * math.pi / self.vae.model_precision
		)
		l2s = torch.sum((x_flat - x_rec) ** 2, dim=1)
		pxz_term = pxz_term - 0.5 * self.vae.model_precision * torch.sum(l2s)
		elbo = elbo + pxz_term
		elbo = elbo + torch.sum(latent_dist.entropy())
		loss = -elbo
		stats = {
			"recon_mse": F.mse_loss(x_rec, x_flat, reduction="mean"),
			"latent_mean_abs": mu.abs().mean(),
			"latent_var_mean": self.vae._latent_variance_mean(u, d),
		}
		return loss, stats

	def _log_stats(self, stats, stage: str, batch_size: int):
		for name, value in stats.items():
			self.log(
				f"{stage}_{name}",
				value,
				on_step=False,
				on_epoch=True,
				prog_bar=False,
				batch_size=batch_size,
			)

	def training_step(self, batch, batch_idx):
		loss, stats = self._compute_loss_and_stats(batch)
		batch_size = batch.shape[0]
		self._train_loss_sum += loss.detach().item()
		self._train_loss_count += batch_size
		self.log(
			"train_loss",
			loss / batch_size,
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			batch_size=batch_size,
		)
		self._log_stats(stats, "train", batch_size)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, stats = self._compute_loss_and_stats(batch)
		batch_size = batch.shape[0]
		self._val_loss_sum += loss.detach().item()
		self._val_loss_count += batch_size
		self.log(
			"val_loss",
			loss / batch_size,
			on_step=False,
			on_epoch=True,
			prog_bar=True,
			batch_size=batch_size,
		)
		self._log_stats(stats, "val", batch_size)
		return loss

	def on_train_epoch_end(self):
		epoch = int(self.current_epoch)
		if self._train_loss_count:
			self.vae.loss['train'][epoch] = (
				self._train_loss_sum / self._train_loss_count
			)
		self._train_loss_sum = 0.0
		self._train_loss_count = 0
		self.vae.epoch = epoch + 1

	def on_validation_epoch_end(self):
		epoch = int(self.current_epoch)
		if self._val_loss_count:
			self.vae.loss['test'][epoch] = (
				self._val_loss_sum / self._val_loss_count
			)
		self._val_loss_sum = 0.0
		self._val_loss_count = 0

	def configure_optimizers(self):
		return self.vae.optimizer


class VAECheckpointCallback(pl.Callback):
	"""Save legacy checkpoints on the same schedule as the original loop."""

	def __init__(self, save_freq: Optional[int] = 10):
		self.save_freq = save_freq

	def on_train_epoch_end(self, trainer, pl_module):
		if self.save_freq is None:
			return
		epoch = int(trainer.current_epoch)
		if epoch <= 0 or epoch % self.save_freq != 0:
			return
		filename = f"checkpoint_{epoch:03d}.tar"
		pl_module.vae.save_state(filename)


class VAEReconstructionCallback(pl.Callback):
	"""Write reconstruction grids on a fixed epoch cadence."""

	def __init__(self, loader, vis_freq: Optional[int] = 1, num_specs: int = 5,
		gap=(2, 6), filename: str = "reconstruction.pdf"):
		self.loader = loader
		self.vis_freq = vis_freq
		self.num_specs = num_specs
		self.gap = gap
		self.filename = filename

	def on_train_epoch_end(self, trainer, pl_module):
		if self.loader is None or self.vis_freq is None:
			return
		epoch = int(trainer.current_epoch)
		if epoch % self.vis_freq != 0:
			return
		pl_module.vae.visualize(self.loader, num_specs=self.num_specs,
			gap=self.gap, save_filename=self.filename)


class VAEMotivatedStoppingCallback(pl.Callback):
	"""Early stop on ELBO plateaus, reconstruction stagnation, or collapse."""

	def __init__(self, val_patience: Optional[int] = None,
		val_min_delta: float = 0.0, recon_patience: Optional[int] = None,
		recon_min_delta: float = 0.0, collapse_patience: Optional[int] = None,
		collapse_mean_threshold: float = 0.01,
		collapse_var_tolerance: float = 0.1, min_epochs: int = 0):
		self.val_patience = val_patience
		self.val_min_delta = val_min_delta
		self.recon_patience = recon_patience
		self.recon_min_delta = recon_min_delta
		self.collapse_patience = collapse_patience
		self.collapse_mean_threshold = collapse_mean_threshold
		self.collapse_var_tolerance = collapse_var_tolerance
		self.min_epochs = min_epochs
		self._reset_state()

	def _reset_state(self):
		self._best_val = None
		self._val_bad_epochs = 0
		self._best_recon = None
		self._recon_bad_epochs = 0
		self._collapse_epochs = 0
		self._stop_reason = None

	def on_fit_start(self, trainer, pl_module):
		self._reset_state()

	def _metric_value(self, metrics, name: str):
		value = metrics.get(name)
		if value is None:
			return None
		if isinstance(value, torch.Tensor):
			value = value.detach()
			if value.numel() != 1:
				value = value.mean()
			return value.item()
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

	def _request_stop(self, trainer, reason: str):
		self._stop_reason = reason
		if hasattr(trainer, "fit_loop") and hasattr(trainer.fit_loop, "should_stop"):
			trainer.fit_loop.should_stop = True
			return
		trainer.should_stop = True

	def on_validation_epoch_end(self, trainer, pl_module):
		epoch = int(trainer.current_epoch)
		if epoch < self.min_epochs:
			return
		metrics = trainer.callback_metrics
		stop_reasons = []
		if self.val_patience is not None:
			val_loss = self._metric_value(metrics, "val_loss")
			if val_loss is not None:
				if (self._best_val is None or
						val_loss < self._best_val - self.val_min_delta):
					self._best_val = val_loss
					self._val_bad_epochs = 0
				else:
					self._val_bad_epochs += 1
				if self._val_bad_epochs >= self.val_patience:
					stop_reasons.append("val_loss plateau")
		if self.recon_patience is not None:
			recon_mse = self._metric_value(metrics, "val_recon_mse")
			if recon_mse is not None:
				if (self._best_recon is None or
						recon_mse < self._best_recon - self.recon_min_delta):
					self._best_recon = recon_mse
					self._recon_bad_epochs = 0
				else:
					self._recon_bad_epochs += 1
				if self._recon_bad_epochs >= self.recon_patience:
					stop_reasons.append("reconstruction plateau")
		if self.collapse_patience is not None:
			mean_abs = self._metric_value(metrics, "val_latent_mean_abs")
			var_mean = self._metric_value(metrics, "val_latent_var_mean")
			if mean_abs is not None and var_mean is not None:
				if (mean_abs <= self.collapse_mean_threshold and
						abs(var_mean - 1.0) <= self.collapse_var_tolerance):
					self._collapse_epochs += 1
				else:
					self._collapse_epochs = 0
				if self._collapse_epochs >= self.collapse_patience:
					stop_reasons.append("posterior collapse")
		if stop_reasons:
			self._request_stop(trainer, "; ".join(stop_reasons))


def build_trainer(save_dir: str = "", epochs: int = 100,
	test_freq: Optional[int] = 2, save_freq: Optional[int] = 10,
	vis_freq: Optional[int] = 1, vis_loader=None, num_specs: int = 5,
	gap=(2, 6), vis_filename: str = "reconstruction.pdf",
	trainer_kwargs: Optional[dict] = None,
	stopping_kwargs: Optional[dict] = None,
	extra_callbacks: Optional[list] = None) -> "pl.Trainer":
	"""Create a Trainer configured like the legacy training loop."""
	if trainer_kwargs is None:
		trainer_kwargs = {}
	callbacks = []
	if save_freq is not None:
		callbacks.append(VAECheckpointCallback(save_freq=save_freq))
	if vis_freq is not None and vis_loader is not None:
		callbacks.append(VAEReconstructionCallback(
			loader=vis_loader,
			vis_freq=vis_freq,
			num_specs=num_specs,
			gap=gap,
			filename=vis_filename,
		))
	if stopping_kwargs is not None:
		callbacks.append(VAEMotivatedStoppingCallback(**stopping_kwargs))
	if extra_callbacks:
		callbacks.extend(extra_callbacks)
	trainer_kwargs = dict(trainer_kwargs)
	trainer_kwargs.setdefault("enable_checkpointing", False)
	trainer_kwargs.setdefault("logger", False)
	if test_freq is not None:
		trainer_kwargs.setdefault("check_val_every_n_epoch", test_freq)
	default_root_dir = save_dir if save_dir else None
	return pl.Trainer(
		max_epochs=epochs,
		default_root_dir=default_root_dir,
		callbacks=callbacks,
		**trainer_kwargs,
	)


def train_vae(loaders: dict, save_dir: str = "", lr: float = 1e-3,
	z_dim: int = 32, model_precision: float = 10.0, epochs: int = 100,
	test_freq: Optional[int] = 2, save_freq: Optional[int] = 10,
	vis_freq: Optional[int] = 1, num_specs: int = 5, gap=(2, 6),
	vis_filename: str = "reconstruction.pdf", trainer_kwargs: Optional[dict] = None,
	vae: Optional[VAE] = None, stopping_kwargs: Optional[dict] = None,
	extra_callbacks: Optional[list] = None,
	input_shape: Optional[tuple[int, int]] = None,
	posterior_type: str = "diag"):
	"""Train a VAE with Lightning while preserving legacy outputs."""
	if "train" not in loaders or loaders["train"] is None:
		raise ValueError("loaders must include a non-empty 'train' dataloader.")
	module = VAELightningModule(
		vae=vae,
		save_dir=save_dir,
		lr=lr,
		z_dim=z_dim,
		model_precision=model_precision,
		input_shape=input_shape,
		posterior_type=posterior_type,
	)
	vis_loader = loaders.get("test") or loaders["train"]
	trainer = build_trainer(
		save_dir=module.save_dir,
		epochs=epochs,
		test_freq=test_freq,
		save_freq=save_freq,
		vis_freq=vis_freq,
		vis_loader=vis_loader,
		num_specs=num_specs,
		gap=gap,
		vis_filename=vis_filename,
		trainer_kwargs=trainer_kwargs,
		stopping_kwargs=stopping_kwargs,
		extra_callbacks=extra_callbacks,
	)
	val_loader = None
	if test_freq is not None:
		val_loader = loaders.get("test")
	trainer.fit(module, train_dataloaders=loaders["train"],
		val_dataloaders=val_loader)
	return module, trainer
