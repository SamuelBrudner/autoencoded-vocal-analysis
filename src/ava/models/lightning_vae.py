"""
PyTorch Lightning training utilities for the VAE.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Union

import torch
import torch.nn.functional as F

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter

patch_torch_onnx_exporter()

try:
	import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - only hit when optional dep missing
	raise ImportError(
		"PyTorch Lightning is required for ava.models.lightning_vae. "
		"Install with `pip install pytorch-lightning`."
	) from exc
from pytorch_lightning.loggers import TensorBoardLogger

from ava.models.vae import VAE
from ava.models.run_metadata import write_run_metadata


def _unwrap_batch(batch):
	if torch.is_tensor(batch):
		return batch
	if isinstance(batch, dict):
		for value in batch.values():
			if torch.is_tensor(value):
				return value
		return next(iter(batch.values()))
	if isinstance(batch, (list, tuple)) and batch:
		for item in batch:
			if torch.is_tensor(item):
				return item
		return batch[0]
	return batch


def _extract_pair(batch):
	if isinstance(batch, dict):
		base = batch.get("x")
		aug = batch.get("x_aug")
		if torch.is_tensor(base) and torch.is_tensor(aug):
			if base.shape == aug.shape:
				return base, aug
			raise ValueError("Paired tensors must have matching shapes.")
	if isinstance(batch, (list, tuple)) and len(batch) == 2:
		base, aug = batch
		if torch.is_tensor(base) and torch.is_tensor(aug):
			if base.shape == aug.shape:
				return base, aug
			raise ValueError("Paired tensors must have matching shapes.")
	return None, None


def _infer_input_shape(loaders: dict) -> tuple[int, int]:
	loader = loaders.get("train")
	if loader is None:
		raise ValueError("Cannot infer input_shape without a train dataloader.")
	batch = next(iter(loader))
	batch = _unwrap_batch(batch)
	if not hasattr(batch, "shape") or len(batch.shape) < 3:
		raise ValueError("Training batches must have shape [batch, height, width].")
	if batch.dim() == 4 and batch.shape[1] == 1:
		return tuple(batch.shape[2:])
	if batch.dim() == 3:
		return tuple(batch.shape[1:])
	raise ValueError("Training batches must have shape [batch, height, width].")


class VAELightningModule(pl.LightningModule):
	"""LightningModule wrapper for the VAE."""

	def __init__(self, vae: Optional[VAE] = None, save_dir: str = "",
		lr: float = 1e-3, z_dim: int = 32, model_precision: float = 10.0,
		learn_observation_scale: bool = False,
		input_shape: Optional[tuple[int, int]] = None,
		posterior_type: str = "diag", conv_arch: str = "plain",
		decoder_type: str = "upsample",
		compile_model: bool = False,
		compile_kwargs: Optional[dict] = None, kl_beta: float = 1.0,
		kl_warmup_epochs: int = 0,
		invariance_weight: float = 0.0,
		invariance_warmup_epochs: int = 0,
		invariance_loss: str = "mse",
		invariance_stop_grad: Optional[Union[str, bool]] = "none"):
		super().__init__()
		if vae is None:
			vae_kwargs = dict(
				save_dir=save_dir,
				lr=lr,
				z_dim=z_dim,
				model_precision=model_precision,
				learn_observation_scale=learn_observation_scale,
				device_name="cpu",
				posterior_type=posterior_type,
				conv_arch=conv_arch,
				decoder_type=decoder_type,
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
		self.kl_beta = kl_beta
		self.kl_warmup_epochs = kl_warmup_epochs
		self.invariance_weight = float(invariance_weight)
		self.invariance_warmup_epochs = int(invariance_warmup_epochs)
		self.invariance_loss = self._normalize_invariance_loss(invariance_loss)
		self.invariance_stop_grad = self._normalize_stop_grad(invariance_stop_grad)
		if self.invariance_weight < 0:
			raise ValueError("invariance_weight must be non-negative.")
		if self.invariance_warmup_epochs < 0:
			raise ValueError("invariance_warmup_epochs must be non-negative.")
		self._compiled_loss_fn = None
		if compile_model:
			if not hasattr(torch, "compile"):
				raise RuntimeError(
					"torch.compile requested but not available in this "
					"PyTorch version."
				)
			compile_kwargs = dict(compile_kwargs or {})
			self._compiled_loss_fn = torch.compile(
				self._compute_loss_and_stats_impl, **compile_kwargs
			)

	def forward(self, x, return_latent_rec: bool = False):
		return self.vae(x, return_latent_rec=return_latent_rec)

	def _compute_loss_and_stats_impl(self, batch, return_mu: bool = False):
		recon, mu, logvar, z, u = self.vae(batch)
		latent_dist = self.vae._posterior_distribution(mu, logvar, u)
		x_flat = batch.view(batch.shape[0], -1)
		recon_flat = recon.view(batch.shape[0], -1)
		log_precision = self.vae.log_precision
		precision = torch.exp(log_precision)
		log_two_pi = math.log(2 * math.pi)
		pxz_term = -0.5 * x_flat.shape[1] * (log_two_pi - log_precision)
		l2s = torch.sum((x_flat - recon_flat) ** 2, dim=1)
		pxz_term = pxz_term - 0.5 * precision * torch.sum(l2s)
		log_pz = -0.5 * (
			torch.sum(z ** 2) + self.vae.z_dim * math.log(2 * math.pi)
		)
		entropy = torch.sum(latent_dist.entropy())
		kl = -(log_pz + entropy)
		kl_weight = batch.new_tensor(self._kl_weight())
		loss = -(pxz_term + kl_weight * (log_pz + entropy))
		recon_mse = F.mse_loss(recon_flat, x_flat, reduction="mean")
		latent_mean_abs = mu.abs().mean()
		latent_var_mean = self.vae._latent_variance_mean(u, logvar)
		recon_nll = (-pxz_term) / batch.shape[0]
		kl_mean = kl / batch.shape[0]
		if return_mu:
			return (
				loss,
				recon_mse,
				latent_mean_abs,
				latent_var_mean,
				recon_nll,
				kl_mean,
				kl_weight,
				mu,
			)
		return loss, recon_mse, latent_mean_abs, latent_var_mean, recon_nll, kl_mean, kl_weight

	def _compute_loss_and_stats(self, batch):
		base, aug = self._split_batch(batch)
		if aug is None and self.invariance_weight > 0:
			raise ValueError(
				"invariance_weight > 0 requires paired batches with x and x_aug."
			)
		if aug is None and self._compiled_loss_fn is not None:
			(loss, recon_mse, latent_mean_abs, latent_var_mean, recon_nll,
				kl_mean, kl_weight) = (
				self._compiled_loss_fn(base)
			)
			mu_base = None
		else:
			if aug is None:
				(loss, recon_mse, latent_mean_abs, latent_var_mean, recon_nll,
					kl_mean, kl_weight) = (
					self._compute_loss_and_stats_impl(base)
				)
				mu_base = None
			else:
				(loss, recon_mse, latent_mean_abs, latent_var_mean, recon_nll,
					kl_mean, kl_weight, mu_base) = (
					self._compute_loss_and_stats_impl(base, return_mu=True)
				)
		stats = {
			"recon_mse": recon_mse,
			"recon_nll": recon_nll,
			"kl": kl_mean,
			"kl_weight": kl_weight,
			"latent_mean_abs": latent_mean_abs,
			"latent_var_mean": latent_var_mean,
		}
		if aug is not None:
			if mu_base is None:
				mu_base, _, _ = self.vae.encode(base)
			loss, inv_stats = self._apply_invariance_loss(
				loss,
				mu_base,
				aug,
				base.shape[0],
			)
			stats.update(inv_stats)
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

	def _kl_weight(self) -> float:
		if self.kl_warmup_epochs and self.kl_warmup_epochs > 0:
			progress = min(1.0, self.current_epoch / self.kl_warmup_epochs)
		else:
			progress = 1.0
		return float(self.kl_beta) * progress

	def _invariance_weight(self) -> float:
		if self.invariance_warmup_epochs and self.invariance_warmup_epochs > 0:
			progress = min(1.0, self.current_epoch / self.invariance_warmup_epochs)
		else:
			progress = 1.0
		return float(self.invariance_weight) * progress

	def _split_batch(self, batch):
		base, aug = _extract_pair(batch)
		if base is not None:
			return base, aug
		return _unwrap_batch(batch), None

	def _normalize_invariance_loss(self, loss_name: str) -> str:
		if loss_name is None:
			return "mse"
		if not isinstance(loss_name, str):
			raise ValueError("invariance_loss must be a string.")
		loss_name = loss_name.strip().lower()
		if loss_name in ("mse", "l2"):
			return "mse"
		if loss_name in ("cosine", "cos", "cosine_distance"):
			return "cosine"
		raise ValueError("invariance_loss must be 'mse' or 'cosine'.")

	def _normalize_stop_grad(self, stop_grad: Optional[Union[str, bool]]) -> str:
		if isinstance(stop_grad, bool):
			return "aug" if stop_grad else "none"
		if stop_grad is None:
			return "none"
		if not isinstance(stop_grad, str):
			raise ValueError("invariance_stop_grad must be a string or bool.")
		stop_grad = stop_grad.strip().lower()
		if stop_grad in ("none", "off", "false", "no"):
			return "none"
		if stop_grad in ("aug", "x_aug", "augmentation"):
			return "aug"
		if stop_grad in ("base", "x", "original"):
			return "base"
		raise ValueError(
			"invariance_stop_grad must be 'none', 'base', or 'aug'."
		)

	def _apply_invariance_loss(self, loss, mu_base, aug, batch_size: int):
		mu_aug, _, _ = self.vae.encode(aug)
		mu_base_detached = mu_base
		mu_aug_detached = mu_aug
		if self.invariance_stop_grad == "base":
			mu_base_detached = mu_base.detach()
		elif self.invariance_stop_grad == "aug":
			mu_aug_detached = mu_aug.detach()
		if self.invariance_loss == "mse":
			inv_loss = F.mse_loss(
				mu_base_detached,
				mu_aug_detached,
				reduction="mean",
			)
		else:
			cosine_sim = F.cosine_similarity(
				mu_base_detached,
				mu_aug_detached,
				dim=1,
			)
			inv_loss = (1.0 - cosine_sim).mean()
		inv_weight = loss.new_tensor(self._invariance_weight())
		if inv_weight.item() > 0:
			loss = loss + inv_weight * inv_loss * batch_size
		mu_delta = (mu_base - mu_aug).detach()
		inv_l2 = torch.norm(mu_delta, dim=1).mean()
		inv_cosine = F.cosine_similarity(
			mu_base.detach(),
			mu_aug.detach(),
			dim=1,
		).mean()
		stats = {
			"invariance_loss": inv_loss,
			"invariance_weight": inv_weight,
			"invariance_mu_l2": inv_l2,
			"invariance_mu_cosine": inv_cosine,
		}
		if inv_weight.item() == 0:
			return loss, stats
		return loss, stats

	def training_step(self, batch, batch_idx):
		loss, stats = self._compute_loss_and_stats(batch)
		batch_size = self._split_batch(batch)[0].shape[0]
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
		batch_size = self._split_batch(batch)[0].shape[0]
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
	trainer_kwargs.setdefault("accelerator", "auto")
	trainer_kwargs.setdefault("devices", 1)
	if "precision" not in trainer_kwargs:
		accelerator = trainer_kwargs.get("accelerator")
		use_amp = (
			accelerator in (None, "auto", "gpu", "cuda")
			and torch.cuda.is_available()
		)
		trainer_kwargs["precision"] = 16 if use_amp else 32
	trainer_kwargs.setdefault("enable_checkpointing", False)
	if "logger" not in trainer_kwargs:
		log_root = save_dir if save_dir else os.getcwd()
		trainer_kwargs["logger"] = TensorBoardLogger(
			save_dir=log_root,
			name="lightning_logs",
		)
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
	z_dim: int = 32, model_precision: float = 10.0,
	learn_observation_scale: bool = False, epochs: int = 100,
	test_freq: Optional[int] = 2, save_freq: Optional[int] = 10,
	vis_freq: Optional[int] = 1, num_specs: int = 5, gap=(2, 6),
	vis_filename: str = "reconstruction.pdf", trainer_kwargs: Optional[dict] = None,
	vae: Optional[VAE] = None, stopping_kwargs: Optional[dict] = None,
	extra_callbacks: Optional[list] = None,
	input_shape: Optional[tuple[int, int]] = None,
	posterior_type: str = "diag", conv_arch: str = "plain",
	decoder_type: str = "upsample",
	compile_model: bool = False,
	compile_kwargs: Optional[dict] = None, kl_beta: float = 1.0,
	kl_warmup_epochs: int = 0,
	invariance_weight: float = 0.0,
	invariance_warmup_epochs: int = 0,
	invariance_loss: str = "mse",
	invariance_stop_grad: Optional[Union[str, bool]] = "none",
	config_path: Optional[str] = None,
	manifest_path: Optional[str] = None,
	dataset_root: Optional[str] = None):
	"""Train a VAE with Lightning while preserving legacy outputs."""
	if "train" not in loaders or loaders["train"] is None:
		raise ValueError("loaders must include a non-empty 'train' dataloader.")
	if input_shape is None:
		input_shape = _infer_input_shape(loaders)
	module = VAELightningModule(
		vae=vae,
		save_dir=save_dir,
		lr=lr,
		z_dim=z_dim,
		model_precision=model_precision,
		learn_observation_scale=learn_observation_scale,
		input_shape=input_shape,
		posterior_type=posterior_type,
		conv_arch=conv_arch,
		decoder_type=decoder_type,
		compile_model=compile_model,
		compile_kwargs=compile_kwargs,
		kl_beta=kl_beta,
		kl_warmup_epochs=kl_warmup_epochs,
		invariance_weight=invariance_weight,
		invariance_warmup_epochs=invariance_warmup_epochs,
		invariance_loss=invariance_loss,
		invariance_stop_grad=invariance_stop_grad,
	)
	write_run_metadata(
		save_dir=module.save_dir,
		config_path=config_path,
		manifest_path=manifest_path,
		dataset_root=dataset_root,
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
