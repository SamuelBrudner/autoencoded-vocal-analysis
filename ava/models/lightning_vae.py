"""
PyTorch Lightning training utilities for the VAE.
"""
from __future__ import annotations

from typing import Optional

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
		lr: float = 1e-3, z_dim: int = 32, model_precision: float = 10.0):
		super().__init__()
		if vae is None:
			vae = VAE(save_dir=save_dir, lr=lr, z_dim=z_dim,
				model_precision=model_precision, device_name="cpu")
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

	def training_step(self, batch, batch_idx):
		loss = self.vae(batch)
		batch_size = batch.shape[0]
		self._train_loss_sum += loss.detach().item()
		self._train_loss_count += batch_size
		self.log("train_loss", loss / batch_size, on_epoch=True,
			prog_bar=True, batch_size=batch_size)
		return loss

	def validation_step(self, batch, batch_idx):
		loss = self.vae(batch)
		batch_size = batch.shape[0]
		self._val_loss_sum += loss.detach().item()
		self._val_loss_count += batch_size
		self.log("val_loss", loss / batch_size, on_epoch=True,
			prog_bar=True, batch_size=batch_size)
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


def build_trainer(save_dir: str = "", epochs: int = 100,
	test_freq: Optional[int] = 2, save_freq: Optional[int] = 10,
	vis_freq: Optional[int] = 1, vis_loader=None, num_specs: int = 5,
	gap=(2, 6), vis_filename: str = "reconstruction.pdf",
	trainer_kwargs: Optional[dict] = None) -> "pl.Trainer":
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
	vae: Optional[VAE] = None):
	"""Train a VAE with Lightning while preserving legacy outputs."""
	if "train" not in loaders or loaders["train"] is None:
		raise ValueError("loaders must include a non-empty 'train' dataloader.")
	module = VAELightningModule(
		vae=vae,
		save_dir=save_dir,
		lr=lr,
		z_dim=z_dim,
		model_precision=model_precision,
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
	)
	val_loader = None
	if test_freq is not None:
		val_loader = loaders.get("test")
	trainer.fit(module, train_dataloaders=loaders["train"],
		val_dataloaders=val_loader)
	return module, trainer
