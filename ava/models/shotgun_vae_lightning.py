"""
PyTorch Lightning wrappers for shotgun VAE training.

These classes preserve the loss/forward behavior of ava.models.vae.VAE while
standardizing the training loop, logging, and checkpointing.
"""

try:
	import pytorch_lightning as pl
except ImportError:
	try:
		import lightning.pytorch as pl
	except ImportError as exc:
		pl = None
		_LIGHTNING_IMPORT_ERROR = exc

from torch.optim import Adam

from ava.models.vae import VAE
from ava.models.window_vae_dataset import get_fixed_window_data_loaders, \
	get_window_partition, get_warped_window_data_loaders


__all__ = ["ShotgunVAELightningModule", "ShotgunVAEDataModule"]


def _require_lightning():
	if pl is None:
		raise ImportError(
			"PyTorch Lightning is required for ava.models.shotgun_vae_lightning. "
			"Install with `pip install pytorch-lightning` or `pip install "
			"lightning`."
		) from _LIGHTNING_IMPORT_ERROR


def _resolve_batch(batch):
	if isinstance(batch, (tuple, list)):
		return batch[0]
	return batch


if pl is not None:
	class ShotgunVAELightningModule(pl.LightningModule):
		"""
		LightningModule wrapper around ava.models.vae.VAE for shotgun training.
		"""

		def __init__(self, lr=1e-3, z_dim=32, model_precision=10.0, \
			save_dir=''):
			_require_lightning()
			super().__init__()
			self.save_hyperparameters(ignore=["save_dir"])
			self.model = VAE(save_dir=save_dir, lr=lr, z_dim=z_dim, \
				model_precision=model_precision, device_name="cpu")

		def forward(self, x, return_latent_rec=False):
			return self.model.forward(x, return_latent_rec=return_latent_rec)

		def training_step(self, batch, batch_idx):
			batch = _resolve_batch(batch)
			loss = self.model(batch)
			batch_size = batch.size(0)
			self.log("train_loss", loss / batch_size, on_step=True, \
				on_epoch=True, prog_bar=True, batch_size=batch_size)
			return loss

		def validation_step(self, batch, batch_idx):
			batch = _resolve_batch(batch)
			loss = self.model(batch)
			batch_size = batch.size(0)
			self.log("val_loss", loss / batch_size, on_step=False, \
				on_epoch=True, prog_bar=True, batch_size=batch_size)
			return loss

		def configure_optimizers(self):
			return Adam(self.model.parameters(), lr=self.hparams.lr)


	class ShotgunVAEDataModule(pl.LightningDataModule):
		"""
		LightningDataModule for fixed or warped shotgun VAE training.
		"""

		def __init__(self, audio_dirs, params, roi_dirs=None, split=0.8, \
			batch_size=64, shuffle=(True, False), num_workers=4, \
			min_spec_val=None, use_train_for_val=False, \
			exclude_empty_roi_files=True, use_warped=False, load_warp=False, \
			warp_fn=None, warp_params=None, warp_type='spectrogram'):
			_require_lightning()
			super().__init__()
			self.audio_dirs = audio_dirs
			self.roi_dirs = roi_dirs
			self.params = params
			self.split = split
			self.batch_size = batch_size
			self.shuffle = shuffle
			self.num_workers = num_workers
			self.min_spec_val = min_spec_val
			self.use_train_for_val = use_train_for_val
			self.exclude_empty_roi_files = exclude_empty_roi_files
			self.use_warped = use_warped
			self.load_warp = load_warp
			self.warp_fn = warp_fn
			self.warp_params = warp_params or {}
			self.warp_type = warp_type
			self._train_loader = None
			self._val_loader = None

		def _build_fixed_loaders(self):
			if self.roi_dirs is None:
				raise ValueError("roi_dirs is required for fixed-window training.")
			partition = get_window_partition(self.audio_dirs, self.roi_dirs, \
				self.split, shuffle=True, \
				exclude_empty_roi_files=self.exclude_empty_roi_files)
			if self.use_train_for_val:
				partition['test'] = partition['train']
			elif len(partition['test']['audio']) == 0:
				partition['test'] = None
			return get_fixed_window_data_loaders(partition, self.params, \
				batch_size=self.batch_size, shuffle=self.shuffle, \
				num_workers=self.num_workers, min_spec_val=self.min_spec_val)

		def _build_warped_loaders(self):
			return get_warped_window_data_loaders(self.audio_dirs, self.params, \
				batch_size=self.batch_size, num_workers=self.num_workers, \
				load_warp=self.load_warp, warp_fn=self.warp_fn, \
				warp_params=self.warp_params, warp_type=self.warp_type)

		def setup(self, stage=None):
			loaders = self._build_warped_loaders() if self.use_warped \
				else self._build_fixed_loaders()
			self._train_loader = loaders['train']
			self._val_loader = loaders['test']

		def train_dataloader(self):
			return self._train_loader

		def val_dataloader(self):
			return self._val_loader
else:
	class ShotgunVAELightningModule:
		def __init__(self, *args, **kwargs):
			_require_lightning()


	class ShotgunVAEDataModule:
		def __init__(self, *args, **kwargs):
			_require_lightning()
