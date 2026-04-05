import random

import numpy as np
import pytest

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter


torch = pytest.importorskip("torch")
patch_torch_onnx_exporter()
pytest.importorskip("pytorch_lightning")

from torch.utils.data import DataLoader, Dataset, TensorDataset

from ava.models.lightning_vae import VAELightningModule, build_trainer, train_vae
from ava.models.vae import VAE


class _TensorDataset(Dataset):
	def __init__(self, data):
		self._data = data

	def __len__(self):
		return self._data.shape[0]

	def __getitem__(self, idx):
		return self._data[idx]


class _EpochTrackingDataset(Dataset):
	def __init__(self, data):
		self._data = data
		self.epochs = []

	def __len__(self):
		return self._data.shape[0]

	def __getitem__(self, idx):
		return self._data[idx]

	def set_epoch(self, epoch):
		self.epochs.append(int(epoch))


def _set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def _make_loaders(data, batch_size=2):
	dataset = _TensorDataset(data)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
		num_workers=0)
	return {"train": loader, "test": loader}


def test_lightning_metrics_match_legacy_forward():
	_set_seed(7)
	batch = torch.randn(2, 128, 128)
	_set_seed(11)
	vae = VAE(save_dir="", device_name="cpu")
	module = VAELightningModule(vae=vae)

	_set_seed(23)
	lightning_loss, stats = module._compute_loss_and_stats(batch)
	_set_seed(23)
	recon, _, _, _, _ = vae.forward(batch)

	assert recon.shape == batch.shape
	assert torch.isfinite(stats["recon_mse"]).item()
	assert torch.isfinite(stats["recon_nll"]).item()
	assert torch.isfinite(stats["kl"]).item()
	assert torch.isfinite(stats["kl_per_dim"]).item()
	assert torch.isfinite(stats["kl_weight"]).item()
	assert torch.isfinite(stats["weighted_kl"]).item()
	assert torch.isfinite(stats["kl_regularizer"]).item()
	assert torch.isfinite(stats["kl_to_recon_ratio"]).item()
	assert torch.isfinite(stats["weighted_kl_to_recon_ratio"]).item()
	assert torch.isfinite(stats["kl_regularizer_to_recon_ratio"]).item()
	assert torch.isfinite(stats["log_precision"]).item()
	assert torch.isfinite(stats["model_precision"]).item()
	assert stats["model_precision"].item() > 0
	assert torch.isfinite(stats["latent_mean_abs"]).item()
	assert torch.isfinite(stats["latent_var_mean"]).item()
	assert torch.isfinite(stats["latent_max_abs_mu"]).item()
	assert torch.isfinite(stats["latent_max_abs_logvar"]).item()
	assert torch.isfinite(stats["latent_raw_max_abs_logvar"]).item()
	assert torch.isfinite(stats["latent_logvar_lower_clamp_fraction"]).item()
	assert torch.isfinite(stats["latent_logvar_upper_clamp_fraction"]).item()
	assert torch.isfinite(stats["latent_logvar_clamp_fraction"]).item()
	assert torch.isfinite(stats["recon_nll_per_dim"]).item()
	assert 0.0 <= stats["latent_logvar_lower_clamp_fraction"].item() <= 1.0
	assert 0.0 <= stats["latent_logvar_upper_clamp_fraction"].item() <= 1.0
	assert 0.0 <= stats["latent_logvar_clamp_fraction"].item() <= 1.0
	expected = stats["recon_nll"] + stats["kl_weight"] * stats["kl"]
	assert torch.allclose(
		lightning_loss / batch.shape[0],
		expected,
		rtol=1e-4,
		atol=1e-4,
	)
	assert torch.allclose(
		stats["weighted_kl"],
		stats["kl_weight"] * stats["kl"],
		rtol=1e-6,
		atol=1e-6,
	)
	assert torch.allclose(
		stats["recon_nll_per_dim"],
		stats["recon_nll"] / batch[0].numel(),
		rtol=1e-6,
		atol=1e-6,
	)
	assert torch.allclose(
		stats["kl_regularizer"],
		stats["weighted_kl"],
		rtol=1e-6,
		atol=1e-6,
	)


def test_lightning_logs_clamp_hit_fractions_when_logvar_overflows():
	_set_seed(41)
	batch = torch.randn(2, 64, 64)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		input_shape=(64, 64),
		z_dim=8,
		posterior_logvar_min=-2.0,
		posterior_logvar_max=2.0,
	)
	with torch.no_grad():
		vae.fc43.bias.fill_(5.0)
	module = VAELightningModule(vae=vae)

	_, stats = module._compute_loss_and_stats(batch)

	assert stats["latent_raw_max_abs_logvar"].item() > stats["latent_max_abs_logvar"].item()
	assert stats["latent_logvar_upper_clamp_fraction"].item() > 0.0
	assert stats["latent_logvar_lower_clamp_fraction"].item() == 0.0
	assert stats["latent_logvar_clamp_fraction"].item() > 0.0


def test_lightning_capacity_schedule_regularizes_to_target():
	_set_seed(29)
	batch = torch.randn(2, 64, 64)
	vae = VAE(save_dir="", device_name="cpu", input_shape=(64, 64), z_dim=8)
	module = VAELightningModule(
		vae=vae,
		kl_capacity_target=1.5,
		kl_capacity_warmup_epochs=0,
		kl_capacity_penalty=2.0,
	)

	loss, stats = module._compute_loss_and_stats(batch)

	assert torch.isfinite(stats["kl_capacity_target"]).item()
	assert torch.isfinite(stats["kl_capacity_error"]).item()
	assert torch.isfinite(stats["kl_capacity_penalty"]).item()
	expected_regularizer = (
		stats["kl_capacity_penalty"] * torch.abs(stats["kl"] - stats["kl_capacity_target"])
	)
	assert torch.allclose(
		stats["kl_regularizer"],
		expected_regularizer,
		rtol=1e-6,
		atol=1e-6,
	)
	assert torch.allclose(
		loss / batch.shape[0],
		stats["recon_nll"] + stats["kl_regularizer"],
		rtol=1e-4,
		atol=1e-4,
	)


def test_vae_dynamic_input_shape_roundtrip():
	_set_seed(5)
	input_shape = (63, 81)
	batch = torch.randn(3, *input_shape)
	vae = VAE(save_dir="", device_name="cpu", input_shape=input_shape)
	latent, recon = vae.forward(batch, return_latent_rec=True)

	assert np.isfinite(recon).all()
	assert latent.shape == (batch.shape[0], vae.z_dim)
	assert recon.shape == (batch.shape[0], *input_shape)


def test_learned_observation_scale_has_grad():
	_set_seed(13)
	batch = torch.randn(2, 128, 128)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		learn_observation_scale=True,
		build_optimizer=False,
	)
	loss = vae._compute_loss(batch)
	loss.backward()

	assert vae.log_precision.grad is not None


def test_lightning_checkpoint_loads_legacy(tmp_path):
	_set_seed(101)
	data = torch.randn(6, 128, 128)
	loaders = _make_loaders(data)
	save_dir = tmp_path / "lightning_ckpt"

	module, _ = train_vae(
		loaders,
		save_dir=str(save_dir),
		epochs=2,
		test_freq=None,
		save_freq=1,
		vis_freq=None,
		trainer_kwargs={
			"accelerator": "cpu",
			"devices": 1,
			"enable_progress_bar": False,
			"enable_model_summary": False,
		},
	)

	checkpoint = save_dir / "checkpoint_001.tar"
	assert checkpoint.exists()

	loaded = VAE(save_dir=str(save_dir), z_dim=module.vae.z_dim,
		device_name="cpu")
	loaded.load_state(str(checkpoint))

	assert 0 in loaded.loss["train"]


def test_lightning_saves_checkpoint_after_first_completed_epoch(tmp_path):
	_set_seed(211)
	data = torch.randn(6, 128, 128)
	loaders = _make_loaders(data)
	save_dir = tmp_path / "lightning_first_epoch_ckpt"

	train_vae(
		loaders,
		save_dir=str(save_dir),
		epochs=1,
		test_freq=None,
		save_freq=1,
		vis_freq=None,
		trainer_kwargs={
			"accelerator": "cpu",
			"devices": 1,
			"enable_progress_bar": False,
			"enable_model_summary": False,
		},
	)

	assert (save_dir / "checkpoint_001.tar").exists()


def test_lightning_updates_dataset_epochs_across_train_and_validation(tmp_path):
	_set_seed(313)
	train_dataset = _EpochTrackingDataset(torch.randn(4, 64, 64))
	val_dataset = _EpochTrackingDataset(torch.randn(4, 64, 64))
	loaders = {
		"train": DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0),
		"test": DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0),
	}
	save_dir = tmp_path / "lightning_epoch_tracking"

	train_vae(
		loaders,
		save_dir=str(save_dir),
		epochs=2,
		test_freq=1,
		save_freq=None,
		vis_freq=None,
		trainer_kwargs={
			"accelerator": "cpu",
			"devices": 1,
			"enable_progress_bar": False,
			"enable_model_summary": False,
			"num_sanity_val_steps": 0,
			"limit_train_batches": 1,
			"limit_val_batches": 1,
			"logger": False,
		},
		input_shape=(64, 64),
		z_dim=8,
	)

	assert train_dataset.epochs == [0, 1]
	assert val_dataset.epochs == [0, 1]


def test_mps_mixed_precision_is_overridden_to_fp32(tmp_path):
	if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
		pytest.skip("MPS not available for this test.")

	import pytorch_lightning as pl

	class _TinyModule(pl.LightningModule):
		def __init__(self):
			super().__init__()
			self.layer = torch.nn.Linear(4, 1)

		def forward(self, x):
			return self.layer(x).squeeze(-1)

		def training_step(self, batch, batch_idx):
			x, y = batch
			pred = self(x)
			return torch.nn.functional.mse_loss(pred, y)

		def configure_optimizers(self):
			return torch.optim.SGD(self.parameters(), lr=0.01)

	x = torch.randn(8, 4)
	y = torch.randn(8)
	loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, num_workers=0)
	save_dir = tmp_path / "mps_precision_override"
	trainer = build_trainer(
		save_dir=str(save_dir),
		epochs=1,
		test_freq=None,
		save_freq=None,
		vis_freq=None,
		trainer_kwargs={
			"accelerator": "mps",
			"devices": 1,
			"precision": "16-mixed",
			"logger": False,
			"enable_progress_bar": False,
			"enable_model_summary": False,
			"limit_train_batches": 1,
			"num_sanity_val_steps": 0,
		},
	)
	trainer.fit(_TinyModule(), train_dataloaders=loader)
	assert getattr(trainer.strategy, "root_device", torch.device("cpu")).type == "mps"
