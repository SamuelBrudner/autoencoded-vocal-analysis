import random

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from torch.utils.data import DataLoader, Dataset

from ava.models.lightning_vae import VAELightningModule, train_vae
from ava.models.vae import VAE


class _TensorDataset(Dataset):
	def __init__(self, data):
		self._data = data

	def __len__(self):
		return self._data.shape[0]

	def __getitem__(self, idx):
		return self._data[idx]


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
	legacy_loss = vae.forward(batch)

	assert torch.allclose(lightning_loss, legacy_loss, rtol=1e-4, atol=1e-4)
	assert torch.isfinite(stats["recon_mse"]).item()
	assert torch.isfinite(stats["latent_mean_abs"]).item()
	assert torch.isfinite(stats["latent_var_mean"]).item()


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
