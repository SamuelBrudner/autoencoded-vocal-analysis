import random

import numpy as np
import pytest

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter


torch = pytest.importorskip("torch")
patch_torch_onnx_exporter()
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
	recon, _, _, _, _ = vae.forward(batch)

	assert recon.shape == batch.shape
	assert torch.isfinite(stats["recon_mse"]).item()
	assert torch.isfinite(stats["recon_nll"]).item()
	assert torch.isfinite(stats["kl"]).item()
	assert torch.isfinite(stats["kl_weight"]).item()
	assert torch.isfinite(stats["log_precision"]).item()
	assert torch.isfinite(stats["model_precision"]).item()
	assert stats["model_precision"].item() > 0
	assert torch.isfinite(stats["latent_mean_abs"]).item()
	assert torch.isfinite(stats["latent_var_mean"]).item()
	expected = stats["recon_nll"] + stats["kl_weight"] * stats["kl"]
	assert torch.allclose(
		lightning_loss / batch.shape[0],
		expected,
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
