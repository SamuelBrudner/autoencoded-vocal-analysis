import pytest


torch = pytest.importorskip("torch")

from ava.models.vae import VAE


@pytest.mark.parametrize("input_shape", [(128, 128), (63, 81)])
def test_residual_roundtrip_backward(input_shape):
	batch = torch.randn(2, *input_shape)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		input_shape=input_shape,
		conv_arch="residual",
		build_optimizer=False,
	)
	recon, _, _, _, _ = vae(batch)

	assert recon.shape == batch.shape
	loss = (recon - batch).pow(2).mean()
	loss.backward()
	assert any(
		param.grad is not None
		for param in vae.parameters()
		if param.requires_grad
	)


@pytest.mark.parametrize("conv_arch", ["plain", "residual"])
@pytest.mark.parametrize("input_shape", [(128, 128), (63, 81)])
def test_shape_agnostic_roundtrip(conv_arch, input_shape):
	batch = torch.randn(2, *input_shape)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		input_shape=input_shape,
		conv_arch=conv_arch,
		build_optimizer=False,
	)
	recon, _, _, _, _ = vae(batch)

	assert recon.shape == batch.shape
