import pytest


torch = pytest.importorskip("torch")

from ava.models.vae import VAE


INPUT_SHAPE = (128, 128)


def _build_vae(input_shape=INPUT_SHAPE):
	torch.manual_seed(0)
	return VAE(
		save_dir="",
		device_name="cpu",
		input_shape=input_shape,
		decoder_type="upsample",
		z_dim=8,
		build_optimizer=False,
	)



def test_decoder_decode_output_is_finite_and_shaped():
	vae = _build_vae()
	torch.manual_seed(1)
	z = torch.randn(2, vae.z_dim)
	decoded = vae.decode(z)

	assert decoded.shape == (2, vae.input_dim)
	decoded_view = decoded.view(-1, *vae.input_shape)
	assert decoded_view.shape == (2, *vae.input_shape)
	assert torch.isfinite(decoded_view).all().item()



def test_decoder_loss_is_finite_and_repeatable():
	vae = _build_vae()
	torch.manual_seed(2)
	batch = torch.randn(2, *vae.input_shape)

	torch.manual_seed(3)
	loss1 = vae._compute_loss(batch)
	torch.manual_seed(3)
	loss2 = vae._compute_loss(batch)

	assert torch.isfinite(loss1).item()
	assert torch.isfinite(loss2).item()
	assert loss1.item() < 1e7
	assert torch.allclose(loss1, loss2, rtol=0.0, atol=1e-6)
