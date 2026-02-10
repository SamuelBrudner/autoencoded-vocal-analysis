import math

import pytest


torch = pytest.importorskip("torch")

from ava.models.vae import VAE  # noqa: E402


def test_log_precision_grad_matches_gaussian_likelihood_derivative():
	torch.manual_seed(0)
	batch = torch.randn(3, 16, 16)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		input_shape=(16, 16),
		z_dim=4,
		model_precision=3.0,
		learn_observation_scale=True,
		build_optimizer=False,
	)

	recon, mu, logvar, z, u = vae(batch)
	latent_dist = vae._posterior_distribution(mu, logvar, u)
	x_flat = batch.view(batch.shape[0], -1)
	recon_flat = recon.view(batch.shape[0], -1)
	log_precision = vae._log_precision_tensor()
	precision = vae._precision_tensor()

	log_two_pi = math.log(2 * math.pi)
	pxz_term = -0.5 * x_flat.shape[1] * (log_two_pi - log_precision)
	l2s = torch.sum((x_flat - recon_flat) ** 2, dim=1)
	sse_total = torch.sum(l2s)
	pxz_term = pxz_term - 0.5 * precision * sse_total

	log_pz = -0.5 * (torch.sum(z ** 2) + vae.z_dim * math.log(2 * math.pi))
	entropy = torch.sum(latent_dist.entropy())
	loss = -(pxz_term + (log_pz + entropy))

	loss.backward()

	expected = 0.5 * precision.detach() * sse_total.detach() - 0.5 * (
		batch.shape[0] * x_flat.shape[1]
	)
	assert vae.log_precision.grad is not None
	assert torch.allclose(
		vae.log_precision.grad,
		expected.to(dtype=vae.log_precision.grad.dtype),
		rtol=1e-5,
		atol=1e-5,
	)


def test_log_precision_clamping_keeps_loss_finite():
	torch.manual_seed(0)
	batch = torch.randn(2, 16, 16)
	vae = VAE(
		save_dir="",
		device_name="cpu",
		input_shape=(16, 16),
		z_dim=4,
		model_precision=1.0,
		learn_observation_scale=True,
		build_optimizer=False,
		log_precision_min=-4.0,
		log_precision_max=4.0,
	)

	with torch.no_grad():
		vae.log_precision.copy_(vae.log_precision.new_tensor(100.0))
	high_loss = vae._compute_loss(batch)
	assert torch.isfinite(high_loss)
	assert torch.allclose(
		vae._log_precision_tensor().detach(),
		vae.log_precision.new_tensor(4.0),
	)
	assert abs(vae.model_precision - math.exp(4.0)) < 1e-6

	with torch.no_grad():
		vae.log_precision.copy_(vae.log_precision.new_tensor(-100.0))
	low_loss = vae._compute_loss(batch)
	assert torch.isfinite(low_loss)
	assert torch.allclose(
		vae._log_precision_tensor().detach(),
		vae.log_precision.new_tensor(-4.0),
	)
	assert abs(vae.model_precision - math.exp(-4.0)) < 1e-6

