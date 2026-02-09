"""
A Variational Autoencoder (VAE) for spectrogram data.

VAE References
--------------
.. [1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
	arXiv preprint arXiv:1312.6114 (2013).

	`<https://arxiv.org/abs/1312.6114>`_


.. [2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
	backpropagation and approximate inference in deep generative models." arXiv
	preprint arXiv:1401.4082 (2014).

	`<https://arxiv.org/abs/1401.4082>`_
"""
__date__ = "November 2018 - November 2019"


import math
import numpy as np
import os
from types import SimpleNamespace

try:
	import torch
	from torch.distributions import LowRankMultivariateNormal
	import torch.nn as nn
	import torch.nn.functional as F
	from torch.optim import Adam
except ImportError as exc:  # pragma: no cover - optional in some envs
	torch = None
	LowRankMultivariateNormal = None
	Adam = None
	_TORCH_IMPORT_ERROR = exc
	nn = SimpleNamespace(Module=object)
	F = None
else:
	_TORCH_IMPORT_ERROR = None

from ava.plotting.grid_plot import grid_plot


DEFAULT_INPUT_SHAPE = (128, 128)
"""Default processed spectrogram shape: ``[freq_bins, time_bins]``."""
X_SHAPE = DEFAULT_INPUT_SHAPE
"""Legacy alias for the default processed spectrogram shape."""
X_DIM = int(np.prod(X_SHAPE))
"""Legacy default spectrogram dimension: ``freq_bins * time_bins``."""


class ResBlock2D(nn.Module):
	"""Residual block with Conv-Norm-Act + skip."""

	def __init__(self, in_channels, out_channels, stride=1,
		norm_factory=None):
		super().__init__()
		if norm_factory is None:
			raise ValueError("norm_factory is required for ResBlock2D.")
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
		self.norm1 = norm_factory(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1)
		self.norm2 = norm_factory(out_channels)
		self.act = nn.SiLU()
		if stride != 1 or in_channels != out_channels:
			self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
		else:
			self.skip = nn.Identity()

	def forward(self, x):
		residual = self.skip(x)
		out = self.act(self.norm1(self.conv1(x)))
		out = self.norm2(self.conv2(out))
		return self.act(out + residual)


class ResBlockUp2D(nn.Module):
	"""Residual block with Upsample-Conv-Norm-Act + skip."""

	def __init__(self, in_channels, out_channels, norm_factory=None):
		super().__init__()
		if norm_factory is None:
			raise ValueError("norm_factory is required for ResBlockUp2D.")
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
		self.norm1 = norm_factory(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
		self.norm2 = norm_factory(out_channels)
		self.act = nn.SiLU()
		if in_channels != out_channels:
			self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
		else:
			self.skip = nn.Identity()

	def forward(self, x, size):
		upsampled = F.interpolate(x, size=size, mode="nearest")
		residual = self.skip(upsampled)
		out = self.act(self.norm1(self.conv1(upsampled)))
		out = self.norm2(self.conv2(out))
		return self.act(out + residual)



class VAE(nn.Module):
	"""Variational Autoencoder class for single-channel images.

	Attributes
	----------
	save_dir : str, optional
		Directory where the model is saved. Defaults to ``''``.
	lr : float, optional
		Model learning rate. Defaults to ``1e-3``.
	z_dim : int, optional
		Latent dimension. Defaults to ``32``.
	model_precision : float, optional
		Initial precision of the observation model. Defaults to ``10.0``.
	learn_observation_scale : bool, optional
		Whether to learn the observation precision instead of keeping it fixed.
		Defaults to ``False``.
	device_name : {'cpu', 'cuda', 'auto'}, optional
		Name of device to train the model on. When ``'auto'`` is passed,
		``'cuda'`` is chosen if ``torch.cuda.is_available()``, otherwise
		``'cpu'`` is chosen. Defaults to ``'auto'``.
	input_shape : tuple of int, optional
		Expected spectrogram shape as ``(freq_bins, time_bins)``.
	posterior_type : {"diag", "lowrank"}, optional
		Posterior covariance structure.
	conv_arch : {"plain", "residual"}, optional
		Convolutional stack architecture. ``"plain"`` uses the legacy
		Conv-Norm-Act blocks; ``"residual"`` adds ResNet-style skip
		connections.

	Notes
	-----
	The model is trained to maximize the standard ELBO objective:

	.. math:: \mathcal{L} = \mathbb{E}_{q(z|x)} log p(x,z) + \mathbb{H}[q(z|x)]

	where :math:`p(x,z) = p(z)p(x|z)` and :math:`\mathbb{H}` is differential
	entropy. The prior :math:`p(z)` is a unit spherical normal distribution. The
	conditional distribution :math:`p(x|z)` is set as a spherical normal
	distribution to prevent overfitting. The variational distribution,
	:math:`q(z|x)` is modeled as a diagonal Gaussian by default for speed, with
	an optional low-rank-plus-diagonal covariance. Here, :math:`q(z|x)` and
	:math:`p(x|z)` are parameterized by neural networks. Gradients are passed
	through stochastic layers via the reparameterization trick, implemented by
	the PyTorch `rsample` method.

	The convolutional stack downsamples the input three times with stride-2
	convolutions. The expected spectrogram shape is configurable via
	``input_shape``, which drives the linear layer sizes and decoder upsampling
	so reconstructions match the input dimensions.
	"""

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0,
		learn_observation_scale=False, device_name="auto",
		input_shape=X_SHAPE, posterior_type="diag", conv_arch="plain",
		build_optimizer=True):
		"""Construct a VAE.

		Parameters
		----------
		save_dir : str, optional
			Directory where the model is saved. Defaults to the current working
			directory.
		lr : float, optional
			Learning rate of the ADAM optimizer. Defaults to 1e-3.
		z_dim : int, optional
			Dimension of the latent space. Defaults to 32.
		model_precision : float, optional
			Initial precision of the noise model, p(x|z) = N(mu(z), \Lambda)
			where \Lambda = model_precision * I. Defaults to 10.0.
		learn_observation_scale : bool, optional
			Whether to learn the observation precision instead of keeping it
			fixed. Defaults to ``False``.
		device_name: str, optional
			Name of device to train the model on. Valid options are ["cpu",
			"cuda", "auto"]. "auto" will choose "cuda" if it is available.
			Defaults to "auto".
		input_shape : tuple of int, optional
			Expected spectrogram shape as ``(freq_bins, time_bins)``. Defaults
			to ``X_SHAPE``.
		posterior_type : {"diag", "lowrank"}, optional
			Posterior covariance structure. ``"diag"`` uses a diagonal Gaussian
			parameterized by ``(mu, logvar)`` for speed, while ``"lowrank"``
			retains the legacy low-rank-plus-diagonal posterior. Defaults to
			``"diag"``.
		conv_arch : {"plain", "residual"}, optional
			Convolutional stack architecture. Defaults to ``"plain"``.
		build_optimizer : bool, optional
			Whether to construct the Adam optimizer during initialization.
			Defaults to ``True``.

		Note
		----
		- The model is built before it's parameters can be loaded from a file.
			This means `self.z_dim` must match `z_dim` of the model being
			loaded, and `input_shape` must match the checkpoint configuration.
		"""
		if _TORCH_IMPORT_ERROR is not None:
			raise ImportError(
				"PyTorch is required for ava.models.vae. "
				"Install with `pip install torch`."
			) from _TORCH_IMPORT_ERROR
		super(VAE, self).__init__()
		self.save_dir = save_dir
		self.lr = lr
		self.z_dim = z_dim
		self.learn_observation_scale = bool(learn_observation_scale)
		model_precision = self._coerce_positive_float(
			model_precision, "model_precision"
		)
		log_precision = math.log(model_precision)
		log_precision_tensor = torch.tensor(
			log_precision, dtype=torch.get_default_dtype()
		)
		if self.learn_observation_scale:
			self.log_precision = nn.Parameter(log_precision_tensor)
		else:
			self.register_buffer("log_precision", log_precision_tensor)
		self.input_shape = self._normalize_input_shape(input_shape)
		self.input_dim = int(np.prod(self.input_shape))
		self.posterior_type = self._normalize_posterior_type(posterior_type)
		self.conv_arch = self._normalize_conv_arch(conv_arch)
		assert device_name != "cuda" or torch.cuda.is_available()
		if device_name == "auto":
			device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self._requested_device = torch.device(device_name)
		if self.save_dir != '' and not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self._build_network()
		self.optimizer = None
		if build_optimizer:
			self._ensure_optimizer()
		self.epoch = 0
		self.loss = {'train':{}, 'test':{}}
		self.to(self._requested_device)


	@property
	def device(self):
		return next(self.parameters()).device


	@property
	def model_precision(self):
		"""Return the current observation precision as a float."""
		return float(torch.exp(self.log_precision).detach().cpu())


	@model_precision.setter
	def model_precision(self, value):
		value = self._coerce_positive_float(value, "model_precision")
		log_precision = math.log(value)
		if not hasattr(self, "log_precision"):
			raise AttributeError("log_precision has not been initialized.")
		with torch.no_grad():
			self.log_precision.copy_(
				self.log_precision.new_tensor(log_precision)
			)


	def _precision_tensor(self):
		return torch.exp(self.log_precision)


	@staticmethod
	def _normalize_input_shape(input_shape):
		if input_shape is None or len(input_shape) != 2:
			raise ValueError(
				"input_shape must be a tuple of (freq_bins, time_bins)."
			)
		try:
			shape = (int(input_shape[0]), int(input_shape[1]))
		except (TypeError, ValueError) as exc:
			raise ValueError("input_shape must contain two integers.") from exc
		if shape[0] <= 0 or shape[1] <= 0:
			raise ValueError("input_shape values must be positive.")
		return shape


	@staticmethod
	def _normalize_posterior_type(posterior_type):
		if posterior_type is None:
			return "diag"
		value = str(posterior_type).strip().lower()
		if value in {"diag", "diagonal", "diagonal_gaussian"}:
			return "diag"
		if value in {"lowrank", "low_rank", "low-rank"}:
			return "lowrank"
		raise ValueError("posterior_type must be 'diag' or 'lowrank'.")


	@staticmethod
	def _normalize_conv_arch(conv_arch):
		if conv_arch is None:
			return "plain"
		value = str(conv_arch).strip().lower()
		if value in {"plain", "legacy", "conv"}:
			return "plain"
		if value in {"residual", "resnet", "res"}:
			return "residual"
		raise ValueError("conv_arch must be 'plain' or 'residual'.")


	@staticmethod
	def _coerce_positive_float(value, name):
		try:
			value = float(value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"{name} must be a positive float.") from exc
		if not math.isfinite(value) or value <= 0:
			raise ValueError(f"{name} must be a positive float.")
		return value


	@staticmethod
	def _make_norm(num_channels, max_groups=8):
		groups = min(max_groups, num_channels)
		while groups > 1 and num_channels % groups != 0:
			groups -= 1
		return nn.GroupNorm(groups, num_channels)


	@staticmethod
	def _downsample_dim(size):
		return (size + 1) // 2


	@classmethod
	def _compute_downsample_shapes(cls, input_shape):
		height, width = input_shape
		shapes = [(height, width)]
		for _ in range(3):
			height = cls._downsample_dim(height)
			width = cls._downsample_dim(width)
			shapes.append((height, width))
		return shapes


	def _infer_conv_shapes(self):
		shapes = [self.input_shape]
		with torch.no_grad():
			x = torch.zeros(1, 1, *self.input_shape)
			if self.conv_arch == "plain":
				x = self._act(self.bn1(self.conv1(x)))
				x = self._act(self.bn2(self.conv2(x)))
				shapes.append((x.shape[2], x.shape[3]))
				x = self._act(self.bn3(self.conv3(x)))
				x = self._act(self.bn4(self.conv4(x)))
				shapes.append((x.shape[2], x.shape[3]))
				x = self._act(self.bn5(self.conv5(x)))
				x = self._act(self.bn6(self.conv6(x)))
				shapes.append((x.shape[2], x.shape[3]))
				x = self._act(self.bn7(self.conv7(x)))
			else:
				for block in self.enc_blocks:
					x = block(x)
					shapes.append((x.shape[2], x.shape[3]))
				x = self.enc_out(x)
		conv_feature_shape = (x.shape[1], x.shape[2], x.shape[3])
		return shapes, conv_feature_shape


	def _check_input(self, x):
		if x.dim() != 3:
			raise ValueError("Input must have shape [batch, height, width].")
		if tuple(x.shape[1:]) != self.input_shape:
			raise ValueError(
				f"Expected input shape {self.input_shape}, got "
				f"{tuple(x.shape[1:])}."
			)


	def _posterior_distribution(self, mu, logvar, u):
		if self.posterior_type == "diag":
			scale = torch.exp(0.5 * logvar)
			base_dist = torch.distributions.Normal(mu, scale)
			return torch.distributions.Independent(base_dist, 1)
		if self.posterior_type == "lowrank":
			if u is None:
				raise ValueError("Low-rank posterior requires u.")
			d = torch.exp(logvar)
			return LowRankMultivariateNormal(mu, u, d)
		raise ValueError(f"Unknown posterior_type '{self.posterior_type}'.")


	def _latent_variance_mean(self, u, logvar):
		if self.posterior_type == "diag":
			return torch.exp(logvar).mean()
		return (u.squeeze(-1) ** 2 + torch.exp(logvar)).mean()


	def _ensure_optimizer(self):
		if self.optimizer is None:
			self.optimizer = Adam(self.parameters(), lr=self.lr)


	@staticmethod
	def _upsample_to(x, size):
		return F.interpolate(x, size=size, mode="nearest")


	def _build_network(self):
		"""Define all the network layers."""
		self._act = nn.SiLU()
		if self.conv_arch == "plain":
			# Encoder (Conv-Norm-Act)
			self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
			self.bn1 = self._make_norm(8)
			self.conv2 = nn.Conv2d(8, 8, 3, 2, padding=1)
			self.bn2 = self._make_norm(8)
			self.conv3 = nn.Conv2d(8, 16, 3, 1, padding=1)
			self.bn3 = self._make_norm(16)
			self.conv4 = nn.Conv2d(16, 16, 3, 2, padding=1)
			self.bn4 = self._make_norm(16)
			self.conv5 = nn.Conv2d(16, 24, 3, 1, padding=1)
			self.bn5 = self._make_norm(24)
			self.conv6 = nn.Conv2d(24, 24, 3, 2, padding=1)
			self.bn6 = self._make_norm(24)
			self.conv7 = nn.Conv2d(24, 32, 3, 1, padding=1)
			self.bn7 = self._make_norm(32)
		else:
			self.enc_blocks = nn.ModuleList([
				ResBlock2D(1, 8, stride=2, norm_factory=self._make_norm),
				ResBlock2D(8, 16, stride=2, norm_factory=self._make_norm),
				ResBlock2D(16, 24, stride=2, norm_factory=self._make_norm),
			])
			self.enc_out = nn.Sequential(
				nn.Conv2d(24, 32, 3, 1, padding=1),
				self._make_norm(32),
				nn.SiLU(),
			)

		self._conv_shapes, self._conv_feature_shape = self._infer_conv_shapes()
		self._conv_feature_dim = int(np.prod(self._conv_feature_shape))
		self._decoder_upsample_shapes = list(
			reversed(self._conv_shapes[:-1])
		)

		self.fc1 = nn.Linear(self._conv_feature_dim, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc31 = nn.Linear(256, 64)
		self.fc32 = nn.Linear(256, 64)
		self.fc33 = nn.Linear(256, 64)
		self.fc41 = nn.Linear(64, self.z_dim)
		self.fc42 = nn.Linear(64, self.z_dim)
		self.fc43 = nn.Linear(64, self.z_dim)

		# Decoder (Upsample + Conv mirrors encoder)
		self.fc5 = nn.Linear(self.z_dim, 64)
		self.fc6 = nn.Linear(64, 256)
		self.fc7 = nn.Linear(256, 1024)
		self.fc8 = nn.Linear(1024, self._conv_feature_dim)
		if self.conv_arch == "plain":
			self.convt1 = nn.Conv2d(32, 24, 3, 1, padding=1)
			self.convt2 = nn.Conv2d(24, 24, 3, 1, padding=1)
			self.convt3 = nn.Conv2d(24, 16, 3, 1, padding=1)
			self.convt4 = nn.Conv2d(16, 16, 3, 1, padding=1)
			self.convt5 = nn.Conv2d(16, 8, 3, 1, padding=1)
			self.convt6 = nn.Conv2d(8, 8, 3, 1, padding=1)
			self.convt7 = nn.Conv2d(8, 1, 3, 1, padding=1)
			self.bn8 = self._make_norm(24)
			self.bn9 = self._make_norm(24)
			self.bn10 = self._make_norm(16)
			self.bn11 = self._make_norm(16)
			self.bn12 = self._make_norm(8)
			self.bn13 = self._make_norm(8)
			self.bn14 = nn.Identity()
		else:
			self.dec_blocks = nn.ModuleList([
				ResBlockUp2D(32, 24, norm_factory=self._make_norm),
				ResBlockUp2D(24, 16, norm_factory=self._make_norm),
				ResBlockUp2D(16, 8, norm_factory=self._make_norm),
			])
			self.dec_out = nn.Conv2d(8, 1, 3, 1, padding=1)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		layers = {
			'fc1': self.fc1, 'fc2': self.fc2, 'fc31': self.fc31,
			'fc32': self.fc32, 'fc33': self.fc33, 'fc41': self.fc41,
			'fc42': self.fc42, 'fc43': self.fc43, 'fc5': self.fc5,
			'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
		}
		if self.conv_arch == "plain":
			layers.update({
				'bn1': self.bn1, 'bn2': self.bn2, 'bn3': self.bn3,
				'bn4': self.bn4, 'bn5': self.bn5, 'bn6': self.bn6,
				'bn7': self.bn7, 'bn8': self.bn8, 'bn9': self.bn9,
				'bn10': self.bn10, 'bn11': self.bn11, 'bn12': self.bn12,
				'bn13': self.bn13, 'bn14': self.bn14,
				'conv1': self.conv1, 'conv2': self.conv2, 'conv3': self.conv3,
				'conv4': self.conv4, 'conv5': self.conv5, 'conv6': self.conv6,
				'conv7': self.conv7, 'convt1': self.convt1,
				'convt2': self.convt2, 'convt3': self.convt3,
				'convt4': self.convt4, 'convt5': self.convt5,
				'convt6': self.convt6, 'convt7': self.convt7,
			})
		else:
			layers.update({
				'enc_blocks': self.enc_blocks,
				'enc_out': self.enc_out,
				'dec_blocks': self.dec_blocks,
				'dec_out': self.dec_out,
			})
		return layers


	def encode(self, x):
		"""
		Compute :math:`q(z|x)`.

		.. math:: q(z|x) = \mathcal{N}(\mu, \Sigma)
		.. math:: \Sigma = u u^{T} + \mathtt{diag}(\exp(\log \sigma^2))

		where :math:`\mu`, :math:`u`, and :math:`d` are deterministic functions
		of `x` and :math:`\Sigma` denotes a covariance matrix.

		Parameters
		----------
		x : torch.Tensor
			The input images, with shape: ``[batch_size, height, width]`` where
			``(height, width)`` matches ``self.input_shape``.

		Returns
		-------
		mu : torch.Tensor
			Posterior mean, with shape ``[batch_size, self.z_dim]``
		logvar : torch.Tensor
			Posterior log-variance, with shape ``[batch_size, self.z_dim]``
		u : torch.Tensor or None
			Posterior covariance factor, as defined above. Shape:
			``[batch_size, self.z_dim, 1]``. For diagonal posteriors, this is
			returned as ``None``.
		"""
		self._check_input(x)
		x = x.unsqueeze(1)
		if self.conv_arch == "plain":
			x = self._act(self.bn1(self.conv1(x)))
			x = self._act(self.bn2(self.conv2(x)))
			x = self._act(self.bn3(self.conv3(x)))
			x = self._act(self.bn4(self.conv4(x)))
			x = self._act(self.bn5(self.conv5(x)))
			x = self._act(self.bn6(self.conv6(x)))
			x = self._act(self.bn7(self.conv7(x)))
		else:
			for block in self.enc_blocks:
				x = block(x)
			x = self.enc_out(x)
		x = x.view(x.shape[0], -1)
		x = self._act(self.fc1(x))
		x = self._act(self.fc2(x))
		mu = self._act(self.fc31(x))
		mu = self.fc41(mu)
		if self.posterior_type == "lowrank":
			u = self._act(self.fc32(x))
			u = self.fc42(u).unsqueeze(-1) # Last dimension is rank \Sigma = 1.
		else:
			u = None
		logvar = self._act(self.fc33(x))
		logvar = self.fc43(logvar)
		return mu, logvar, u


	def decode(self, z):
		"""
		Compute :math:`p(x|z)`.

		.. math:: p(x|z) = \mathcal{N}(\mu, \Lambda)

		.. math:: \Lambda = \mathtt{model\_precision} \cdot I

		where :math:`\mu` is a deterministic function of `z`, :math:`\Lambda` is
		a precision matrix, and :math:`I` is the identity matrix. When
		``learn_observation_scale=True``, :math:`\mathtt{model\_precision}` is
		learned during training.

		Parameters
		----------
		z : torch.Tensor
			Batch of latent samples with shape ``[batch_size, self.z_dim]``

		Returns
		-------
		x : torch.Tensor
			Batch of means mu, described above. Shape: ``[batch_size,
			self.input_dim]``
		"""
		z = self._act(self.fc5(z))
		z = self._act(self.fc6(z))
		z = self._act(self.fc7(z))
		z = self._act(self.fc8(z))
		z = z.view(-1, *self._conv_feature_shape)
		if self.conv_arch == "plain":
			z = self._act(self.bn8(self.convt1(z)))
			z = self._upsample_to(z, self._decoder_upsample_shapes[0])
			z = self._act(self.bn9(self.convt2(z)))
			z = self._act(self.bn10(self.convt3(z)))
			z = self._upsample_to(z, self._decoder_upsample_shapes[1])
			z = self._act(self.bn11(self.convt4(z)))
			z = self._act(self.bn12(self.convt5(z)))
			z = self._upsample_to(z, self._decoder_upsample_shapes[2])
			z = self._act(self.bn13(self.convt6(z)))
			z = self.convt7(z)
		else:
			for block, size in zip(self.dec_blocks, self._decoder_upsample_shapes):
				z = block(z, size)
			z = self.dec_out(z)
		return z.view(-1, self.input_dim)


	def forward(self, x, return_latent_rec=False):
		"""
		Send `x` round trip and return reconstructions plus latent statistics.

		Parameters
		----------
		x : torch.Tensor
			A batch of samples from the data distribution (spectrograms).
			Shape: ``[batch_size, height, width]`` where ``(height, width)``
			matches ``self.input_shape``.
		return_latent_rec : bool, optional
			Whether to return latent samples and reconstructions as numpy arrays.
			Defaults to ``False``.

		Returns
		-------
		reconstructions : torch.Tensor
			Reconstructed means. Shape: ``[batch_size, height, width]``.
		mu : torch.Tensor
			Posterior mean. Shape: ``[batch_size, self.z_dim]``.
		logvar : torch.Tensor
			Posterior log-variance. Shape: ``[batch_size, self.z_dim]``.
		z : torch.Tensor
			Latent samples. Shape: ``[batch_size, self.z_dim]``.
		u : torch.Tensor or None
			Low-rank covariance factor when ``posterior_type="lowrank"``.
		"""
		mu, logvar, u = self.encode(x)
		latent_dist = self._posterior_distribution(mu, logvar, u)
		z = latent_dist.rsample()
		x_rec = self.decode(z).view(-1, *self.input_shape)
		if return_latent_rec:
			return (
				z.detach().cpu().numpy(),
				x_rec.detach().cpu().numpy(),
			)
		return x_rec, mu, logvar, z, u


	def _compute_loss(self, x, beta=1.0):
		"""Compute the negative ELBO for legacy training loops."""
		recon, mu, logvar, z, u = self.forward(x)
		latent_dist = self._posterior_distribution(mu, logvar, u)
		x_flat = x.view(x.shape[0], -1)
		recon_flat = recon.view(x.shape[0], -1)
		log_precision = self.log_precision
		precision = self._precision_tensor()
		log_two_pi = math.log(2 * math.pi)
		pxz_term = -0.5 * x_flat.shape[1] * (
			log_two_pi - log_precision
		)
		l2s = torch.sum(torch.pow(x_flat - recon_flat, 2), dim=1)
		pxz_term = pxz_term - 0.5 * precision * torch.sum(l2s)
		log_pz = -0.5 * (
			torch.sum(torch.pow(z, 2)) + self.z_dim * np.log(2 * np.pi)
		)
		entropy = torch.sum(latent_dist.entropy())
		return -(pxz_term + beta * (log_pz + entropy))

	def _set_dataset_epoch(self, dataset, epoch):
		if hasattr(dataset, "set_epoch"):
			dataset.set_epoch(epoch)
			return
		nested = getattr(dataset, "dataset", None)
		if nested is not None:
			self._set_dataset_epoch(nested, epoch)

	def _set_loader_epoch(self, loader, epoch):
		if loader is None:
			return
		dataset = getattr(loader, "dataset", None)
		if dataset is None:
			return
		self._set_dataset_epoch(dataset, epoch)

	def train_epoch(self, train_loader):
		"""
		Train the model for a single epoch.

		Parameters
		----------
		train_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for training set

		Returns
		-------
		elbo : float
			A biased estimate of the ELBO, estimated using samples from
			`train_loader`.
		"""
		self.train()
		self._ensure_optimizer()
		train_loss = 0.0
		for batch_idx, data in enumerate(train_loader):
			self.optimizer.zero_grad()
			data = data.to(self.device)
			loss = self._compute_loss(data)
			train_loss += loss.item()
			loss.backward()
			self.optimizer.step()
		train_loss /= len(train_loader.dataset)
		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		self.epoch += 1
		return train_loss


	def test_epoch(self, test_loader):
		"""
		Test the model on a held-out test set, return an ELBO estimate.

		Parameters
		----------
		test_loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader for test set

		Returns
		-------
		elbo : float
			An unbiased estimate of the ELBO, estimated using samples from
			`test_loader`.
		"""
		self.eval()
		test_loss = 0.0
		with torch.no_grad():
			for i, data in enumerate(test_loader):
				data = data.to(self.device)
				loss = self._compute_loss(data)
				test_loss += loss.item()
		test_loss /= len(test_loader.dataset)
		print('Test loss: {:.4f}'.format(test_loss))
		return test_loss


	def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys ``'test'`` and ``'train'`` to respective
			torch.utils.data.Dataloader objects.
		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to ``100``.
		test_freq : int, optional
			Testing is performed every `test_freq` epochs. Defaults to ``2``.
		save_freq : int, optional
			The model is saved every `save_freq` epochs. Defaults to ``10``.
		vis_freq : int, optional
			Syllable reconstructions are plotted every `vis_freq` epochs.
			Defaults to ``1``.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Training set:", len(loaders['train'].dataset))
		print("Test set:", len(loaders['test'].dataset))
		print("="*40)
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			self._set_loader_epoch(loaders.get('train'), epoch)
			self._set_loader_epoch(loaders.get('test'), epoch)
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if (test_freq is not None) and (epoch % test_freq == 0):
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if (save_freq is not None) and (epoch % save_freq == 0) and \
					(epoch > 0):
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state(filename)
			# Plot reconstructions.
			if (vis_freq is not None) and (epoch % vis_freq == 0):
				self.visualize(loaders['test'])


	def save_state(self, filename):
		"""Save all the model parameters to the given file."""
		layers = self._get_layers()
		state = {}
		for layer_name in layers:
			state[layer_name] = layers[layer_name].state_dict()
		self._ensure_optimizer()
		state['optimizer_state'] = self.optimizer.state_dict()
		state['loss'] = self.loss
		state['z_dim'] = self.z_dim
		state['input_shape'] = self.input_shape
		state['posterior_type'] = self.posterior_type
		state['epoch'] = self.epoch
		state['lr'] = self.lr
		state['save_dir'] = self.save_dir
		state['conv_arch'] = self.conv_arch
		state['model_precision'] = self.model_precision
		state['log_precision'] = float(self.log_precision.detach().cpu())
		state['learn_observation_scale'] = self.learn_observation_scale
		state['decoder_type'] = "upsample"
		filename = os.path.join(self.save_dir, filename)
		torch.save(state, filename)


	def load_state(self, filename):
		"""
		Load all the model parameters from the given ``.tar`` file.

		The ``.tar`` file should be written by `self.save_state`.

		Parameters
		----------
		filename : str
			File containing a model state.

		Note
		----
		- `self.lr` and `self.save_dir` are not loaded. `input_shape` is
		  validated against the checkpoint when available.
		"""
		checkpoint = torch.load(filename, map_location=self.device)
		assert checkpoint['z_dim'] == self.z_dim
		if 'input_shape' in checkpoint:
			if tuple(checkpoint['input_shape']) != self.input_shape:
				raise ValueError(
					"Checkpoint input_shape "
					f"{tuple(checkpoint['input_shape'])} does not match "
					f"model input_shape {self.input_shape}."
				)
		checkpoint_arch = checkpoint.get('conv_arch', 'plain')
		if checkpoint_arch != self.conv_arch:
			raise ValueError(
				"Checkpoint conv_arch "
				f"{checkpoint_arch!r} does not match model conv_arch "
				f"{self.conv_arch!r}."
			)
		checkpoint_learned = checkpoint.get("learn_observation_scale")
		if (checkpoint_learned is not None
				and checkpoint_learned != self.learn_observation_scale):
			raise ValueError(
				"Checkpoint learn_observation_scale "
				f"{checkpoint_learned!r} does not match model setting "
				f"{self.learn_observation_scale!r}."
			)
		checkpoint_decoder = checkpoint.get("decoder_type", "convtranspose")
		if checkpoint_decoder != "upsample":
			raise ValueError(
				"Checkpoint decoder_type "
				f"{checkpoint_decoder!r} is incompatible with the current "
				"upsample decoder. Please retrain or export a compatible "
				"checkpoint."
			)
		if 'posterior_type' in checkpoint:
			self.posterior_type = checkpoint['posterior_type']
		else:
			self.posterior_type = "lowrank"
		if "log_precision" in checkpoint:
			with torch.no_grad():
				self.log_precision.copy_(
					self.log_precision.new_tensor(checkpoint["log_precision"])
				)
		elif "model_precision" in checkpoint:
			self.model_precision = checkpoint["model_precision"]
		layers = self._get_layers()
		for layer_name in layers:
			layer = layers[layer_name]
			layer.load_state_dict(checkpoint[layer_name])
		self._ensure_optimizer()
		self.optimizer.load_state_dict(checkpoint['optimizer_state'])
		self.loss = checkpoint['loss']
		self.epoch = checkpoint['epoch']


	def visualize(self, loader, num_specs=5, gap=(2,6), \
		save_filename='reconstruction.pdf'):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader
		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to ``5``.
		gap : int or tuple of two ints, optional
			The vertical and horizontal gap between images, in pixels. Defaults
			to ``(2,6)``.
		save_filename : str, optional
			Where to save the plot, relative to `self.save_dir`. Defaults to
			``'temp.pdf'``.

		Returns
		-------
		specs : numpy.ndarray
			Spectgorams from `loader`.
		rec_specs : numpy.ndarray
			Corresponding spectrogram reconstructions.
		"""
		# Collect random indices.
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		# Retrieve spectrograms from the loader.
		specs = torch.stack(loader.dataset[indices]).to(self.device)
		# Get resonstructions.
		with torch.no_grad():
			_, rec_specs = self.forward(specs, return_latent_rec=True)
		specs = specs.detach().cpu().numpy()
		all_specs = np.stack([specs, rec_specs])
		# Plot.
		save_filename = os.path.join(self.save_dir, save_filename)
		grid_plot(all_specs, gap=gap, filename=save_filename)
		return specs, rec_specs


	def get_latent(self, loader):
		"""
		Get latent means for all syllable in the given loader.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			ava.models.vae_dataset.SyllableDataset Dataloader.

		Returns
		-------
		latent : numpy.ndarray
			Latent means. Shape: ``[len(loader.dataset), self.z_dim]``

		Note
		----
		- Make sure your loader is not set to shuffle if you're going to match
		  these with labels or other fields later.
		"""
		latent = np.zeros((len(loader.dataset), self.z_dim))
		i = 0
		for data in loader:
			data = data.to(self.device)
			with torch.no_grad():
				mu, _, _ = self.encode(data)
			mu = mu.detach().cpu().numpy()
			latent[i:i+len(mu)] = mu
			i += len(mu)
		return latent



if __name__ == '__main__':
	pass


###
