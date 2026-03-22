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
from ava.models.blocks import ResBlock2D, ResBlockUp2D


DEFAULT_INPUT_SHAPE = (128, 128)
"""Default processed spectrogram shape: ``[freq_bins, time_bins]``."""
X_SHAPE = DEFAULT_INPUT_SHAPE
"""Legacy alias for the default processed spectrogram shape."""
X_DIM = int(np.prod(X_SHAPE))
"""Legacy default spectrogram dimension: ``freq_bins * time_bins``."""


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
		Decoder convolutional architecture. The decoder now uses residual
		blocks; ``"plain"`` is retained as a legacy alias for backward
		compatibility.
	decoder_type : {"upsample", "convtranspose"}, optional
		Decoder upsampling implementation. ``"upsample"`` uses nearest-neighbor
		upsampling plus convolutions, while ``"convtranspose"`` uses
		ConvTranspose2d layers for legacy compatibility.

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
		decoder_type="upsample",
		build_optimizer=True,
		log_precision_min=None,
		log_precision_max=None,
		posterior_logvar_min=None,
		posterior_logvar_max=None):
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
			Decoder convolutional architecture. Residual blocks are used in the
			decoder; ``"plain"`` is accepted as a legacy alias. Defaults to
			``"plain"``.
		decoder_type : {"upsample", "convtranspose"}, optional
			Decoder upsampling implementation. ``"upsample"`` uses
			nearest-neighbor upsampling plus convolutions, while
			``"convtranspose"`` uses ConvTranspose2d layers for legacy
			compatibility. Defaults to ``"upsample"``.
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
		self.set_log_precision_bounds(
			log_precision_min=log_precision_min,
			log_precision_max=log_precision_max,
		)
		self.set_logvar_bounds(
			posterior_logvar_min=posterior_logvar_min,
			posterior_logvar_max=posterior_logvar_max,
		)
		self.input_shape = self._normalize_input_shape(input_shape)
		self.input_dim = int(np.prod(self.input_shape))
		self.posterior_type = self._normalize_posterior_type(posterior_type)
		self.conv_arch = self._normalize_conv_arch(conv_arch)
		self.decoder_type = self._normalize_decoder_type(decoder_type)
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
		return float(torch.exp(self._log_precision_tensor()).detach().cpu())


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

	def set_log_precision_bounds(
			self,
			log_precision_min=None,
			log_precision_max=None,
	) -> None:
		"""Optionally clamp log-precision for numerical stability."""
		log_precision_min = self._coerce_optional_float(
			log_precision_min, "log_precision_min"
		)
		log_precision_max = self._coerce_optional_float(
			log_precision_max, "log_precision_max"
		)
		if (log_precision_min is not None and log_precision_max is not None
				and log_precision_min > log_precision_max):
			raise ValueError("log_precision_min must be <= log_precision_max.")
		self.log_precision_min = log_precision_min
		self.log_precision_max = log_precision_max

	def _log_precision_tensor(self):
		log_precision = self.log_precision
		if self.log_precision_min is None and self.log_precision_max is None:
			return log_precision
		if self.log_precision_min is not None and self.log_precision_max is not None:
			return torch.clamp(
				log_precision,
				min=float(self.log_precision_min),
				max=float(self.log_precision_max),
			)
		if self.log_precision_min is not None:
			return torch.clamp(
				log_precision,
				min=float(self.log_precision_min),
			)
		return torch.clamp(
			log_precision,
			max=float(self.log_precision_max),
		)


	def _precision_tensor(self):
		return torch.exp(self._log_precision_tensor())


	def set_logvar_bounds(
			self,
			posterior_logvar_min=None,
			posterior_logvar_max=None,
	) -> None:
		"""Optionally clamp posterior log-variance for numerical stability."""
		posterior_logvar_min = self._coerce_optional_float(
			posterior_logvar_min, "posterior_logvar_min"
		)
		posterior_logvar_max = self._coerce_optional_float(
			posterior_logvar_max, "posterior_logvar_max"
		)
		if (posterior_logvar_min is not None and posterior_logvar_max is not None
				and posterior_logvar_min > posterior_logvar_max):
			raise ValueError(
				"posterior_logvar_min must be <= posterior_logvar_max."
			)
		self.posterior_logvar_min = posterior_logvar_min
		self.posterior_logvar_max = posterior_logvar_max


	def _clamp_logvar_tensor(self, logvar):
		if (self.posterior_logvar_min is None
				and self.posterior_logvar_max is None):
			return logvar
		if (self.posterior_logvar_min is not None
				and self.posterior_logvar_max is not None):
			return torch.clamp(
				logvar,
				min=float(self.posterior_logvar_min),
				max=float(self.posterior_logvar_max),
			)
		if self.posterior_logvar_min is not None:
			return torch.clamp(
				logvar,
				min=float(self.posterior_logvar_min),
			)
		return torch.clamp(
			logvar,
			max=float(self.posterior_logvar_max),
		)

	def _logvar_clamp_hit_fractions(self, raw_logvar):
		"""Return fractions of posterior logvar entries clipped by each bound."""
		zero = raw_logvar.new_zeros(())
		if raw_logvar.numel() == 0:
			return zero, zero, zero
		lower_hits = torch.zeros_like(raw_logvar, dtype=torch.bool)
		upper_hits = torch.zeros_like(raw_logvar, dtype=torch.bool)
		if self.posterior_logvar_min is not None:
			lower_hits = raw_logvar < float(self.posterior_logvar_min)
		if self.posterior_logvar_max is not None:
			upper_hits = raw_logvar > float(self.posterior_logvar_max)
		dtype = raw_logvar.dtype
		lower_fraction = lower_hits.to(dtype=dtype).mean()
		upper_fraction = upper_hits.to(dtype=dtype).mean()
		any_fraction = (lower_hits | upper_hits).to(dtype=dtype).mean()
		return lower_fraction, upper_fraction, any_fraction


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
			return "residual"
		value = str(conv_arch).strip().lower()
		if value in {"plain", "legacy", "conv"}:
			return "residual"
		if value in {"residual", "resnet", "res"}:
			return "residual"
		raise ValueError("conv_arch must be 'residual' (legacy 'plain' accepted).")

	@staticmethod
	def _normalize_decoder_type(decoder_type):
		if decoder_type is None:
			return "upsample"
		value = str(decoder_type).strip().lower()
		if value in {"upsample", "upsampling", "resize", "nearest"}:
			return "upsample"
		if value in {"convtranspose", "conv_transpose", "deconv", "transpose",
				"legacy"}:
			return "convtranspose"
		raise ValueError(
			"decoder_type must be 'upsample' or 'convtranspose'."
		)


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
	def _coerce_optional_float(value, name):
		if value is None:
			return None
		if isinstance(value, str) and not value.strip():
			return None
		try:
			value = float(value)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"{name} must be a float or null.") from exc
		if not math.isfinite(value):
			raise ValueError(f"{name} must be a finite float or null.")
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

	@staticmethod
	def _output_padding_dim(input_size, target_size, kernel_size=3,
			stride=2, padding=1):
		base = (input_size - 1) * stride - 2 * padding + kernel_size
		output_padding = target_size - base
		if output_padding < 0 or output_padding >= stride:
			raise ValueError(
				"Cannot match target size "
				f"{target_size} from input size {input_size} "
				"with ConvTranspose2d configuration."
			)
		return int(output_padding)


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
		logvar = self._clamp_logvar_tensor(logvar)
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
		logvar = self._clamp_logvar_tensor(logvar)
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

		# Decoder (upsample or convtranspose blocks mirror encoder)
		self.fc5 = nn.Linear(self.z_dim, 64)
		self.fc6 = nn.Linear(64, 256)
		self.fc7 = nn.Linear(256, 1024)
		self.fc8 = nn.Linear(1024, self._conv_feature_dim)
		if self.decoder_type == "upsample":
			self.dec_blocks = nn.ModuleList([
				ResBlockUp2D(32, 24, norm_factory=self._make_norm),
				ResBlockUp2D(24, 16, norm_factory=self._make_norm),
				ResBlockUp2D(16, 8, norm_factory=self._make_norm),
			])
		else:
			output_paddings = []
			current_shape = self._conv_feature_shape[1:]
			for target_shape in self._decoder_upsample_shapes:
				pad_h = self._output_padding_dim(
					current_shape[0], target_shape[0]
				)
				pad_w = self._output_padding_dim(
					current_shape[1], target_shape[1]
				)
				output_paddings.append((pad_h, pad_w))
				current_shape = target_shape
			self.dec_blocks = nn.ModuleList([
				nn.Sequential(
					nn.ConvTranspose2d(
						32, 24, 3, stride=2, padding=1,
						output_padding=output_paddings[0],
					),
					self._make_norm(24),
					nn.SiLU(),
				),
				nn.Sequential(
					nn.ConvTranspose2d(
						24, 16, 3, stride=2, padding=1,
						output_padding=output_paddings[1],
					),
					self._make_norm(16),
					nn.SiLU(),
				),
				nn.Sequential(
					nn.ConvTranspose2d(
						16, 8, 3, stride=2, padding=1,
						output_padding=output_paddings[2],
					),
					self._make_norm(8),
					nn.SiLU(),
				),
			])
		self.dec_out = nn.Conv2d(8, 1, 3, 1, padding=1)


	def _get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		layers = {
			'fc1': self.fc1, 'fc2': self.fc2, 'fc31': self.fc31,
			'fc32': self.fc32, 'fc33': self.fc33, 'fc41': self.fc41,
			'fc42': self.fc42, 'fc43': self.fc43, 'fc5': self.fc5,
			'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
			'enc_blocks': self.enc_blocks, 'enc_out': self.enc_out,
		}
		layers.update({
			'dec_blocks': self.dec_blocks,
			'dec_out': self.dec_out,
		})
		return layers


	def encode_with_raw_logvar(self, x):
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
		raw_logvar : torch.Tensor
			Unclamped posterior log-variance prior to optional bounds.
		"""
		self._check_input(x)
		x = x.unsqueeze(1)
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
		raw_logvar = self._act(self.fc33(x))
		raw_logvar = self.fc43(raw_logvar)
		logvar = self._clamp_logvar_tensor(raw_logvar)
		return mu, logvar, u, raw_logvar

	def encode(self, x):
		mu, logvar, u, _ = self.encode_with_raw_logvar(x)
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
		if self.decoder_type == "upsample":
			for block, size in zip(self.dec_blocks,
					self._decoder_upsample_shapes):
				z = block(z, size)
		else:
			for block in self.dec_blocks:
				z = block(z)
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
				self._to_numpy(z),
				self._to_numpy(x_rec),
			)
		return x_rec, mu, logvar, z, u

	def forward_with_raw_logvar(self, x, return_latent_rec=False):
		"""Forward pass that also returns unclamped posterior log-variance."""
		mu, logvar, u, raw_logvar = self.encode_with_raw_logvar(x)
		latent_dist = self._posterior_distribution(mu, logvar, u)
		z = latent_dist.rsample()
		x_rec = self.decode(z).view(-1, *self.input_shape)
		if return_latent_rec:
			return (
				self._to_numpy(z),
				self._to_numpy(x_rec),
				self._to_numpy(raw_logvar),
			)
		return x_rec, mu, logvar, z, u, raw_logvar

	def _to_numpy(self, tensor: "torch.Tensor") -> np.ndarray:
		tensor = tensor.detach().cpu()
		try:
			return tensor.numpy()
		except RuntimeError as exc:
			if "Numpy is not available" not in str(exc):
				raise
			return np.array(tensor.tolist())


	def _compute_loss(self, x, beta=1.0):
		"""Compute the negative ELBO for legacy training loops."""
		recon, mu, logvar, z, u = self.forward(x)
		latent_dist = self._posterior_distribution(mu, logvar, u)
		x_flat = x.view(x.shape[0], -1)
		recon_flat = recon.view(x.shape[0], -1)
		log_precision = self._log_precision_tensor()
		precision = self._precision_tensor()
		log_two_pi = math.log(2 * math.pi)
		pxz_term = -0.5 * x.shape[0] * x_flat.shape[1] * (
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
			completed_epoch = epoch + 1
			if (save_freq is not None) and (completed_epoch % save_freq == 0):
				filename = "checkpoint_"+str(completed_epoch).zfill(3)+'.tar'
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
		state['log_precision_min'] = self.log_precision_min
		state['log_precision_max'] = self.log_precision_max
		state['posterior_logvar_min'] = self.posterior_logvar_min
		state['posterior_logvar_max'] = self.posterior_logvar_max
		state['learn_observation_scale'] = self.learn_observation_scale
		state['decoder_type'] = self.decoder_type
		filename = os.path.join(self.save_dir, filename)
		torch.save(state, filename)


	def load_state(self, filename, load_optimizer=True):
		"""
		Load all the model parameters from the given ``.tar`` file.

		The ``.tar`` file should be written by `self.save_state`.

		Parameters
		----------
		filename : str
			File containing a model state.
		load_optimizer : bool, optional
			Whether to restore optimizer state from the checkpoint when present.
			Defaults to ``True``.

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
		if checkpoint_arch in {"plain", "legacy", "conv"}:
			raise ValueError(
				"Checkpoint decoder uses the legacy conv stack which is "
				"incompatible with the residual-block decoder. Retrain or "
				"export a compatible checkpoint."
			)
		if checkpoint_arch != self.conv_arch:
			raise ValueError(
				"Checkpoint conv_arch "
				f"{checkpoint_arch!r} does not match model conv_arch "
				f"{self.conv_arch!r}."
			)
		if "enc_blocks" not in checkpoint:
			raise ValueError(
				"Checkpoint encoder uses the legacy conv stack which is "
				"incompatible with the residual-block encoder. Retrain or "
				"export a compatible checkpoint."
			)
		if "dec_blocks" not in checkpoint:
			raise ValueError(
				"Checkpoint decoder uses the legacy conv stack which is "
				"incompatible with the residual-block decoder. Retrain or "
				"export a compatible checkpoint."
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
		checkpoint_decoder = self._normalize_decoder_type(checkpoint_decoder)
		if checkpoint_decoder != self.decoder_type:
			raise ValueError(
				"Checkpoint decoder_type "
				f"{checkpoint_decoder!r} does not match model decoder_type "
				f"{self.decoder_type!r}. Initialize the VAE with "
				"decoder_type set to the checkpoint value or retrain."
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
		if "log_precision_min" in checkpoint or "log_precision_max" in checkpoint:
			self.set_log_precision_bounds(
				log_precision_min=checkpoint.get("log_precision_min"),
				log_precision_max=checkpoint.get("log_precision_max"),
			)
		if ("posterior_logvar_min" in checkpoint
				or "posterior_logvar_max" in checkpoint):
			self.set_logvar_bounds(
				posterior_logvar_min=checkpoint.get("posterior_logvar_min"),
				posterior_logvar_max=checkpoint.get("posterior_logvar_max"),
			)
		layers = self._get_layers()
		for layer_name in layers:
			layer = layers[layer_name]
			layer.load_state_dict(checkpoint[layer_name])
		optimizer_state = checkpoint.get("optimizer_state")
		if load_optimizer and self.optimizer is None and optimizer_state is not None:
			self._ensure_optimizer()
		if load_optimizer and self.optimizer is not None and optimizer_state is not None:
			self.optimizer.load_state_dict(optimizer_state)
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
