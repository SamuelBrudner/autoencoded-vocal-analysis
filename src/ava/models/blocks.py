"""
Residual block primitives for 2D spectrogram models.
"""
from types import SimpleNamespace

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional in some envs
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None
	_TORCH_IMPORT_ERROR = exc
else:
	_TORCH_IMPORT_ERROR = None


__all__ = ["ResBlock2D", "ResBlockUp2D"]


def _make_norm(num_channels, max_groups=8):
	groups = min(max_groups, num_channels)
	while groups > 1 and num_channels % groups != 0:
		groups -= 1
	return nn.GroupNorm(groups, num_channels)


class ResBlock2D(nn.Module):
	"""Residual block with Conv-Norm-Act + skip."""

	def __init__(self, in_channels, out_channels, stride=1,
		norm_factory=None):
		super().__init__()
		if _TORCH_IMPORT_ERROR is not None:
			raise ImportError(
				"PyTorch is required for ava.models.blocks. "
				"Install with `pip install torch`."
			) from _TORCH_IMPORT_ERROR
		if norm_factory is None:
			norm_factory = _make_norm
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
		if _TORCH_IMPORT_ERROR is not None:
			raise ImportError(
				"PyTorch is required for ava.models.blocks. "
				"Install with `pip install torch`."
			) from _TORCH_IMPORT_ERROR
		if norm_factory is None:
			norm_factory = _make_norm
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
