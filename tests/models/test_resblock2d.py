import pytest

torch = pytest.importorskip("torch")

from ava.models.blocks import ResBlock2D, ResBlockUp2D


def _make_norm(num_channels):
	groups = min(4, num_channels)
	while groups > 1 and num_channels % groups != 0:
		groups -= 1
	return torch.nn.GroupNorm(groups, num_channels)


def test_resblock2d_stride1_shape_and_backward():
	block = ResBlock2D(4, 4, stride=1, norm_factory=_make_norm)
	x = torch.randn(2, 4, 33, 65, requires_grad=True)
	out = block(x)
	assert out.shape == x.shape
	out.mean().backward()
	assert any(param.grad is not None for param in block.parameters())


def test_resblock2d_stride2_downsample_shape():
	block = ResBlock2D(3, 6, stride=2, norm_factory=_make_norm)
	x = torch.randn(2, 3, 31, 63)
	out = block(x)
	expected = ((x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2)
	assert out.shape == (x.shape[0], 6, *expected)


def test_resblockup2d_upsample_shape():
	block = ResBlockUp2D(5, 7, norm_factory=_make_norm)
	x = torch.randn(2, 5, 8, 11)
	size = (17, 21)
	out = block(x, size)
	assert out.shape == (x.shape[0], 7, *size)
