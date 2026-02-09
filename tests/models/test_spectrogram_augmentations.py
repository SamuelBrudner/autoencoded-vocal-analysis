import pytest

torch = pytest.importorskip("torch")

from ava.models.augmentations import apply_augmentations
from ava.models.fixed_window_config import FixedWindowAugmentationConfig


def _make_config():
    return FixedWindowAugmentationConfig(
        enabled=True,
        seed=None,
        amplitude_scale=(0.8, 1.2),
        noise_std=0.05,
        time_shift_max_bins=1,
        freq_shift_max_bins=1,
        time_mask_max_bins=1,
        time_mask_count=1,
        freq_mask_max_bins=1,
        freq_mask_count=1,
    )


def test_apply_augmentations_deterministic_seed():
    spec = torch.linspace(0.0, 1.0, steps=12, dtype=torch.float32).reshape(3, 4)
    config = _make_config()
    out1 = apply_augmentations(spec, config, seed=123)
    out2 = apply_augmentations(spec, config, seed=123)
    assert torch.allclose(out1, out2)


def test_apply_augmentations_batch_shape():
    spec = torch.ones((2, 3, 4), dtype=torch.float32)
    config = _make_config()
    out = apply_augmentations(spec, config, seed=7)
    assert out.shape == spec.shape
