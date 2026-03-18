"""
Deterministic spectrogram augmentations for fixed-window training/evaluation.
"""
from __future__ import annotations

from typing import Optional, Union

import torch

from ava.models.fixed_window_config import FixedWindowAugmentationConfig


TensorLike = torch.Tensor


def _as_batch(spec: TensorLike) -> tuple[TensorLike, bool]:
	if not torch.is_tensor(spec):
		raise TypeError("spec must be a torch.Tensor.")
	if spec.ndim == 2:
		return spec.unsqueeze(0), True
	if spec.ndim == 3:
		return spec, False
	raise ValueError("spec must have shape [H, W] or [B, H, W].")


def _get_generator(
		spec: TensorLike,
		generator: Optional[torch.Generator],
		seed: Optional[int],
) -> Optional[torch.Generator]:
	if generator is not None:
		return generator
	if seed is None:
		return None
	device = spec.device if spec.device.type in ("cpu", "cuda") else "cpu"
	rng = torch.Generator(device=device)
	rng.manual_seed(int(seed))
	return rng


def _coerce_nonnegative_int(value, name) -> int:
	try:
		value = int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must be a non-negative integer.") from exc
	if value < 0:
		raise ValueError(f"{name} must be a non-negative integer.")
	return value


def _coerce_nonnegative_float(value, name) -> float:
	try:
		value = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must be a non-negative number.") from exc
	if not torch.isfinite(torch.tensor(value)) or value < 0:
		raise ValueError(f"{name} must be a non-negative number.")
	return value


def _coerce_scale_range(value, name) -> tuple[float, float]:
	if isinstance(value, (int, float)):
		low = high = value
	elif isinstance(value, (list, tuple)) and len(value) == 2:
		low, high = value
	else:
		raise ValueError(f"{name} must be a length-2 tuple.")
	try:
		low = float(low)
		high = float(high)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{name} must contain numeric values.") from exc
	if low > high:
		raise ValueError(f"{name} must satisfy low <= high.")
	if not torch.isfinite(torch.tensor(low)) or not torch.isfinite(torch.tensor(high)):
		raise ValueError(f"{name} must contain finite values.")
	return (low, high)


def _sample_uniform(
		shape: tuple[int, ...],
		low: float,
		high: float,
		spec: TensorLike,
		generator: Optional[torch.Generator],
) -> TensorLike:
	if low == high:
		return torch.full(shape, low, device=spec.device, dtype=spec.dtype)
	values = torch.empty(shape, device=spec.device, dtype=spec.dtype)
	return values.uniform_(low, high, generator=generator)


def amplitude_jitter(
		spec: TensorLike,
		scale_range: tuple[float, float],
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	spec_b, single = _as_batch(spec)
	low, high = _coerce_scale_range(scale_range, "amplitude_scale")
	if low == 1.0 and high == 1.0:
		return spec
	scales = _sample_uniform(
		(spec_b.shape[0], 1, 1),
		low,
		high,
		spec_b,
		generator,
	)
	result = spec_b * scales
	if single:
		return result[0]
	return result


def additive_noise(
		spec: TensorLike,
		noise_std: Optional[Union[float, tuple[float, float]]] = None,
		snr_db: Optional[Union[float, tuple[float, float]]] = None,
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	if noise_std is None and snr_db is None:
		return spec
	if noise_std is not None and snr_db is not None:
		raise ValueError("Specify either noise_std or snr_db, not both.")
	spec_b, single = _as_batch(spec)
	if snr_db is not None:
		snr_low, snr_high = _coerce_scale_range(snr_db, "snr_db")
		snr_vals = _sample_uniform(
			(spec_b.shape[0], 1, 1),
			snr_low,
			snr_high,
			spec_b,
			generator,
		)
		rms = torch.sqrt(torch.mean(spec_b ** 2, dim=(1, 2), keepdim=True))
		noise_sigma = rms / torch.pow(10.0, snr_vals / 20.0)
	else:
		if isinstance(noise_std, (list, tuple)):
			low, high = _coerce_scale_range(noise_std, "noise_std")
			noise_sigma = _sample_uniform(
				(spec_b.shape[0], 1, 1),
				low,
				high,
				spec_b,
				generator,
			)
		else:
			sigma = _coerce_nonnegative_float(noise_std, "noise_std")
			noise_sigma = torch.full(
				(spec_b.shape[0], 1, 1),
				sigma,
				device=spec_b.device,
				dtype=spec_b.dtype,
			)
	noise = torch.randn(
		spec_b.shape,
		device=spec_b.device,
		dtype=spec_b.dtype,
		generator=generator,
	) * noise_sigma
	result = spec_b + noise
	if single:
		return result[0]
	return result


def _shift_along_time(
		spec_b: TensorLike,
		shifts: TensorLike,
) -> TensorLike:
	result = torch.zeros_like(spec_b)
	for idx, shift in enumerate(shifts.tolist()):
		if shift == 0:
			result[idx] = spec_b[idx]
		elif shift > 0:
			result[idx, :, shift:] = spec_b[idx, :, :-shift]
		else:
			k = -shift
			result[idx, :, :-k] = spec_b[idx, :, k:]
	return result


def _shift_along_freq(
		spec_b: TensorLike,
		shifts: TensorLike,
) -> TensorLike:
	result = torch.zeros_like(spec_b)
	for idx, shift in enumerate(shifts.tolist()):
		if shift == 0:
			result[idx] = spec_b[idx]
		elif shift > 0:
			result[idx, shift:, :] = spec_b[idx, :-shift, :]
		else:
			k = -shift
			result[idx, :-k, :] = spec_b[idx, k:, :]
	return result


def time_shift(
		spec: TensorLike,
		max_bins: int,
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	spec_b, single = _as_batch(spec)
	max_bins = _coerce_nonnegative_int(max_bins, "time_shift_max_bins")
	if max_bins == 0 or spec_b.shape[2] <= 1:
		return spec
	max_bins = min(max_bins, spec_b.shape[2] - 1)
	shift_vals = torch.randint(
		0,
		2 * max_bins + 1,
		(spec_b.shape[0],),
		device=spec_b.device,
		generator=generator,
	) - max_bins
	result = _shift_along_time(spec_b, shift_vals)
	if single:
		return result[0]
	return result


def freq_shift(
		spec: TensorLike,
		max_bins: int,
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	spec_b, single = _as_batch(spec)
	max_bins = _coerce_nonnegative_int(max_bins, "freq_shift_max_bins")
	if max_bins == 0 or spec_b.shape[1] <= 1:
		return spec
	max_bins = min(max_bins, spec_b.shape[1] - 1)
	shift_vals = torch.randint(
		0,
		2 * max_bins + 1,
		(spec_b.shape[0],),
		device=spec_b.device,
		generator=generator,
	) - max_bins
	result = _shift_along_freq(spec_b, shift_vals)
	if single:
		return result[0]
	return result


def _apply_masks(
		spec_b: TensorLike,
		mask_axis: int,
		max_bins: int,
		count: int,
		generator: Optional[torch.Generator],
) -> TensorLike:
	max_bins = _coerce_nonnegative_int(max_bins, "mask_max_bins")
	count = _coerce_nonnegative_int(count, "mask_count")
	if max_bins == 0 or count == 0:
		return spec_b
	limit = spec_b.shape[mask_axis]
	if limit == 0:
		return spec_b
	for idx in range(spec_b.shape[0]):
		for _ in range(count):
			width = torch.randint(
				0,
				min(max_bins, limit) + 1,
				(1,),
				device=spec_b.device,
				generator=generator,
			).item()
			if width == 0:
				continue
			if width >= limit:
				start = 0
				stop = limit
			else:
				start = torch.randint(
					0,
					limit - width + 1,
					(1,),
					device=spec_b.device,
					generator=generator,
				).item()
				stop = start + width
			if mask_axis == 2:
				spec_b[idx, :, start:stop] = 0.0
			else:
				spec_b[idx, start:stop, :] = 0.0
	return spec_b


def time_mask(
		spec: TensorLike,
		max_bins: int,
		count: int,
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	spec_b, single = _as_batch(spec)
	result = spec_b.clone()
	result = _apply_masks(result, 2, max_bins, count, generator)
	if single:
		return result[0]
	return result


def freq_mask(
		spec: TensorLike,
		max_bins: int,
		count: int,
		generator: Optional[torch.Generator] = None,
) -> TensorLike:
	spec_b, single = _as_batch(spec)
	result = spec_b.clone()
	result = _apply_masks(result, 1, max_bins, count, generator)
	if single:
		return result[0]
	return result


def _parse_config(
		config: Optional[Union[FixedWindowAugmentationConfig, dict, bool]],
) -> FixedWindowAugmentationConfig:
	if config is None:
		return FixedWindowAugmentationConfig()
	if isinstance(config, FixedWindowAugmentationConfig):
		return config
	if isinstance(config, dict):
		return FixedWindowAugmentationConfig.from_dict(config)
	if isinstance(config, bool):
		return FixedWindowAugmentationConfig(enabled=config)
	raise TypeError(
		"augmentations must be a dict, bool, or FixedWindowAugmentationConfig."
	)


def apply_augmentations(
		spec: TensorLike,
		augmentations: Optional[Union[FixedWindowAugmentationConfig, dict, bool]] = None,
		generator: Optional[torch.Generator] = None,
		seed: Optional[int] = None,
) -> TensorLike:
	config = _parse_config(augmentations)
	if not config.enabled:
		return spec
	rng = _get_generator(spec, generator, seed if seed is not None else config.seed)
	result = amplitude_jitter(spec, config.amplitude_scale, generator=rng)
	if config.noise_std is not None and config.noise_std != 0:
		result = additive_noise(result, noise_std=config.noise_std, generator=rng)
	if config.time_shift_max_bins:
		result = time_shift(result, config.time_shift_max_bins, generator=rng)
	if config.freq_shift_max_bins:
		result = freq_shift(result, config.freq_shift_max_bins, generator=rng)
	if config.time_mask_count and config.time_mask_max_bins:
		result = time_mask(
			result,
			config.time_mask_max_bins,
			config.time_mask_count,
			generator=rng,
		)
	if config.freq_mask_count and config.freq_mask_max_bins:
		result = freq_mask(
			result,
			config.freq_mask_max_bins,
			config.freq_mask_count,
			generator=rng,
		)
	return result


class SpectrogramAugmenter:
	"""
	Callable augmentation pipeline with deterministic seeding.
	"""

	def __init__(
			self,
			augmentations: Optional[Union[FixedWindowAugmentationConfig, dict, bool]] = None,
			seed: Optional[int] = None,
	) -> None:
		self.config = _parse_config(augmentations)
		self.seed = seed

	def __call__(
			self,
			spec: TensorLike,
			generator: Optional[torch.Generator] = None,
			seed: Optional[int] = None,
	) -> TensorLike:
		return apply_augmentations(
			spec,
			self.config,
			generator=generator,
			seed=self.seed if seed is None else seed,
		)
