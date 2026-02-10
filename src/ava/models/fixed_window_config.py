"""
Structured configuration for fixed-window VAE experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import yaml

from ava.models.vae import X_SHAPE
from ava.preprocessing.utils import get_spec


_SPEC_REGISTRY = {
	"get_spec": get_spec,
}


def _resolve_get_spec(name):
	if callable(name):
		return name
	if name is None:
		return get_spec
	if not isinstance(name, str):
		raise TypeError("get_spec must be a string or callable.")
	try:
		return _SPEC_REGISTRY[name]
	except KeyError as exc:
		available = ", ".join(sorted(_SPEC_REGISTRY))
		raise ValueError(
			f"Unknown get_spec '{name}'. Available: {available}."
		) from exc


def _ensure_tuple(value, default=()):
	if value is None:
		return tuple(default)
	if isinstance(value, tuple):
		return value
	if isinstance(value, list):
		return tuple(value)
	return (value,)


def _validate_keys(data, cls):
	field_names = {field.name for field in fields(cls)}
	extras = set(data) - field_names
	if extras:
		raise ValueError(
			f"Unexpected keys for {cls.__name__}: {sorted(extras)}"
		)


@dataclass
class FixedWindowPreprocessConfig:
	fs: int
	get_spec: str = "get_spec"
	num_freq_bins: int = X_SHAPE[0]
	num_time_bins: int = X_SHAPE[1]
	nperseg: int = 512
	noverlap: int = 256
	max_dur: float = 1e9
	window_length: float = 0.12
	min_freq: float = 400.0
	max_freq: float = 10e3
	spec_min_val: float = 2.0
	spec_max_val: float = 6.5
	mel: bool = True
	time_stretch: bool = False
	within_syll_normalize: bool = False
	normalization_mode: str = "none"
	normalization_method: str = "mean_std"
	normalization_num_samples: int = 128
	normalization_seed: int = 0
	file_weight_mode: str = "duration"
	file_weight_cap: Optional[float] = None
	roi_weight_mode: str = "duration"
	sampling_seed: Optional[int] = None
	log_window_indices: bool = False
	real_preprocess_params: tuple = (
		"min_freq",
		"max_freq",
		"spec_min_val",
		"spec_max_val",
	)
	int_preprocess_params: tuple = ()
	binary_preprocess_params: tuple = ("mel", "within_syll_normalize")

	def to_params(self) -> dict:
		return {
			"fs": self.fs,
			"get_spec": _resolve_get_spec(self.get_spec),
			"num_freq_bins": self.num_freq_bins,
			"num_time_bins": self.num_time_bins,
			"nperseg": self.nperseg,
			"noverlap": self.noverlap,
			"max_dur": self.max_dur,
			"window_length": self.window_length,
			"min_freq": self.min_freq,
			"max_freq": self.max_freq,
			"spec_min_val": self.spec_min_val,
			"spec_max_val": self.spec_max_val,
			"mel": self.mel,
			"time_stretch": self.time_stretch,
			"within_syll_normalize": self.within_syll_normalize,
			"normalization_mode": self.normalization_mode,
			"normalization_method": self.normalization_method,
			"normalization_num_samples": self.normalization_num_samples,
			"normalization_seed": self.normalization_seed,
			"file_weight_mode": self.file_weight_mode,
			"file_weight_cap": self.file_weight_cap,
			"roi_weight_mode": self.roi_weight_mode,
			"sampling_seed": self.sampling_seed,
			"log_window_indices": self.log_window_indices,
			"real_preprocess_params": self.real_preprocess_params,
			"int_preprocess_params": self.int_preprocess_params,
			"binary_preprocess_params": self.binary_preprocess_params,
		}

	def to_dict(self) -> dict:
		return {
			"fs": self.fs,
			"get_spec": self.get_spec,
			"num_freq_bins": self.num_freq_bins,
			"num_time_bins": self.num_time_bins,
			"nperseg": self.nperseg,
			"noverlap": self.noverlap,
			"max_dur": self.max_dur,
			"window_length": self.window_length,
			"min_freq": self.min_freq,
			"max_freq": self.max_freq,
			"spec_min_val": self.spec_min_val,
			"spec_max_val": self.spec_max_val,
			"mel": self.mel,
			"time_stretch": self.time_stretch,
			"within_syll_normalize": self.within_syll_normalize,
			"normalization_mode": self.normalization_mode,
			"normalization_method": self.normalization_method,
			"normalization_num_samples": self.normalization_num_samples,
			"normalization_seed": self.normalization_seed,
			"file_weight_mode": self.file_weight_mode,
			"file_weight_cap": self.file_weight_cap,
			"roi_weight_mode": self.roi_weight_mode,
			"sampling_seed": self.sampling_seed,
			"log_window_indices": self.log_window_indices,
			"real_preprocess_params": list(self.real_preprocess_params),
			"int_preprocess_params": list(self.int_preprocess_params),
			"binary_preprocess_params": list(self.binary_preprocess_params),
		}

	@classmethod
	def from_dict(cls, data: dict) -> "FixedWindowPreprocessConfig":
		data = dict(data or {})
		_validate_keys(data, cls)
		data["real_preprocess_params"] = _ensure_tuple(
			data.get("real_preprocess_params"),
			default=cls.real_preprocess_params,
		)
		data["int_preprocess_params"] = _ensure_tuple(
			data.get("int_preprocess_params"),
			default=cls.int_preprocess_params,
		)
		data["binary_preprocess_params"] = _ensure_tuple(
			data.get("binary_preprocess_params"),
			default=cls.binary_preprocess_params,
		)
		return cls(**data)


@dataclass
class FixedWindowDataConfig:
	batch_size: int = 64
	num_workers: Optional[int] = None
	shuffle_train: bool = True
	shuffle_test: bool = False
	min_spec_val: Optional[float] = None
	min_audio_energy: Optional[float] = None
	spec_cache_dir: Optional[str] = None
	audio_cache_size: int = 0
	pin_memory: Optional[bool] = None
	persistent_workers: Optional[bool] = None
	prefetch_factor: Optional[int] = 2

	def to_loader_kwargs(self) -> dict:
		return {
			"batch_size": self.batch_size,
			"shuffle": (self.shuffle_train, self.shuffle_test),
			"num_workers": self.num_workers,
			"min_spec_val": self.min_spec_val,
			"min_audio_energy": self.min_audio_energy,
			"spec_cache_dir": self.spec_cache_dir,
			"audio_cache_size": self.audio_cache_size,
			"pin_memory": self.pin_memory,
			"persistent_workers": self.persistent_workers,
			"prefetch_factor": self.prefetch_factor,
		}

	def to_dict(self) -> dict:
		return {
			"batch_size": self.batch_size,
			"num_workers": self.num_workers,
			"shuffle_train": self.shuffle_train,
			"shuffle_test": self.shuffle_test,
			"min_spec_val": self.min_spec_val,
			"min_audio_energy": self.min_audio_energy,
			"spec_cache_dir": self.spec_cache_dir,
			"audio_cache_size": self.audio_cache_size,
			"pin_memory": self.pin_memory,
			"persistent_workers": self.persistent_workers,
			"prefetch_factor": self.prefetch_factor,
		}

	@classmethod
	def from_dict(cls, data: dict) -> "FixedWindowDataConfig":
		data = dict(data or {})
		if "shuffle" in data and "shuffle_train" not in data:
			shuffle = data.pop("shuffle")
			if isinstance(shuffle, (list, tuple)) and len(shuffle) == 2:
				data["shuffle_train"], data["shuffle_test"] = shuffle
		_validate_keys(data, cls)
		return cls(**data)


@dataclass
class FixedWindowAugmentationConfig:
	enabled: bool = False
	seed: Optional[int] = None
	amplitude_scale: tuple = (0.9, 1.1)
	noise_std: float = 0.0
	time_shift_max_bins: int = 0
	freq_shift_max_bins: int = 0
	time_mask_max_bins: int = 0
	time_mask_count: int = 0
	freq_mask_max_bins: int = 0
	freq_mask_count: int = 0

	def to_dict(self) -> dict:
		return {
			"enabled": self.enabled,
			"seed": self.seed,
			"amplitude_scale": list(self.amplitude_scale),
			"noise_std": self.noise_std,
			"time_shift_max_bins": self.time_shift_max_bins,
			"freq_shift_max_bins": self.freq_shift_max_bins,
			"time_mask_max_bins": self.time_mask_max_bins,
			"time_mask_count": self.time_mask_count,
			"freq_mask_max_bins": self.freq_mask_max_bins,
			"freq_mask_count": self.freq_mask_count,
		}

	@classmethod
	def from_dict(cls, data: dict) -> "FixedWindowAugmentationConfig":
		data = dict(data or {})
		_validate_keys(data, cls)
		if "amplitude_scale" in data:
			data["amplitude_scale"] = _ensure_tuple(
				data.get("amplitude_scale"),
				default=cls.amplitude_scale,
			)
		return cls(**data)


@dataclass
class FixedWindowTrainConfig:
	lr: float = 1e-3
	z_dim: int = 32
	model_precision: float = 10.0
	learn_observation_scale: bool = False
	log_precision_min: Optional[float] = None
	log_precision_max: Optional[float] = None
	epochs: int = 100
	test_freq: Optional[int] = 2
	save_freq: Optional[int] = 10
	vis_freq: Optional[int] = 1
	num_specs: int = 5
	gap: tuple = (2, 6)
	vis_filename: str = "reconstruction.pdf"
	input_shape: Optional[tuple] = None
	posterior_type: str = "diag"
	conv_arch: str = "residual"
	decoder_type: str = "upsample"
	kl_beta: float = 1.0
	kl_warmup_epochs: int = 0
	invariance_weight: float = 0.0
	invariance_warmup_epochs: int = 0
	invariance_loss: str = "mse"
	invariance_stop_grad: str = "none"
	compile_model: bool = False
	compile_kwargs: Optional[dict] = None
	trainer_kwargs: dict = field(default_factory=dict)
	stopping_kwargs: Optional[dict] = None

	def to_train_kwargs(self) -> dict:
		return {
			"lr": self.lr,
			"z_dim": self.z_dim,
			"model_precision": self.model_precision,
			"learn_observation_scale": self.learn_observation_scale,
			"log_precision_min": self.log_precision_min,
			"log_precision_max": self.log_precision_max,
			"epochs": self.epochs,
			"test_freq": self.test_freq,
			"save_freq": self.save_freq,
			"vis_freq": self.vis_freq,
			"num_specs": self.num_specs,
			"gap": self.gap,
			"vis_filename": self.vis_filename,
			"trainer_kwargs": self.trainer_kwargs,
			"stopping_kwargs": self.stopping_kwargs,
			"input_shape": self.input_shape,
			"posterior_type": self.posterior_type,
			"conv_arch": self.conv_arch,
			"decoder_type": self.decoder_type,
			"kl_beta": self.kl_beta,
			"kl_warmup_epochs": self.kl_warmup_epochs,
			"invariance_weight": self.invariance_weight,
			"invariance_warmup_epochs": self.invariance_warmup_epochs,
			"invariance_loss": self.invariance_loss,
			"invariance_stop_grad": self.invariance_stop_grad,
			"compile_model": self.compile_model,
			"compile_kwargs": self.compile_kwargs,
		}

	def to_dict(self) -> dict:
		return {
			"lr": self.lr,
			"z_dim": self.z_dim,
			"model_precision": self.model_precision,
			"learn_observation_scale": self.learn_observation_scale,
			"log_precision_min": self.log_precision_min,
			"log_precision_max": self.log_precision_max,
			"epochs": self.epochs,
			"test_freq": self.test_freq,
			"save_freq": self.save_freq,
			"vis_freq": self.vis_freq,
			"num_specs": self.num_specs,
			"gap": list(self.gap),
			"vis_filename": self.vis_filename,
			"input_shape": list(self.input_shape) if self.input_shape else None,
			"posterior_type": self.posterior_type,
			"conv_arch": self.conv_arch,
			"decoder_type": self.decoder_type,
			"kl_beta": self.kl_beta,
			"kl_warmup_epochs": self.kl_warmup_epochs,
			"invariance_weight": self.invariance_weight,
			"invariance_warmup_epochs": self.invariance_warmup_epochs,
			"invariance_loss": self.invariance_loss,
			"invariance_stop_grad": self.invariance_stop_grad,
			"compile_model": self.compile_model,
			"compile_kwargs": self.compile_kwargs,
			"trainer_kwargs": self.trainer_kwargs,
			"stopping_kwargs": self.stopping_kwargs,
		}

	@classmethod
	def from_dict(cls, data: dict) -> "FixedWindowTrainConfig":
		data = dict(data or {})
		_validate_keys(data, cls)
		if "gap" in data:
			data["gap"] = _ensure_tuple(data.get("gap"), default=cls.gap)
		if "input_shape" in data and data["input_shape"] is not None:
			data["input_shape"] = _ensure_tuple(data.get("input_shape"))
		return cls(**data)


@dataclass
class FixedWindowExperimentConfig:
	preprocess: FixedWindowPreprocessConfig
	augmentations: FixedWindowAugmentationConfig = field(
		default_factory=FixedWindowAugmentationConfig
	)
	data: FixedWindowDataConfig = field(default_factory=FixedWindowDataConfig)
	training: FixedWindowTrainConfig = field(default_factory=FixedWindowTrainConfig)

	def to_dict(self) -> dict:
		return {
			"preprocess": self.preprocess.to_dict(),
			"augmentations": self.augmentations.to_dict(),
			"data": self.data.to_dict(),
			"training": self.training.to_dict(),
		}

	def to_yaml(self, path: str) -> None:
		with open(path, "w", encoding="utf-8") as handle:
			yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

	@classmethod
	def from_dict(cls, data: dict) -> "FixedWindowExperimentConfig":
		data = dict(data or {})
		if "preprocess" not in data:
			raise ValueError("Config must include a 'preprocess' section.")
		_validate_keys(data, cls)
		return cls(
			preprocess=FixedWindowPreprocessConfig.from_dict(data["preprocess"]),
			augmentations=FixedWindowAugmentationConfig.from_dict(
				data.get("augmentations")
			),
			data=FixedWindowDataConfig.from_dict(data.get("data")),
			training=FixedWindowTrainConfig.from_dict(data.get("training")),
		)

	@classmethod
	def from_yaml(cls, path: str) -> "FixedWindowExperimentConfig":
		with open(path, "r", encoding="utf-8") as handle:
			payload = yaml.safe_load(handle) or {}
		return cls.from_dict(payload)
