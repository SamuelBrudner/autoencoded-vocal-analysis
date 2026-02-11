"""
Latent-sequence export utilities.

This module provides a Python API to encode a single audio clip into a
time-indexed sequence of posterior statistics (mu/logvar) using a trained AVA
VAE checkpoint and fixed-window preprocessing configuration.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
	import torch
except ImportError as exc:  # pragma: no cover - optional in some envs
	torch = None
	_TORCH_IMPORT_ERROR = exc
else:
	_TORCH_IMPORT_ERROR = None

from scipy.io import wavfile
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import resample_poly, stft

from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.latent_metrics import load_vae_from_checkpoint
from ava.preprocessing.utils import EPSILON, _inv_mel, _mel


SCHEMA_VERSION = "ava_latent_sequence_v1"


@dataclass
class LatentSequence:
	"""Container for a single clip's latent sequence export."""

	start_times_sec: np.ndarray
	window_length_sec: float
	hop_length_sec: float
	mu: np.ndarray
	logvar: np.ndarray
	energy: Optional[np.ndarray] = None
	gating_weight: Optional[np.ndarray] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_npz_arrays(self) -> Dict[str, np.ndarray]:
		arrays = {
			"start_times_sec": np.asarray(self.start_times_sec, dtype=np.float64),
			"window_length_sec": np.asarray(self.window_length_sec, dtype=np.float64),
			"hop_length_sec": np.asarray(self.hop_length_sec, dtype=np.float64),
			"mu": np.asarray(self.mu, dtype=np.float32),
			"logvar": np.asarray(self.logvar, dtype=np.float32),
		}
		if self.energy is not None:
			arrays["energy"] = np.asarray(self.energy, dtype=np.float32)
		if self.gating_weight is not None:
			arrays["gating_weight"] = np.asarray(self.gating_weight, dtype=np.float32)
		return arrays


def _sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as handle:
		for chunk in iter(lambda: handle.read(chunk_size), b""):
			h.update(chunk)
	return h.hexdigest()


def _as_mono(audio: np.ndarray) -> np.ndarray:
	audio = np.asarray(audio)
	if audio.ndim == 1:
		return audio
	if audio.ndim == 2:
		return np.mean(audio, axis=1)
	raise ValueError("Audio array must be 1D (mono) or 2D (samples, channels).")


def _to_float_audio(audio: np.ndarray) -> np.ndarray:
	audio = np.asarray(audio)
	if np.issubdtype(audio.dtype, np.floating):
		return audio.astype(np.float64, copy=False)
	if np.issubdtype(audio.dtype, np.integer):
		info = np.iinfo(audio.dtype)
		denom = float(max(abs(int(info.min)), int(info.max)))
		return audio.astype(np.float64) / denom
	raise TypeError(f"Unsupported audio dtype {audio.dtype!r}.")


def _resample_audio(audio: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
	if fs_in == fs_out:
		return audio
	if fs_in <= 0 or fs_out <= 0:
		raise ValueError("Sample rates must be positive.")
	g = math.gcd(int(fs_in), int(fs_out))
	up = int(fs_out // g)
	down = int(fs_in // g)
	return resample_poly(audio, up=up, down=down)


def _load_rois(path: Union[str, Path]) -> np.ndarray:
	"""Load ROI onset/offset pairs from a text file."""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(path.as_posix())
	rois = []
	with open(path, "r", encoding="utf-8") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			parts = stripped.split()
			if len(parts) < 2:
				continue
			try:
				t1 = float(parts[0])
				t2 = float(parts[1])
			except ValueError:
				continue
			if not np.isfinite(t1) or not np.isfinite(t2):
				continue
			if t2 <= t1:
				continue
			rois.append((t1, t2))
	if not rois:
		return np.zeros((0, 2), dtype=np.float64)
	return np.asarray(rois, dtype=np.float64)


def _window_starts_from_rois(
		rois: np.ndarray,
		window_length_sec: float,
		hop_length_sec: float,
		start_time_sec: float,
		end_time_sec: float,
) -> np.ndarray:
	starts: list[np.ndarray] = []
	for onset, offset in np.asarray(rois, dtype=np.float64):
		onset = max(float(onset), float(start_time_sec))
		offset = min(float(offset), float(end_time_sec))
		latest = offset - float(window_length_sec)
		if latest < onset:
			continue
		grid = np.arange(onset, latest + 1e-12, float(hop_length_sec), dtype=np.float64)
		if grid.size:
			starts.append(grid)
	if not starts:
		return np.zeros((0,), dtype=np.float64)
	return np.unique(np.concatenate(starts, axis=0))


def _default_target_freqs(params: dict) -> np.ndarray:
	if params.get("mel", True):
		freqs = np.linspace(
			_mel(params["min_freq"]),
			_mel(params["max_freq"]),
			int(params["num_freq_bins"]),
		)
		return _inv_mel(freqs)
	return np.linspace(
		float(params["min_freq"]),
		float(params["max_freq"]),
		int(params["num_freq_bins"]),
	)


def _compute_normalization_stats(
		interp: RegularGridInterpolator,
		target_freqs: np.ndarray,
		start_times: np.ndarray,
		window_length_sec: float,
		num_time_bins: int,
		params: dict,
) -> Tuple[Optional[float], Optional[float]]:
	mode = str(params.get("normalization_mode", "none")).strip().lower()
	if mode in ("none", "off", "disable"):
		return None, None

	method = str(params.get("normalization_method", "mean_std")).strip().lower()
	num_samples = int(params.get("normalization_num_samples", 128))
	seed = params.get("normalization_seed", 0)
	try:
		seed = int(seed)
	except (TypeError, ValueError) as exc:
		raise ValueError("normalization_seed must be an integer.") from exc

	if start_times.size == 0:
		return None, None
	if start_times.size <= num_samples:
		sample_times = start_times
	else:
		rng = np.random.default_rng(seed)
		indices = rng.choice(start_times.size, size=num_samples, replace=False)
		sample_times = start_times[np.sort(indices)]

	time_offsets = np.linspace(
		0.0, float(window_length_sec), int(num_time_bins), dtype=np.float64
	)
	target_times = sample_times[:, None] + time_offsets[None, :]
	freq_grid = np.broadcast_to(
		target_freqs[:, None, None],
		(target_freqs.size, sample_times.size, num_time_bins),
	)
	time_grid = np.broadcast_to(
		target_times[None, :, :],
		(target_freqs.size, sample_times.size, num_time_bins),
	)
	points = np.stack([freq_grid, time_grid], axis=-1)
	spec = interp(points).transpose(1, 0, 2)
	spec = (spec - float(params["spec_min_val"])) / (
		float(params["spec_max_val"]) - float(params["spec_min_val"])
	)
	spec = np.clip(spec, 0.0, 1.0)
	values = np.asarray(spec, dtype=np.float64).ravel()
	if values.size == 0:
		return None, None

	if method in ("robust", "median_iqr", "median-iqr"):
		center = float(np.median(values))
		q75, q25 = np.percentile(values, [75, 25])
		scale = float(q75 - q25)
		if not np.isfinite(scale) or scale <= 0:
			scale = 1.0
		return center, scale

	mean = float(np.mean(values))
	std = float(np.std(values))
	if not np.isfinite(std) or std <= 0:
		std = 1.0
	return mean, std


class LatentSequenceEncoder:
	"""
	Reusable encoder that amortizes config + checkpoint loading across many clips.

	This is useful for exporting large manifests where reloading the model for
	every clip would be prohibitively slow.
	"""

	def __init__(
			self,
			checkpoint_path: Union[str, Path],
			config: Union[str, Path, FixedWindowExperimentConfig],
			device: str = "auto",
	) -> None:
		if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover
			raise ImportError(
				"PyTorch is required for ava.models.latent_sequence. "
				"Install with `pip install torch`."
			) from _TORCH_IMPORT_ERROR
		self.checkpoint_path = Path(checkpoint_path)
		self.device = str(device)
		if isinstance(config, (str, Path)):
			self.config_path = Path(config)
			self.cfg = FixedWindowExperimentConfig.from_yaml(
				self.config_path.as_posix()
			)
		else:
			self.cfg = config
			self.config_path = None

		self.params = self.cfg.preprocess.to_params()
		self.fs_target = int(self.cfg.preprocess.fs)
		self.window_length_sec = float(self.cfg.preprocess.window_length)
		if self.window_length_sec <= 0:
			raise ValueError("window_length_sec must be positive.")

		self.num_freq_bins = int(self.params["num_freq_bins"])
		self.num_time_bins = int(self.params["num_time_bins"])
		self.target_freqs = _default_target_freqs(self.params)
		self.time_offsets = np.linspace(
			0.0,
			self.window_length_sec,
			self.num_time_bins,
			dtype=np.float64,
		)

		model = load_vae_from_checkpoint(self.checkpoint_path.as_posix(), device=device)
		if tuple(model.input_shape) != (self.num_freq_bins, self.num_time_bins):
			raise ValueError(
				"Config spectrogram shape does not match checkpoint input_shape: "
				f"config={(self.num_freq_bins, self.num_time_bins)} "
				f"checkpoint={model.input_shape}"
			)
		model.eval()
		self.model = model

	def encode(
			self,
			audio_path: Union[str, Path],
			roi_path: Optional[Union[str, Path]] = None,
			batch_size: int = 64,
			hop_length_sec: Optional[float] = None,
			start_time_sec: float = 0.0,
			end_time_sec: Optional[float] = None,
			return_energy: bool = False,
			compute_audio_sha256: bool = False,
	) -> LatentSequence:
		"""
		Encode a single audio clip into a time-indexed latent sequence.

		See ``encode_clip_to_latent_sequence`` for parameter details.
		"""
		audio_path = Path(audio_path)
		params = self.params
		fs_target = int(self.fs_target)
		window_length_sec = float(self.window_length_sec)

		if hop_length_sec is None:
			hop_length_sec = window_length_sec
		hop_length_sec = float(hop_length_sec)
		if hop_length_sec <= 0:
			raise ValueError("hop_length_sec must be positive.")

		fs_in, audio = wavfile.read(audio_path.as_posix())
		audio = _to_float_audio(_as_mono(audio))
		audio = audio - float(np.mean(audio))
		audio = _resample_audio(audio, int(fs_in), fs_target)

		duration_sec = float(len(audio) / fs_target) if fs_target else 0.0
		if duration_sec <= 0:
			raise ValueError("Audio duration is zero after loading/resampling.")

		start_time_sec = float(start_time_sec)
		if end_time_sec is None:
			end_time_sec = duration_sec
		end_time_sec = float(end_time_sec)
		if start_time_sec < 0 or end_time_sec <= 0:
			raise ValueError("start_time_sec/end_time_sec must be positive.")
		if end_time_sec <= start_time_sec:
			raise ValueError("end_time_sec must be greater than start_time_sec.")

		end_time_sec = min(end_time_sec, duration_sec)

		if roi_path is not None:
			rois = _load_rois(roi_path)
			start_times = _window_starts_from_rois(
				rois,
				window_length_sec=window_length_sec,
				hop_length_sec=hop_length_sec,
				start_time_sec=start_time_sec,
				end_time_sec=end_time_sec,
			)
		else:
			latest = end_time_sec - window_length_sec
			if latest >= start_time_sec:
				start_times = np.arange(
					start_time_sec,
					latest + 1e-12,
					hop_length_sec,
					dtype=np.float64,
				)
			else:
				start_times = np.array([start_time_sec], dtype=np.float64)

		if start_times.size == 0:
			raise ValueError("No windows available after ROI/time filtering.")

		f_frames, t_frames, zxx = stft(
			audio,
			fs=fs_target,
			nperseg=int(params["nperseg"]),
			noverlap=int(params["noverlap"]),
		)
		spec_log = np.log(np.abs(zxx) + EPSILON)
		interp = RegularGridInterpolator(
			(f_frames, t_frames),
			spec_log,
			bounds_error=False,
			fill_value=-1.0 / EPSILON,
		)

		norm_center, norm_scale = _compute_normalization_stats(
			interp=interp,
			target_freqs=self.target_freqs,
			start_times=start_times,
			window_length_sec=window_length_sec,
			num_time_bins=self.num_time_bins,
			params=params,
		)

		time_offsets = self.time_offsets
		mu_chunks: list[np.ndarray] = []
		logvar_chunks: list[np.ndarray] = []
		energy_chunks: list[np.ndarray] = []

		model = self.model
		with torch.inference_mode():
			for start in range(0, start_times.size, int(batch_size)):
				stop = min(start + int(batch_size), start_times.size)
				batch_starts = start_times[start:stop]
				target_times = batch_starts[:, None] + time_offsets[None, :]
				freq_grid = np.broadcast_to(
					self.target_freqs[:, None, None],
					(self.target_freqs.size, batch_starts.size, self.num_time_bins),
				)
				time_grid = np.broadcast_to(
					target_times[None, :, :],
					(self.target_freqs.size, batch_starts.size, self.num_time_bins),
				)
				points = np.stack([freq_grid, time_grid], axis=-1)
				spec = interp(points).transpose(1, 0, 2)
				spec = (spec - float(params["spec_min_val"])) / (
					float(params["spec_max_val"]) - float(params["spec_min_val"])
				)
				spec = np.clip(spec, 0.0, 1.0)
				if norm_center is not None and norm_scale is not None:
					spec = (spec - norm_center) / norm_scale
				spec = np.asarray(spec, dtype=np.float32)
				x = torch.from_numpy(spec).to(model.device)
				mu, logvar, _ = model.encode(x)
				mu_chunks.append(mu.detach().cpu().to(dtype=torch.float32).numpy())
				logvar_chunks.append(logvar.detach().cpu().to(dtype=torch.float32).numpy())
				if return_energy:
					s1 = np.floor(batch_starts * fs_target).astype(int)
					s2 = np.floor((batch_starts + window_length_sec) * fs_target).astype(int)
					s1 = np.clip(s1, 0, len(audio))
					s2 = np.clip(s2, 0, len(audio))
					energies = np.zeros(len(batch_starts), dtype=np.float32)
					for i, (a, b) in enumerate(zip(s1, s2)):
						if b <= a:
							continue
						chunk = audio[a:b]
						energies[i] = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
					energy_chunks.append(energies)

		mu_arr = np.concatenate(mu_chunks, axis=0)
		logvar_arr = np.concatenate(logvar_chunks, axis=0)
		if mu_arr.shape[0] != start_times.shape[0]:
			raise RuntimeError("Internal error: latent count does not match timestamps.")

		energy_arr = None
		if return_energy:
			energy_arr = np.concatenate(energy_chunks, axis=0) if energy_chunks else None

		metadata: Dict[str, Any] = {
			"schema_version": SCHEMA_VERSION,
			"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
			"clip_id": audio_path.stem,
			"audio_path": audio_path.as_posix(),
			"audio_sha256": _sha256_file(audio_path) if compute_audio_sha256 else None,
			"sample_rate_hz": fs_target,
			"roi_path": Path(roi_path).as_posix() if roi_path is not None else None,
			"config_path": self.config_path.as_posix() if self.config_path is not None else None,
			"checkpoint_path": self.checkpoint_path.as_posix(),
		}

		return LatentSequence(
			start_times_sec=start_times,
			window_length_sec=window_length_sec,
			hop_length_sec=hop_length_sec,
			mu=mu_arr,
			logvar=logvar_arr,
			energy=energy_arr,
			metadata=metadata,
		)


def encode_clip_to_latent_sequence(
		checkpoint_path: Union[str, Path],
		config: Union[str, Path, FixedWindowExperimentConfig],
		audio_path: Union[str, Path],
		roi_path: Optional[Union[str, Path]] = None,
		device: str = "auto",
		batch_size: int = 64,
		hop_length_sec: Optional[float] = None,
		start_time_sec: float = 0.0,
		end_time_sec: Optional[float] = None,
		return_energy: bool = False,
		compute_audio_sha256: bool = False,
) -> LatentSequence:
	"""
	Encode a single audio clip into a time-indexed latent sequence.

	Parameters
	----------
	checkpoint_path:
		Path to a ``.tar`` checkpoint saved by ``VAE.save_state``.
	config:
		Fixed-window experiment config (YAML path or loaded config).
	audio_path:
		Path to a wav file.
	roi_path:
		Optional ROI file containing onset/offset times in seconds. If provided,
		windows are sampled only within ROIs.
	device:
		{"cpu", "cuda", "auto"}.
	batch_size:
		Number of windows to encode per forward pass.
	hop_length_sec:
		Step between consecutive windows. Defaults to ``window_length_sec``.
	start_time_sec / end_time_sec:
		Time range (seconds) to consider within the clip before ROI filtering.
	return_energy:
		If ``True``, also compute per-window RMS energy from the resampled audio.
	compute_audio_sha256:
		If ``True``, compute SHA-256 of the audio file for provenance.
	"""
	encoder = LatentSequenceEncoder(
		checkpoint_path=checkpoint_path,
		config=config,
		device=device,
	)
	return encoder.encode(
		audio_path=audio_path,
		roi_path=roi_path,
		batch_size=batch_size,
		hop_length_sec=hop_length_sec,
		start_time_sec=start_time_sec,
		end_time_sec=end_time_sec,
		return_energy=return_energy,
		compute_audio_sha256=compute_audio_sha256,
	)
