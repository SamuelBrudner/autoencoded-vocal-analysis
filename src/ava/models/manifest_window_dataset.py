"""Streaming fixed-window dataset driven by birdsong manifest entries.

This module provides a fixed-window dataset that scales to very large manifests
by sampling windows on-demand from per-directory ROI bundles (parquet) or
per-clip ROI text files, without loading all ROIs into memory at init.
"""

from __future__ import annotations

import multiprocessing as mp
import hashlib
import json
import os
import tempfile
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from torch.utils.data import DataLoader, Dataset

from ava.models.augmentations import SpectrogramAugmenter
from ava.models.utils import numpy_to_tensor


APPLEDOUBLE_PREFIX = "._"
EPSILON = 1e-9


_FILE_WEIGHT_MODE_ALIASES = {
    "duration": "duration",
    "durations": "duration",
    "roi_duration": "duration",
    "roi_durations": "duration",
    "length": "duration",
    "uniform": "uniform",
    "equal": "uniform",
    "roi_count": "roi_count",
    "n_rois": "roi_count",
    "count": "roi_count",
}

_ROI_WEIGHT_MODE_ALIASES = {
    "duration": "duration",
    "durations": "duration",
    "length": "duration",
    "uniform": "uniform",
    "equal": "uniform",
}

_NORMALIZATION_MODE_ALIASES = {
    "none": "none",
    "off": "none",
    "disable": "none",
    "global": "global",
    "dataset": "global",
    # per-file normalization requires enumerating all files; unsupported here.
    "per_file": "per_file",
    "per-file": "per_file",
    "file": "per_file",
}

_NORMALIZATION_METHOD_ALIASES = {
    "mean_std": "mean_std",
    "meanstd": "mean_std",
    "mean": "mean_std",
    "std": "mean_std",
    "zscore": "mean_std",
    "z_score": "mean_std",
    "robust": "robust",
    "median_iqr": "robust",
    "median": "robust",
    "iqr": "robust",
}


def _default_num_workers(max_workers: int = 8) -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(max_workers, cpu_count - 1))


def _normalize_mode(value: Any, default: str, aliases: dict, name: str, allow_bool: bool = False) -> str:
    if value is None:
        return default
    if allow_bool and isinstance(value, bool):
        value = "duration" if value else "uniform"
    if isinstance(value, str):
        key = value.strip().lower()
        if key in aliases:
            return aliases[key]
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{name} must be one of: {valid}.")
    raise ValueError(f"{name} must be a string.")


def _normalize_choice(value: Any, default: str, aliases: dict, name: str, allow_bool: bool = False) -> str:
    if value is None:
        return default
    if allow_bool and isinstance(value, bool):
        return "global" if value else "none"
    if isinstance(value, str):
        key = value.strip().lower()
        if key in aliases:
            return aliases[key]
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"{name} must be one of: {valid}.")
    raise ValueError(f"{name} must be a string or boolean.")


def _normalize_bool(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean.")


def _normalize_positive_int(value: Any, default: int, name: str) -> int:
    if value is None:
        return int(default)
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _normalize_weights(weights: Sequence[float], label: str) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(~np.isfinite(weights)):
        raise ValueError(f"{label} weights contain non-finite values.")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError(f"{label} weights must sum to a positive value.")
    return weights / total


def _seed_streaming_worker(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = worker_info.seed % 2**32
    dataset = worker_info.dataset
    if hasattr(dataset, "seed"):
        dataset.seed(seed)


def _list_wavs(dir_path: str) -> list[str]:
    try:
        filenames = os.listdir(dir_path)
    except FileNotFoundError:
        return []
    return [
        os.path.join(dir_path, name)
        for name in sorted(filenames)
        if name.lower().endswith(".wav") and not name.startswith(APPLEDOUBLE_PREFIX)
    ]


def _load_txt_rois(path: str) -> np.ndarray:
    rois = np.loadtxt(path, ndmin=2)
    if rois.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rois = np.asarray(rois, dtype=np.float32)
    if rois.shape[1] < 2:
        raise ValueError(f"ROI file must have onset/offset columns: {path}")
    return rois[:, :2]


@dataclass(frozen=True)
class _ParquetDirectoryIndex:
    clip_stems: Tuple[str, ...]
    clip_rois: Tuple[np.ndarray, ...]  # per-clip ROIs (already filtered)
    clip_weights: np.ndarray  # normalized weights over clips


def _load_parquet_index(
    roi_parquet_path: Path,
    window_length: float,
    file_weight_mode: str,
    file_weight_cap: Optional[float],
) -> _ParquetDirectoryIndex:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet ROI bundles require pyarrow. Install pyarrow or use txt ROIs."
        ) from exc

    table = pq.read_table(
        roi_parquet_path.as_posix(),
        columns=["clip_stem", "onsets_sec", "offsets_sec"],
    )
    payload = table.to_pydict()
    stems = payload.get("clip_stem") or []
    onsets = payload.get("onsets_sec") or []
    offsets = payload.get("offsets_sec") or []
    if not (len(stems) == len(onsets) == len(offsets)):
        raise ValueError(f"Malformed ROI parquet: {roi_parquet_path}")

    clip_stems: list[str] = []
    clip_rois: list[np.ndarray] = []
    weights: list[float] = []

    for stem, ons, offs in zip(stems, onsets, offsets):
        stem = str(stem)
        ons = ons or []
        offs = offs or []
        count = min(len(ons), len(offs))
        if count:
            rois = np.stack([np.asarray(ons[:count], dtype=np.float32), np.asarray(offs[:count], dtype=np.float32)], axis=1)
        else:
            rois = np.zeros((0, 2), dtype=np.float32)
        if rois.size:
            durations = rois[:, 1] - rois[:, 0]
            mask = (
                np.isfinite(durations)
                & (durations > 0.0)
                & (durations >= (window_length - EPSILON))
            )
            rois = rois[mask]
            durations = durations[mask]
        else:
            durations = np.zeros((0,), dtype=np.float32)

        if rois.shape[0] == 0:
            continue

        if file_weight_mode == "uniform":
            weight = 1.0
        elif file_weight_mode == "roi_count":
            weight = float(rois.shape[0])
        else:
            weight = float(np.sum(durations))

        clip_stems.append(stem)
        clip_rois.append(rois)
        weights.append(weight)

    if not clip_stems:
        raise ValueError(f"No compatible ROI segments found in {roi_parquet_path}.")

    weights_arr = np.asarray(weights, dtype=float)
    if file_weight_cap is not None:
        try:
            cap = float(file_weight_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError("file_weight_cap must be a positive number.") from exc
        if cap <= 0:
            raise ValueError("file_weight_cap must be positive.")
        weights_arr = np.minimum(weights_arr, cap)

    weights_arr = _normalize_weights(weights_arr, "File")
    return _ParquetDirectoryIndex(
        clip_stems=tuple(clip_stems),
        clip_rois=tuple(clip_rois),
        clip_weights=weights_arr,
    )


class ManifestFixedWindowDataset(Dataset):
    """Sample random fixed-length spectrogram windows from manifest entries."""

    def __init__(
        self,
        entries: Sequence[dict],
        p: dict,
        *,
        roi_format: str = "parquet",
        roi_parquet_name: str = "roi.parquet",
        transform=None,
        dataset_length: int = 2048,
        min_spec_val: Optional[float] = None,
        min_audio_energy: Optional[float] = None,
        spec_cache_dir: Optional[str] = None,
        spec_cache: Optional[dict] = None,
        audio_cache_size: int = 0,
        roi_cache_size: int = 16,
        normalization_stats: Optional[dict] = None,
        augmentations: Optional[object] = None,
        return_pair: bool = False,
        pair_format: str = "tuple",
        pair_with_original: bool = False,
    ) -> None:
        if not isinstance(p, dict):
            raise TypeError("p must be a dict of preprocessing parameters.")
        window_length = p.get("window_length")
        if window_length is None:
            raise ValueError("ManifestFixedWindowDataset requires p['window_length'].")
        self.p = p
        self.window_length = float(window_length)
        self.fs = p.get("fs")

        roi_format = (roi_format or "parquet").strip().lower()
        if roi_format not in ("parquet", "txt"):
            raise ValueError("roi_format must be 'parquet' or 'txt'.")
        self.roi_format = roi_format
        self.roi_parquet_name = str(roi_parquet_name)

        self.entries = [dict(entry) for entry in entries]
        if not self.entries:
            raise ValueError("entries must be non-empty.")

        weights = []
        kept_entries = []
        for entry in self.entries:
            audio_dir = entry.get("audio_dir")
            roi_dir = entry.get("roi_dir")
            if not audio_dir or not roi_dir:
                continue
            w = entry.get("num_files", 1)
            try:
                w = float(w)
            except (TypeError, ValueError):
                w = 1.0
            if not np.isfinite(w) or w <= 0:
                w = 1.0
            kept_entries.append(entry)
            weights.append(w)
        if not kept_entries:
            raise ValueError("entries must include audio_dir and roi_dir.")
        self.entries = kept_entries
        self.entry_weights = _normalize_weights(weights, "Entry")

        self.dataset_length = int(dataset_length)
        self.min_spec_val = min_spec_val
        self.min_audio_energy = None
        if min_audio_energy is not None:
            try:
                min_audio_energy = float(min_audio_energy)
            except (TypeError, ValueError) as exc:
                raise ValueError("min_audio_energy must be a non-negative number.") from exc
            if not np.isfinite(min_audio_energy) or min_audio_energy < 0:
                raise ValueError("min_audio_energy must be a non-negative number.")
            self.min_audio_energy = float(min_audio_energy)

        self.audio_cache_size = max(int(audio_cache_size), 0)
        self._audio_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        self.spec_cache_dir = spec_cache_dir
        self.spec_cache = spec_cache
        self._spec_cache_keys: dict[str, str] = {}
        if self.spec_cache_dir:
            os.makedirs(self.spec_cache_dir, exist_ok=True)

        self.roi_cache_size = max(int(roi_cache_size), 0)
        self._roi_cache: "OrderedDict[str, _ParquetDirectoryIndex]" = OrderedDict()
        self._wav_cache: "OrderedDict[str, list[str]]" = OrderedDict()

        self._rng = np.random.RandomState()
        sampling_seed = self.p.get("sampling_seed")
        self._sampling_seed = None
        if sampling_seed is not None:
            try:
                self._sampling_seed = int(sampling_seed)
            except (TypeError, ValueError) as exc:
                raise ValueError("sampling_seed must be an integer.") from exc

        self._log_window_indices = _normalize_bool(
            self.p.get("log_window_indices"),
            default=self._sampling_seed is not None,
            name="log_window_indices",
        )
        self.window_log: dict[int, list[dict]] = {}
        self._epoch = 0
        self._epoch_ref = mp.Value("i", 0)
        self.set_epoch(0)

        self._configure_normalization(normalization_stats)
        self._configure_augmentations(
            augmentations,
            return_pair=return_pair,
            pair_format=pair_format,
            pair_with_original=pair_with_original,
        )
        self.transform = transform if transform is not None else numpy_to_tensor

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._rng = np.random.RandomState()
            self._seed_augmentations(None)
        else:
            seed = int(seed)
            self._rng = np.random.RandomState(seed)
            self._seed_augmentations(seed)

    def set_epoch(self, epoch: int) -> None:
        try:
            epoch = int(epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError("epoch must be an integer.") from exc
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self._epoch = int(epoch)
        with self._epoch_ref.get_lock():
            self._epoch_ref.value = int(epoch)
        if self._log_window_indices:
            self.window_log.setdefault(self._epoch, [])

    def _current_epoch(self) -> int:
        with self._epoch_ref.get_lock():
            return int(self._epoch_ref.value)

    def _make_window_seed(self, epoch: int, index: int) -> int:
        payload = f"{self._sampling_seed}:{epoch}:{int(index)}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little")

    def _record_window(self, epoch: int, dataset_index: int, entry_index: int, clip_stem: str, roi_index: int, onset: float, offset: float, seed: Optional[int]) -> None:
        if not self._log_window_indices:
            return
        entry = {
            "dataset_index": int(dataset_index),
            "entry_index": int(entry_index),
            "clip_stem": str(clip_stem),
            "roi_index": int(roi_index),
            "onset": float(onset),
            "offset": float(offset),
        }
        if seed is not None:
            entry["seed"] = int(seed)
        self.window_log.setdefault(int(epoch), []).append(entry)

    def __len__(self) -> int:
        return int(self.dataset_length)

    def _read_wav(self, filename: str) -> Tuple[int, np.ndarray]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=WavFileWarning)
            try:
                fs, audio = wavfile.read(filename, mmap=True)
            except TypeError:
                fs, audio = wavfile.read(filename)
        return int(fs), audio

    def _get_audio(self, filename: str) -> np.ndarray:
        if self.audio_cache_size > 0 and filename in self._audio_cache:
            self._audio_cache.move_to_end(filename)
            return self._audio_cache[filename]
        fs, audio = self._read_wav(filename)
        if self.fs is None:
            self.fs = fs
        elif fs != self.fs:
            warnings.warn(
                f"Found inconsistent sample rate for {filename}: {fs} != {self.fs}.",
                UserWarning,
            )
        if self.audio_cache_size > 0:
            self._audio_cache[filename] = audio
            self._audio_cache.move_to_end(filename)
            while len(self._audio_cache) > self.audio_cache_size:
                self._audio_cache.popitem(last=False)
        return audio

    def _window_energy(self, audio: np.ndarray, onset: float, offset: float) -> float:
        start = int(round(onset * float(self.fs)))
        stop = int(round(offset * float(self.fs)))
        if stop <= 0 or start >= len(audio):
            return 0.0
        segment = audio[max(0, start) : min(len(audio), stop)]
        if segment.size == 0:
            return 0.0
        segment = segment.astype(np.float32)
        segment = segment - np.mean(segment)
        return float(np.mean(np.square(segment)))

    def _spec_cache_params(self) -> dict:
        spec_keys = [
            "nperseg",
            "noverlap",
            "min_freq",
            "max_freq",
            "spec_min_val",
            "spec_max_val",
            "num_freq_bins",
            "num_time_bins",
            "mel",
            "time_stretch",
            "max_dur",
            "within_syll_normalize",
            "normalize_quantile",
            "window_length",
        ]
        params = {key: self.p.get(key) for key in spec_keys}
        get_spec = self.p.get("get_spec")
        params["get_spec"] = getattr(get_spec, "__name__", repr(get_spec))
        return params

    def _make_spec_cache_key(self, filename: str, onset: float, offset: float, shoulder: float) -> str:
        if filename in self._spec_cache_keys:
            base_key = self._spec_cache_keys[filename]
        else:
            try:
                file_stats = {
                    "path": os.path.abspath(filename),
                    "mtime": os.path.getmtime(filename),
                    "size": os.path.getsize(filename),
                }
            except OSError:
                file_stats = {"path": os.path.abspath(filename), "mtime": None, "size": None}
            base_payload = {"file": file_stats, "params": self._spec_cache_params()}
            base_json = json.dumps(base_payload, sort_keys=True, default=str)
            base_key = hashlib.sha256(base_json.encode("utf-8")).hexdigest()
            self._spec_cache_keys[filename] = base_key
        window_payload = {
            "base": base_key,
            "onset": float(onset),
            "offset": float(offset),
            "shoulder": float(shoulder),
            "num_time_bins": int(self.p["num_time_bins"]),
        }
        window_json = json.dumps(window_payload, sort_keys=True, default=str)
        return hashlib.sha256(window_json.encode("utf-8")).hexdigest()

    def _spec_cache_get(self, cache_key: str) -> Optional[dict]:
        if self.spec_cache is not None and cache_key in self.spec_cache:
            return self.spec_cache[cache_key]
        if self.spec_cache_dir:
            cache_path = os.path.join(self.spec_cache_dir, cache_key + ".npz")
            if os.path.exists(cache_path):
                with np.load(cache_path) as data:
                    return {"spec": data["spec"], "amp": data["amp"]}
        return None

    def _spec_cache_set(self, cache_key: str, spec: np.ndarray, amp: np.ndarray) -> None:
        if self.spec_cache is not None:
            self.spec_cache[cache_key] = {"spec": spec, "amp": amp}
        if self.spec_cache_dir:
            cache_path = os.path.join(self.spec_cache_dir, cache_key + ".npz")
            if not os.path.exists(cache_path):
                with tempfile.NamedTemporaryFile(suffix=".npz", dir=self.spec_cache_dir, delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    np.savez_compressed(tmp_path, spec=spec, amp=amp)
                    os.replace(tmp_path, cache_path)
                except OSError:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    def _get_roi_index(self, roi_dir: str) -> _ParquetDirectoryIndex:
        key = os.path.abspath(roi_dir)
        if key in self._roi_cache:
            self._roi_cache.move_to_end(key)
            return self._roi_cache[key]
        roi_path = Path(roi_dir) / self.roi_parquet_name
        file_weight_mode = _normalize_mode(
            self.p.get("file_weight_mode"),
            default="duration",
            aliases=_FILE_WEIGHT_MODE_ALIASES,
            name="file_weight_mode",
        )
        index = _load_parquet_index(
            roi_path,
            window_length=self.window_length,
            file_weight_mode=file_weight_mode,
            file_weight_cap=self.p.get("file_weight_cap"),
        )
        if self.roi_cache_size > 0:
            self._roi_cache[key] = index
            self._roi_cache.move_to_end(key)
            while len(self._roi_cache) > self.roi_cache_size:
                self._roi_cache.popitem(last=False)
        return index

    def _draw_window(
        self,
        rng: np.random.RandomState,
        *,
        shoulder: float,
        apply_normalization: bool,
        max_attempts: int,
        return_info: bool,
    ) -> Tuple[Any, int, str, float, float, int]:
        attempts = 0
        while True:
            if attempts >= max_attempts:
                raise ValueError("Unable to sample a valid window after many attempts.")
            attempts += 1

            entry_index = int(rng.choice(len(self.entries), p=self.entry_weights))
            entry = self.entries[entry_index]
            audio_dir = str(entry["audio_dir"])
            roi_dir = str(entry["roi_dir"])

            if self.roi_format == "parquet":
                roi_parquet_path = Path(roi_dir) / self.roi_parquet_name
                if not roi_parquet_path.exists():
                    continue
                try:
                    index = self._get_roi_index(roi_dir)
                except Exception:
                    continue

                clip_idx = int(rng.choice(len(index.clip_stems), p=index.clip_weights))
                clip_stem = index.clip_stems[clip_idx]
                rois = index.clip_rois[clip_idx]
                wav_path = os.path.join(audio_dir, f"{clip_stem}.wav")
            else:
                wavs = None
                if audio_dir in self._wav_cache:
                    self._wav_cache.move_to_end(audio_dir)
                    wavs = self._wav_cache[audio_dir]
                if wavs is None:
                    wavs = _list_wavs(audio_dir)
                    if self.roi_cache_size > 0:
                        self._wav_cache[audio_dir] = wavs
                        self._wav_cache.move_to_end(audio_dir)
                        while len(self._wav_cache) > self.roi_cache_size:
                            self._wav_cache.popitem(last=False)
                if not wavs:
                    continue
                wav_path = str(wavs[int(rng.randint(len(wavs)))])
                clip_stem = Path(wav_path).stem
                roi_txt_path = os.path.join(roi_dir, f"{clip_stem}.txt")
                try:
                    rois = _load_txt_rois(roi_txt_path)
                except FileNotFoundError:
                    continue
                durations = rois[:, 1] - rois[:, 0]
                mask = (
                    np.isfinite(durations)
                    & (durations > 0.0)
                    & (durations >= (self.window_length - EPSILON))
                )
                rois = rois[mask]
                if rois.shape[0] == 0:
                    continue

            roi_weight_mode = _normalize_mode(
                self.p.get("roi_weight_mode"),
                default="duration",
                aliases=_ROI_WEIGHT_MODE_ALIASES,
                name="roi_weight_mode",
                allow_bool=True,
            )
            if roi_weight_mode == "uniform":
                roi_weights = np.ones(rois.shape[0], dtype=float)
            else:
                roi_weights = (rois[:, 1] - rois[:, 0]).astype(float)
                roi_weights[~np.isfinite(roi_weights)] = 0.0
                roi_weights = np.maximum(roi_weights, 0.0)
            try:
                roi_weights = _normalize_weights(roi_weights, "ROI")
            except ValueError:
                continue
            roi_index = int(rng.choice(rois.shape[0], p=roi_weights))
            roi = rois[roi_index]
            onset = float(roi[0] + (roi[1] - roi[0] - self.window_length) * rng.rand())
            offset = float(onset + self.window_length)

            audio = None
            if self.min_audio_energy is not None:
                try:
                    audio = self._get_audio(wav_path)
                except FileNotFoundError:
                    continue
                if self._window_energy(audio, onset, offset) < float(self.min_audio_energy):
                    continue

            target_times = np.linspace(onset, offset, int(self.p["num_time_bins"]))
            spec = None
            cache_key = None
            if self.spec_cache_dir or self.spec_cache is not None:
                cache_key = self._make_spec_cache_key(wav_path, onset, offset, shoulder)
                cached = self._spec_cache_get(cache_key)
                if cached is not None:
                    spec = cached["spec"]
            if spec is None:
                if audio is None:
                    try:
                        audio = self._get_audio(wav_path)
                    except FileNotFoundError:
                        continue
                spec, flag = self.p["get_spec"](
                    max(0.0, onset - shoulder),
                    offset + shoulder,
                    audio,
                    self.p,
                    fs=float(self.fs) if self.fs is not None else None,
                    target_times=target_times,
                )
                if cache_key is not None:
                    amp = np.sum(spec, axis=0, keepdims=True).T
                    self._spec_cache_set(cache_key, spec, amp)
            else:
                flag = True
            if not flag:
                continue
            if self.min_spec_val is not None and np.max(spec) < float(self.min_spec_val):
                continue
            if apply_normalization:
                spec = self._apply_normalization(spec)
            if return_info:
                return spec, entry_index, clip_stem, onset, offset, roi_index
            return spec, entry_index, clip_stem, onset, offset, roi_index

    def _configure_normalization(self, normalization_stats: Optional[dict]) -> None:
        self.normalization_mode = _normalize_choice(
            self.p.get("normalization_mode"),
            default="none",
            aliases=_NORMALIZATION_MODE_ALIASES,
            name="normalization_mode",
            allow_bool=True,
        )
        self.normalization_method = _normalize_choice(
            self.p.get("normalization_method"),
            default="mean_std",
            aliases=_NORMALIZATION_METHOD_ALIASES,
            name="normalization_method",
        )
        if self.normalization_mode == "none":
            self.normalization_stats = None
            self._norm_center = None
            self._norm_scale = None
            return
        if self.normalization_mode == "per_file":
            raise ValueError(
                "normalization_mode=per_file is not supported for manifest streaming datasets."
            )
        self.normalization_num_samples = _normalize_positive_int(
            self.p.get("normalization_num_samples"),
            default=128,
            name="normalization_num_samples",
        )
        seed = self.p.get("normalization_seed", 0)
        try:
            self.normalization_seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("normalization_seed must be an integer.") from exc
        if normalization_stats is None:
            normalization_stats = self._compute_normalization_stats()
        self._set_normalization_stats(normalization_stats)

    def _set_normalization_stats(self, stats: dict) -> None:
        if not isinstance(stats, dict):
            raise ValueError("normalization_stats must be a dict.")
        mode = stats.get("mode", self.normalization_mode)
        method = stats.get("method", self.normalization_method)
        if mode != self.normalization_mode:
            raise ValueError("normalization_stats mode does not match normalization_mode.")
        if method != self.normalization_method:
            raise ValueError("normalization_stats method does not match normalization_method.")
        center = float(stats.get("center"))
        scale = float(stats.get("scale"))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("normalization_stats scale must be positive.")
        self.normalization_stats = {"mode": mode, "method": method, "center": center, "scale": scale}
        self._norm_center = center
        self._norm_scale = scale

    def _compute_normalization_stats(self) -> dict:
        rng = np.random.RandomState(int(self.normalization_seed))
        if self.normalization_method == "robust":
            values = []
            for _ in range(int(self.normalization_num_samples)):
                spec, *_ = self._draw_window(
                    rng,
                    shoulder=0.05,
                    apply_normalization=False,
                    max_attempts=200,
                    return_info=False,
                )
                values.append(np.ravel(np.asarray(spec, dtype=np.float64)))
            if not values:
                raise ValueError("Failed to sample windows for normalization.")
            values = np.concatenate(values, axis=0)
            median = float(np.median(values))
            q75, q25 = np.percentile(values, [75, 25])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            return {"mode": "global", "method": "robust", "center": median, "scale": scale}

        total = 0.0
        total_sq = 0.0
        count = 0
        for _ in range(int(self.normalization_num_samples)):
            spec, *_ = self._draw_window(
                rng,
                shoulder=0.05,
                apply_normalization=False,
                max_attempts=200,
                return_info=False,
            )
            values = np.asarray(spec, dtype=np.float64)
            total += float(values.sum())
            total_sq += float(np.square(values).sum())
            count += int(values.size)
        if count == 0:
            raise ValueError("Failed to sample windows for normalization.")
        mean = float(total / count)
        var = float(total_sq / count - mean**2)
        if var < 0:
            var = 0.0
        std = float(np.sqrt(var))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        return {"mode": "global", "method": "mean_std", "center": mean, "scale": std}

    def _apply_normalization(self, spec: np.ndarray) -> np.ndarray:
        if getattr(self, "_norm_center", None) is None:
            return spec
        return (spec - float(self._norm_center)) / float(self._norm_scale)

    def _configure_augmentations(self, augmentations: Any, return_pair: bool, pair_format: str, pair_with_original: bool) -> None:
        self._augmenter = SpectrogramAugmenter(augmentations)
        self._augmentations_enabled = bool(self._augmenter.config.enabled)
        self._return_pair = _normalize_bool(return_pair, default=False, name="return_pair")
        self._pair_with_original = _normalize_bool(pair_with_original, default=False, name="pair_with_original")
        if pair_format is None:
            pair_format = "tuple"
        if not isinstance(pair_format, str):
            raise ValueError("pair_format must be a string.")
        pair_format = pair_format.strip().lower()
        if pair_format not in ("tuple", "dict"):
            raise ValueError("pair_format must be 'tuple' or 'dict'.")
        self._pair_format = pair_format
        self._augment_seed = None
        self._augment_generator = None
        self._seed_augmentations(None)

    def _seed_augmentations(self, seed: Optional[int]) -> None:
        self._augment_generator = None
        self._augment_seed = None
        if not self._augmentations_enabled:
            return
        base_seed = self._augmenter.config.seed
        if seed is not None:
            seed = int(seed)
            if base_seed is None:
                base_seed = seed
            else:
                base_seed = (int(base_seed) + seed) % 2**32
        if base_seed is None:
            return
        self._augment_seed = int(base_seed)
        generator = torch.Generator()
        generator.manual_seed(self._augment_seed)
        self._augment_generator = generator

    def _ensure_tensor(self, spec: Any) -> torch.Tensor:
        if torch.is_tensor(spec):
            return spec
        return torch.as_tensor(spec, dtype=torch.float32)

    def _apply_augmentations(self, spec: Any) -> Any:
        if not self._augmentations_enabled:
            return spec
        spec = self._ensure_tensor(spec)
        return self._augmenter(spec, generator=self._augment_generator)

    def _format_pair(self, base: Any, aug: Any) -> Any:
        if self._pair_format == "dict":
            return {"x": base, "x_aug": aug}
        return (base, aug)

    def __getitem__(self, index: int, seed: Optional[int] = None, shoulder: float = 0.05) -> Any:
        try:
            iterator = iter(index)
        except TypeError:
            iterator = None
        else:
            if isinstance(index, (str, bytes)):
                iterator = None
        if iterator is not None:
            result = []
            for offset, item in enumerate(iterator):
                item_seed = None if seed is None else int(seed) + offset
                result.append(self.__getitem__(int(item), seed=item_seed, shoulder=shoulder))
            return result

        epoch = self._current_epoch()
        if seed is not None:
            rng = np.random.RandomState(int(seed))
            window_seed = None
        elif self._sampling_seed is not None:
            window_seed = self._make_window_seed(epoch, int(index))
            rng = np.random.RandomState(window_seed)
        else:
            rng = self._rng
            window_seed = None

        spec, entry_index, clip_stem, onset, offset, roi_index = self._draw_window(
            rng,
            shoulder=float(shoulder),
            apply_normalization=True,
            max_attempts=200,
            return_info=True,
        )
        if self._log_window_indices:
            self._record_window(epoch, int(index), entry_index, clip_stem, roi_index, onset, offset, seed=window_seed)

        if self.transform:
            spec = self.transform(spec)
        if self._augmentations_enabled and not torch.is_tensor(spec):
            spec = self._ensure_tensor(spec)
        if self._return_pair:
            if self._pair_with_original:
                base = spec
                aug = self._apply_augmentations(spec)
            else:
                base = self._apply_augmentations(spec)
                aug = self._apply_augmentations(spec)
            spec = self._format_pair(base, aug)
        else:
            spec = self._apply_augmentations(spec)
        return spec


def get_manifest_fixed_window_data_loaders(
    train_entries: Sequence[dict],
    test_entries: Optional[Sequence[dict]],
    p: dict,
    *,
    roi_format: str = "parquet",
    roi_parquet_name: str = "roi.parquet",
    dataset_length: int = 2048,
    train_dataset_length: Optional[int] = None,
    test_dataset_length: Optional[int] = None,
    roi_cache_size: int = 16,
    batch_size: int = 64,
    shuffle: Tuple[bool, bool] = (True, False),
    num_workers: Optional[int] = None,
    min_spec_val: Optional[float] = None,
    min_audio_energy: Optional[float] = None,
    spec_cache_dir: Optional[str] = None,
    spec_cache: Optional[dict] = None,
    audio_cache_size: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    augmentations: Optional[object] = None,
    augmentations_eval: bool = False,
    return_pair: bool = False,
    pair_format: str = "tuple",
    pair_with_original: bool = False,
) -> dict:
    if num_workers is None:
        num_workers = _default_num_workers()
    num_workers = max(int(num_workers), 0)
    train_dataset_length = _normalize_positive_int(
        train_dataset_length,
        default=dataset_length,
        name="train_dataset_length",
    )
    test_dataset_length = _normalize_positive_int(
        test_dataset_length,
        default=dataset_length,
        name="test_dataset_length",
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if num_workers == 0:
        persistent_workers = False
    augmentations_eval = _normalize_bool(augmentations_eval, default=False, name="augmentations_eval")

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_streaming_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    normalization_stats = p.get("normalization_stats")
    train_dataset = ManifestFixedWindowDataset(
        train_entries,
        p,
        roi_format=roi_format,
        roi_parquet_name=roi_parquet_name,
        dataset_length=train_dataset_length,
        min_spec_val=min_spec_val,
        min_audio_energy=min_audio_energy,
        spec_cache_dir=spec_cache_dir,
        spec_cache=spec_cache,
        audio_cache_size=audio_cache_size,
        roi_cache_size=roi_cache_size,
        normalization_stats=normalization_stats,
        augmentations=augmentations,
        return_pair=return_pair,
        pair_format=pair_format,
        pair_with_original=pair_with_original,
    )
    if normalization_stats is None and getattr(train_dataset, "normalization_mode", None) == "global":
        normalization_stats = train_dataset.normalization_stats

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=bool(shuffle[0]), **loader_kwargs)

    if not test_entries:
        return {"train": train_loader, "test": None}

    test_augmentations = augmentations if augmentations_eval else None
    test_dataset = ManifestFixedWindowDataset(
        test_entries,
        p,
        roi_format=roi_format,
        roi_parquet_name=roi_parquet_name,
        dataset_length=test_dataset_length,
        min_spec_val=min_spec_val,
        min_audio_energy=min_audio_energy,
        spec_cache_dir=spec_cache_dir,
        spec_cache=spec_cache,
        audio_cache_size=audio_cache_size,
        roi_cache_size=roi_cache_size,
        normalization_stats=normalization_stats,
        augmentations=test_augmentations,
        return_pair=return_pair,
        pair_format=pair_format,
        pair_with_original=pair_with_original,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=bool(shuffle[1]), **loader_kwargs)
    return {"train": train_loader, "test": test_loader}
