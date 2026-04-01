"""
Sequence datasets for contiguous whole-file training.

This module keeps the shotgun-VAE spectrogram window representation, but
returns each file as an ordered sequence of overlapping windows so recurrent
models can learn latent dynamics over contiguous audio.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from typing import Optional
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import torch
from torch.utils.data import DataLoader, Dataset

EPSILON = 1e-9


def numpy_to_tensor(x):
    if torch.is_tensor(x):
        return x.to(dtype=torch.float32)
    try:
        return torch.from_numpy(x).type(torch.FloatTensor)
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        return torch.tensor(np.asarray(x).tolist(), dtype=torch.float32)


def _default_num_workers(max_workers: int = 8) -> int:
    cpu_count = os.cpu_count() or 1
    return max(0, min(max_workers, cpu_count - 1))


def _normalize_sequence_hop_length(value: Optional[float], window_length: float) -> float:
    if value is None:
        value = window_length / 2.0
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sequence_hop_length must be a positive number.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("sequence_hop_length must be a positive number.")
    return value


def _load_roi_bounds(filename: str) -> Optional[tuple[float, float]]:
    try:
        rois = np.loadtxt(filename, ndmin=2)
    except (OSError, ValueError):
        return None
    rois = np.asarray(rois, dtype=np.float64)
    if rois.size == 0:
        return None
    rois = np.atleast_2d(rois)
    if rois.shape[1] < 2:
        return None
    rois = rois[:, :2]
    valid = (
        np.isfinite(rois[:, 0])
        & np.isfinite(rois[:, 1])
        & (rois[:, 1] > rois[:, 0])
    )
    rois = rois[valid]
    if rois.size == 0:
        return None
    return float(np.min(rois[:, 0])), float(np.max(rois[:, 1]))


def _compute_window_starts(
    onset: float,
    offset: float,
    window_length: float,
    hop_length: float,
) -> np.ndarray:
    latest_start = float(offset) - float(window_length)
    if latest_start < float(onset) - EPSILON:
        return np.zeros((0,), dtype=np.float64)
    starts = np.arange(
        float(onset),
        latest_start + 1e-12,
        float(hop_length),
        dtype=np.float64,
    )
    if starts.size == 0:
        starts = np.asarray([float(onset)], dtype=np.float64)
    elif latest_start - float(starts[-1]) > 1e-9:
        starts = np.concatenate(
            [starts, np.asarray([latest_start], dtype=np.float64)],
            axis=0,
        )
    return np.unique(starts)


def pad_sequence_batch(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_length = max(int(item["x"].shape[0]) for item in batch)
    batch_size = len(batch)
    sample_shape = tuple(batch[0]["x"].shape[1:])
    x = batch[0]["x"].new_zeros((batch_size, max_length, *sample_shape))
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    start_times = torch.zeros((batch_size, max_length), dtype=torch.float32)
    lengths = torch.zeros((batch_size,), dtype=torch.long)
    audio_filenames: list[str] = []
    roi_filenames: list[str] = []
    for batch_index, item in enumerate(batch):
        length = int(item["x"].shape[0])
        x[batch_index, :length] = item["x"]
        mask[batch_index, :length] = True
        start_times[batch_index, :length] = item["start_times"]
        lengths[batch_index] = length
        audio_filenames.append(str(item["audio_filename"]))
        roi_filenames.append(str(item["roi_filename"]))
    return {
        "x": x,
        "mask": mask,
        "start_times": start_times,
        "lengths": lengths,
        "audio_filenames": audio_filenames,
        "roi_filenames": roi_filenames,
    }


class FixedWindowSequenceDataset(Dataset):
    """
    Dataset returning each file as an ordered sequence of fixed-duration windows.

    The file-level sequence spans from the first ROI onset to the last ROI
    offset, sampled at ``sequence_hop_length``. Each time step is the same
    spectrogram window shape consumed by the shotgun VAE.
    """

    def __init__(
        self,
        audio_filenames,
        roi_filenames,
        p: dict,
        transform=numpy_to_tensor,
        shoulder: Optional[float] = None,
        audio_cache_size: int = 0,
    ) -> None:
        sorted_pairs = sorted(zip(audio_filenames, roi_filenames), key=lambda pair: pair[0])
        self.filenames = []
        self.roi_filenames = []
        self.sequence_start_times: list[np.ndarray] = []
        self.sequence_bounds: list[tuple[float, float]] = []
        self.p = dict(p)
        self.transform = transform
        self.fs = self.p.get("fs")
        window_length = self.p.get("window_length")
        if window_length is None:
            raise ValueError("FixedWindowSequenceDataset requires p['window_length'].")
        self.window_length = float(window_length)
        self.hop_length = _normalize_sequence_hop_length(
            self.p.get("sequence_hop_length"),
            self.window_length,
        )
        if shoulder is None:
            shoulder = self.p.get("sequence_shoulder", 0.05)
        self.shoulder = float(shoulder)
        self.audio_cache_size = max(int(audio_cache_size), 0)
        self._audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        for audio_filename, roi_filename in sorted_pairs:
            bounds = _load_roi_bounds(str(roi_filename))
            if bounds is None:
                warnings.warn(
                    f"Could not derive ROI bounds from {roi_filename}; skipping {audio_filename}.",
                    UserWarning,
                )
                continue
            onset, offset = bounds
            starts = _compute_window_starts(
                onset=onset,
                offset=offset,
                window_length=self.window_length,
                hop_length=self.hop_length,
            )
            if starts.size == 0:
                warnings.warn(
                    "No valid contiguous windows found for "
                    f"{audio_filename} with window_length={self.window_length}.",
                    UserWarning,
                )
                continue
            self.filenames.append(str(audio_filename))
            self.roi_filenames.append(str(roi_filename))
            self.sequence_start_times.append(starts)
            self.sequence_bounds.append((onset, offset))
        if not self.filenames:
            raise ValueError(
                "No files produced valid contiguous sequences. Check ROI files, "
                "window_length, and sequence_hop_length."
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def _read_wav(self, filename: str) -> tuple[int, np.ndarray]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=WavFileWarning)
            try:
                fs, audio = wavfile.read(filename, mmap=True)
            except TypeError:
                fs, audio = wavfile.read(filename)
        return fs, audio

    def _get_audio(self, filename: str) -> np.ndarray:
        if self.audio_cache_size > 0 and filename in self._audio_cache:
            self._audio_cache.move_to_end(filename)
            return self._audio_cache[filename]
        fs, audio = self._read_wav(filename)
        if self.fs is None:
            self.fs = fs
        elif fs != self.fs:
            warnings.warn(
                f"Found inconsistent sample rate for {filename}.",
                UserWarning,
            )
        if self.audio_cache_size > 0:
            self._audio_cache[filename] = audio
            self._audio_cache.move_to_end(filename)
            while len(self._audio_cache) > self.audio_cache_size:
                self._audio_cache.popitem(last=False)
        return audio

    def _compute_window_spec(self, audio: np.ndarray, onset: float) -> torch.Tensor:
        offset = onset + self.window_length
        target_times = np.linspace(
            onset,
            offset,
            int(self.p["num_time_bins"]),
            dtype=np.float64,
        )
        spec, flag = self.p["get_spec"](
            max(0.0, onset - self.shoulder),
            offset + self.shoulder,
            audio,
            self.p,
            fs=self.fs,
            target_times=target_times,
        )
        if not flag:
            spec = np.zeros(
                (int(self.p["num_freq_bins"]), int(self.p["num_time_bins"])),
                dtype=np.float32,
            )
        if self.transform is not None:
            spec = self.transform(spec)
        elif not torch.is_tensor(spec):
            spec = torch.as_tensor(spec, dtype=torch.float32)
        if not torch.is_tensor(spec):
            spec = torch.as_tensor(spec, dtype=torch.float32)
        return spec.to(dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        filename = self.filenames[index]
        starts = self.sequence_start_times[index]
        audio = self._get_audio(filename)
        specs = [self._compute_window_spec(audio, float(onset)) for onset in starts]
        x = torch.stack(specs, dim=0)
        return {
            "x": x,
            "mask": torch.ones((x.shape[0],), dtype=torch.bool),
            "start_times": torch.as_tensor(starts, dtype=torch.float32),
            "audio_filename": filename,
            "roi_filename": self.roi_filenames[index],
        }


def _make_loader_kwargs(
    num_workers: int,
    pin_memory: Optional[bool],
    persistent_workers: Optional[bool],
    prefetch_factor: Optional[int],
) -> dict:
    loader_kwargs = {"num_workers": int(num_workers)}
    if pin_memory is not None:
        loader_kwargs["pin_memory"] = bool(pin_memory)
    if int(num_workers) > 0:
        if persistent_workers is not None:
            loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return loader_kwargs


def get_sequence_window_data_loaders(
    partition: dict,
    p: dict,
    batch_size: int = 4,
    shuffle: tuple[bool, bool] = (True, False),
    num_workers: Optional[int] = None,
    audio_cache_size: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
) -> dict:
    """
    Get DataLoaders that emit padded whole-file spectrogram sequences.
    """
    if num_workers is None:
        num_workers = _default_num_workers()
    loader_kwargs = _make_loader_kwargs(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_dataset = FixedWindowSequenceDataset(
        partition["train"]["audio"],
        partition["train"]["rois"],
        p,
        transform=numpy_to_tensor,
        audio_cache_size=audio_cache_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle[0]),
        collate_fn=pad_sequence_batch,
        **loader_kwargs,
    )
    test_loader = None
    test_partition = partition.get("test") or {}
    if test_partition.get("audio") is not None and len(test_partition.get("audio")) > 0:
        test_dataset = FixedWindowSequenceDataset(
            test_partition["audio"],
            test_partition["rois"],
            p,
            transform=numpy_to_tensor,
            audio_cache_size=audio_cache_size,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle[1]),
            collate_fn=pad_sequence_batch,
            **loader_kwargs,
        )
    return {"train": train_loader, "test": test_loader}


__all__ = [
    "FixedWindowSequenceDataset",
    "get_sequence_window_data_loaders",
    "pad_sequence_batch",
]
