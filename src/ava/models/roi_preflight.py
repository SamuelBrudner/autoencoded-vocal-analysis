"""Preflight checks for fixed-window training against ROI durations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


EPSILON = 1e-9


def _load_roi_durations(roi_path: str) -> np.ndarray:
    rois = np.loadtxt(roi_path, ndmin=2)
    if rois.size == 0:
        return np.array([], dtype=np.float64)
    if rois.shape[1] < 2:
        raise ValueError(f"ROI file must have onset/offset columns: {roi_path}")
    durations = np.diff(rois[:, :2], axis=1).reshape(-1)
    durations = durations[np.isfinite(durations)]
    return durations[durations > 0.0]


def collect_roi_duration_stats(roi_files: Iterable[str], window_length: float) -> dict:
    durations = []
    for roi_path in roi_files:
        roi_file = Path(roi_path)
        if not roi_file.exists():
            continue
        durations.append(_load_roi_durations(roi_file.as_posix()))

    if not durations:
        raise ValueError("No ROI files were available for duration preflight.")

    all_durations = np.concatenate(durations)
    if all_durations.size == 0:
        raise ValueError(
            "No positive ROI durations found. Check ROI generation outputs before training."
        )

    compatible = all_durations >= (window_length - EPSILON)
    return {
        "window_length_sec": float(window_length),
        "roi_segments_total": int(all_durations.size),
        "roi_segments_compatible": int(np.count_nonzero(compatible)),
        "compatible_fraction": float(np.mean(compatible)),
        "min_duration_sec": float(np.min(all_durations)),
        "p05_duration_sec": float(np.percentile(all_durations, 5.0)),
        "median_duration_sec": float(np.median(all_durations)),
        "p95_duration_sec": float(np.percentile(all_durations, 95.0)),
        "max_duration_sec": float(np.max(all_durations)),
    }


def assert_window_length_compatible(roi_files: Iterable[str], window_length: float) -> dict:
    stats = collect_roi_duration_stats(roi_files, window_length)
    if stats["roi_segments_compatible"] == 0:
        raise ValueError(
            "window_length is incompatible with ROI durations: "
            f"window_length={window_length:.5f}s, "
            f"max_duration={stats['max_duration_sec']:.5f}s, "
            f"roi_segments={stats['roi_segments_total']}. "
            "Use a shorter window_length or regenerate ROIs with longer segments."
        )
    return stats
