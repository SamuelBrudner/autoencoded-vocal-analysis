"""Preflight checks for fixed-window training against ROI durations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

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


def _load_parquet_roi_durations(roi_parquet_path: str) -> np.ndarray:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet ROI preflight requires pyarrow. Install pyarrow or skip parquet preflight."
        ) from exc

    table = pq.read_table(
        str(roi_parquet_path),
        columns=["onsets_sec", "offsets_sec"],
    )
    payload = table.to_pydict()
    onsets = payload.get("onsets_sec") or []
    offsets = payload.get("offsets_sec") or []
    if len(onsets) != len(offsets):
        raise ValueError(f"Malformed ROI parquet: {roi_parquet_path}")

    durations: list[np.ndarray] = []
    for ons, offs in zip(onsets, offsets):
        ons = ons or []
        offs = offs or []
        count = min(len(ons), len(offs))
        if count <= 0:
            continue
        ons_arr = np.asarray(ons[:count], dtype=np.float64)
        offs_arr = np.asarray(offs[:count], dtype=np.float64)
        dur = offs_arr - ons_arr
        dur = dur[np.isfinite(dur)]
        dur = dur[dur > 0.0]
        if dur.size:
            durations.append(dur)

    if not durations:
        return np.array([], dtype=np.float64)
    return np.concatenate(durations)


def sample_parquet_roi_duration_stats(
    roi_parquet_files: Sequence[str],
    *,
    window_length: float,
    sample_dirs: int = 25,
    sample_segments: int = 5000,
    seed: int = 0,
) -> dict:
    """Sample ROI durations from parquet bundles and return compatibility stats."""
    if not roi_parquet_files:
        raise ValueError("No parquet ROI bundle paths were provided for preflight.")
    sample_dirs = int(sample_dirs)
    if sample_dirs <= 0:
        raise ValueError("sample_dirs must be positive.")
    sample_segments = int(sample_segments)
    if sample_segments <= 0:
        raise ValueError("sample_segments must be positive.")

    rng = np.random.default_rng(int(seed))
    sample_n = min(sample_dirs, len(roi_parquet_files))
    indices = rng.choice(len(roi_parquet_files), size=sample_n, replace=False)
    sampled_paths = [str(roi_parquet_files[int(i)]) for i in indices]

    missing = 0
    empty = 0
    durations = []
    total_durations = 0
    for path in sampled_paths:
        roi_file = Path(path)
        if not roi_file.exists():
            missing += 1
            continue
        dur = _load_parquet_roi_durations(roi_file.as_posix())
        if dur.size == 0:
            empty += 1
            continue
        durations.append(dur)
        total_durations += int(dur.size)

    if not durations:
        raise ValueError(
            "No positive ROI durations found in sampled parquet bundles. "
            f"sampled={sample_n}, missing={missing}, empty={empty}."
        )

    all_durations = np.concatenate(durations)
    if all_durations.size > sample_segments:
        keep = rng.choice(all_durations.size, size=sample_segments, replace=False)
        all_durations = all_durations[keep]

    compatible = all_durations >= (float(window_length) - EPSILON)
    return {
        "window_length_sec": float(window_length),
        "parquet_files_total": int(len(roi_parquet_files)),
        "parquet_files_sampled": int(sample_n),
        "parquet_files_missing_in_sample": int(missing),
        "parquet_files_empty_in_sample": int(empty),
        "roi_segments_total_in_sampled_files": int(total_durations),
        "roi_segments_sampled": int(all_durations.size),
        "roi_segments_compatible": int(np.count_nonzero(compatible)),
        "compatible_fraction": float(np.mean(compatible)),
        "min_duration_sec": float(np.min(all_durations)),
        "p05_duration_sec": float(np.percentile(all_durations, 5.0)),
        "median_duration_sec": float(np.median(all_durations)),
        "p95_duration_sec": float(np.percentile(all_durations, 95.0)),
        "max_duration_sec": float(np.max(all_durations)),
    }


def assert_window_length_compatible_parquet_sample(
    roi_parquet_files: Sequence[str],
    *,
    window_length: float,
    sample_dirs: int = 25,
    sample_segments: int = 5000,
    seed: int = 0,
) -> dict:
    stats = sample_parquet_roi_duration_stats(
        roi_parquet_files,
        window_length=window_length,
        sample_dirs=sample_dirs,
        sample_segments=sample_segments,
        seed=seed,
    )
    if stats["roi_segments_compatible"] == 0:
        raise ValueError(
            "window_length is incompatible with ROI durations (parquet sample): "
            f"window_length={window_length:.5f}s, "
            f"max_duration={stats['max_duration_sec']:.5f}s, "
            f"roi_segments_sampled={stats['roi_segments_sampled']}. "
            "Use a shorter window_length or regenerate ROIs with longer segments."
        )
    return stats
