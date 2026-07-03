#!/usr/bin/env python3
"""Compare latent sequence exports across hop lengths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _parse_labeled_dir(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Latent directory labels must be non-empty.")
    return label, Path(raw_path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(arrays: np.lib.npyio.NpzFile, key: str) -> Optional[float]:
    if key not in arrays.files:
        return None
    value = np.asarray(arrays[key])
    if value.shape == ():
        return float(value)
    if value.size == 1:
        return float(value.reshape(-1)[0])
    return None


def _finite_values(values: Iterable[Optional[float]]) -> np.ndarray:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=np.float64)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def _lag1_autocorr(mu: np.ndarray) -> Optional[float]:
    if mu.shape[0] < 3:
        return None
    corrs = []
    for dim in range(mu.shape[1]):
        x = np.asarray(mu[:-1, dim], dtype=np.float64)
        y = np.asarray(mu[1:, dim], dtype=np.float64)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
        if denom <= 0 or not math.isfinite(denom):
            continue
        corr = float(np.dot(x, y) / denom)
        if math.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        return None
    return float(np.mean(corrs))


def _ar1_effective_windows(n_windows: int, lag1_autocorr: Optional[float]) -> Optional[float]:
    if lag1_autocorr is None or n_windows <= 0:
        return None
    rho = max(0.0, min(0.999, float(lag1_autocorr)))
    effective = float(n_windows) * (1.0 - rho) / (1.0 + rho)
    return min(float(n_windows), max(1.0, effective))


def _analyze_clip(npz_path: Path, latent_dir: Path, label: str) -> Dict[str, Any]:
    json_path = npz_path.with_suffix(".json")
    meta = _load_json(json_path)
    clip_id = meta.get("clip_id")
    if not clip_id:
        clip_id = npz_path.relative_to(latent_dir).with_suffix("").as_posix()

    with np.load(npz_path.as_posix()) as arrays:
        if "mu" not in arrays.files:
            raise KeyError(f"Missing mu array in {npz_path.as_posix()}")
        mu = np.asarray(arrays["mu"], dtype=np.float64)
        if mu.ndim != 2:
            raise ValueError(f"mu must be [T, z_dim] in {npz_path.as_posix()}")
        n_windows = int(mu.shape[0])
        z_dim = int(mu.shape[1])
        hop_length_sec = _scalar(arrays, "hop_length_sec")
        window_length_sec = _scalar(arrays, "window_length_sec")
        start_times = np.asarray(arrays["start_times_sec"], dtype=np.float64) if "start_times_sec" in arrays.files else None

    if n_windows >= 2:
        steps = np.linalg.norm(np.diff(mu, axis=0), axis=1)
        mean_step_l2 = float(np.mean(steps))
        median_step_l2 = float(np.median(steps))
        p95_step_l2 = float(np.percentile(steps, 95))
        path_length_l2 = float(np.sum(steps))
        displacement_l2 = float(np.linalg.norm(mu[-1] - mu[0]))
        path_efficiency = (
            displacement_l2 / path_length_l2 if path_length_l2 > 0 else None
        )
    else:
        mean_step_l2 = None
        median_step_l2 = None
        p95_step_l2 = None
        path_length_l2 = None
        displacement_l2 = None
        path_efficiency = None

    if start_times is not None and start_times.size:
        if window_length_sec is not None:
            covered_duration_sec = (
                float(start_times[-1]) - float(start_times[0]) + float(window_length_sec)
            )
        else:
            covered_duration_sec = float(start_times[-1]) - float(start_times[0])
    else:
        covered_duration_sec = None

    lag1 = _lag1_autocorr(mu)
    effective_windows = _ar1_effective_windows(n_windows, lag1)
    effective_fraction = (
        effective_windows / float(n_windows)
        if effective_windows is not None and n_windows > 0
        else None
    )

    return {
        "label": label,
        "clip_id": clip_id,
        "relative_npz_path": npz_path.relative_to(latent_dir).as_posix(),
        "n_windows": n_windows,
        "z_dim": z_dim,
        "window_length_sec": window_length_sec,
        "hop_length_sec": hop_length_sec,
        "covered_duration_sec": covered_duration_sec,
        "mean_step_l2": mean_step_l2,
        "median_step_l2": median_step_l2,
        "p95_step_l2": p95_step_l2,
        "path_length_l2": path_length_l2,
        "displacement_l2": displacement_l2,
        "path_efficiency": path_efficiency,
        "lag1_autocorr": lag1,
        "ar1_effective_windows": effective_windows,
        "ar1_effective_fraction": effective_fraction,
    }


def _summarize(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "label": label,
        "n_clips": len(rows),
        "total_windows": int(sum(int(row["n_windows"]) for row in rows)),
        "median_windows_per_clip": _median(row["n_windows"] for row in rows),
        "median_window_length_sec": _median(row["window_length_sec"] for row in rows),
        "median_hop_length_sec": _median(row["hop_length_sec"] for row in rows),
        "median_mean_step_l2": _median(row["mean_step_l2"] for row in rows),
        "median_path_length_l2": _median(row["path_length_l2"] for row in rows),
        "median_path_efficiency": _median(row["path_efficiency"] for row in rows),
        "median_lag1_autocorr": _median(row["lag1_autocorr"] for row in rows),
        "p95_lag1_autocorr": _percentile(
            (row["lag1_autocorr"] for row in rows),
            q=95,
        ),
        "median_ar1_effective_windows": _median(row["ar1_effective_windows"] for row in rows),
        "median_ar1_effective_fraction": _median(row["ar1_effective_fraction"] for row in rows),
        "mean_ar1_effective_fraction": _mean(row["ar1_effective_fraction"] for row in rows),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare latent-sequence path metrics across export hop lengths."
    )
    parser.add_argument(
        "--latent-dir",
        action="append",
        required=True,
        help="Latent export directory, optionally labeled as LABEL=PATH. Repeatable.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--clips-csv", type=Path, default=None)
    args = parser.parse_args()

    labeled_dirs = [_parse_labeled_dir(value) for value in args.latent_dir]
    labels = [label for label, _ in labeled_dirs]
    if len(labels) != len(set(labels)):
        raise ValueError("Latent directory labels must be unique.")

    all_rows: List[Dict[str, Any]] = []
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for label, latent_dir in labeled_dirs:
        if not latent_dir.exists():
            raise FileNotFoundError(latent_dir.as_posix())
        paths = sorted(latent_dir.rglob("*.npz"))
        if not paths:
            raise ValueError(f"No .npz files found under {latent_dir.as_posix()}")
        rows = [_analyze_clip(path, latent_dir=latent_dir, label=label) for path in paths]
        rows_by_label[label] = rows
        all_rows.extend(rows)

    summaries = [_summarize(label, rows_by_label[label]) for label in labels]
    clip_sets = [
        {str(row["clip_id"]) for row in rows_by_label[label]}
        for label in labels
    ]
    common_clip_ids = sorted(set.intersection(*clip_sets)) if clip_sets else []
    payload = {
        "latent_dirs": [
            {"label": label, "path": path.as_posix()} for label, path in labeled_dirs
        ],
        "n_labels": len(labels),
        "common_clip_count": len(common_clip_ids),
        "common_clip_ids": common_clip_ids,
        "summaries": summaries,
        "clips": all_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.summary_csv is not None:
        _write_csv(
            args.summary_csv,
            summaries,
            fieldnames=[
                "label",
                "n_clips",
                "total_windows",
                "median_windows_per_clip",
                "median_window_length_sec",
                "median_hop_length_sec",
                "median_mean_step_l2",
                "median_path_length_l2",
                "median_path_efficiency",
                "median_lag1_autocorr",
                "p95_lag1_autocorr",
                "median_ar1_effective_windows",
                "median_ar1_effective_fraction",
                "mean_ar1_effective_fraction",
            ],
        )

    if args.clips_csv is not None:
        _write_csv(
            args.clips_csv,
            all_rows,
            fieldnames=[
                "label",
                "clip_id",
                "relative_npz_path",
                "n_windows",
                "z_dim",
                "window_length_sec",
                "hop_length_sec",
                "covered_duration_sec",
                "mean_step_l2",
                "median_step_l2",
                "p95_step_l2",
                "path_length_l2",
                "displacement_l2",
                "path_efficiency",
                "lag1_autocorr",
                "ar1_effective_windows",
                "ar1_effective_fraction",
            ],
        )

    print(
        f"Analyzed {len(all_rows)} clip exports across {len(labels)} labels; "
        f"common clips={len(common_clip_ids)}"
    )


if __name__ == "__main__":
    main()
