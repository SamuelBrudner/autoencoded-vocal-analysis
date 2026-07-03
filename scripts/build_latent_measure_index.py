#!/usr/bin/env python3
"""Index frozen latent sequence exports into developmental measures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


TUTOR_DAY_RE = re.compile(r"\bday\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_tutor_start_day(entry: Dict[str, Any]) -> Optional[float]:
    direct = _finite_float(entry.get("tutor_start_day"))
    if direct is not None:
        return direct
    for key in ("top_dir", "pre_bird_path", "audio_dir_rel"):
        value = entry.get(key)
        if not value:
            continue
        match = TUTOR_DAY_RE.search(str(value))
        if match:
            return float(match.group(1))
    return None


def _load_json(path: Path) -> Dict[str, Any]:
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


def _read_clip(npz_path: Path, latent_dir: Path) -> Dict[str, Any]:
    json_path = npz_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing metadata JSON for {npz_path.as_posix()}")

    meta = _load_json(json_path)
    entry = meta.get("entry") or {}
    if not isinstance(entry, dict):
        entry = {}

    with np.load(npz_path.as_posix()) as arrays:
        if "mu" not in arrays.files:
            raise KeyError(f"Missing mu array in {npz_path.as_posix()}")
        mu = arrays["mu"]
        if mu.ndim != 2:
            raise ValueError(f"mu must be time-major [T, z_dim] in {npz_path.as_posix()}")
        n_windows = int(mu.shape[0])
        z_dim = int(mu.shape[1])
        has_logvar = "logvar" in arrays.files
        if has_logvar and arrays["logvar"].shape != mu.shape:
            raise ValueError(
                f"logvar shape does not match mu in {npz_path.as_posix()}"
            )
        has_energy = "energy" in arrays.files
        window_length_sec = _scalar(arrays, "window_length_sec")
        hop_length_sec = _scalar(arrays, "hop_length_sec")

    bird_id_norm = entry.get("bird_id_norm") or meta.get("bird_id_norm")
    regime = entry.get("regime") or meta.get("regime")
    split = entry.get("split") or meta.get("split")
    dph = _finite_float(entry.get("dph", meta.get("dph")))
    tutor_start_day = _parse_tutor_start_day(entry)
    days_since_tutor = None
    if dph is not None and tutor_start_day is not None:
        days_since_tutor = float(dph) - float(tutor_start_day)

    return {
        "clip_id": meta.get("clip_id") or npz_path.relative_to(latent_dir).with_suffix("").as_posix(),
        "npz_path": npz_path.as_posix(),
        "json_path": json_path.as_posix(),
        "relative_npz_path": npz_path.relative_to(latent_dir).as_posix(),
        "schema_version": meta.get("schema_version"),
        "bird_id_norm": bird_id_norm,
        "regime": regime,
        "split": split,
        "dph": dph,
        "tutor_start_day": tutor_start_day,
        "days_since_tutor": days_since_tutor,
        "n_windows": n_windows,
        "z_dim": z_dim,
        "has_logvar": bool(has_logvar),
        "has_energy": bool(has_energy),
        "window_length_sec": window_length_sec,
        "hop_length_sec": hop_length_sec,
    }


def _measure_key(clip: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        clip.get("bird_id_norm"),
        clip.get("regime"),
        clip.get("split"),
        clip.get("dph"),
        clip.get("tutor_start_day"),
        clip.get("days_since_tutor"),
    )


def _build_measures(clips: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for clip in clips:
        grouped[_measure_key(clip)].append(clip)

    measures: List[Dict[str, Any]] = []
    for idx, (key, group) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        bird_id_norm, regime, split, dph, tutor_start_day, days_since_tutor = key
        z_dims = sorted({int(clip["z_dim"]) for clip in group})
        measures.append(
            {
                "measure_id": f"measure_{idx:06d}",
                "bird_id_norm": bird_id_norm,
                "regime": regime,
                "split": split,
                "dph": dph,
                "tutor_start_day": tutor_start_day,
                "days_since_tutor": days_since_tutor,
                "n_clips": len(group),
                "total_windows": sum(int(clip["n_windows"]) for clip in group),
                "z_dims": z_dims,
                "all_have_logvar": all(bool(clip["has_logvar"]) for clip in group),
                "any_have_energy": any(bool(clip["has_energy"]) for clip in group),
                "clip_ids": [str(clip["clip_id"]) for clip in sorted(group, key=lambda c: str(c["clip_id"]))],
            }
        )
    return measures


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-destructive index of frozen latent measures."
    )
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--clips-csv", type=Path, default=None)
    parser.add_argument("--measures-csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.latent_dir.exists():
        raise FileNotFoundError(args.latent_dir.as_posix())

    npz_paths = sorted(args.latent_dir.rglob("*.npz"))
    if not npz_paths:
        raise ValueError(f"No .npz latent sequence files found under {args.latent_dir}")

    clips = [_read_clip(path, latent_dir=args.latent_dir) for path in npz_paths]
    measures = _build_measures(clips)
    payload = {
        "latent_dir": args.latent_dir.as_posix(),
        "n_clips": len(clips),
        "n_measures": len(measures),
        "clips": clips,
        "measures": measures,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.clips_csv is not None:
        _write_csv(
            args.clips_csv,
            clips,
            fieldnames=[
                "clip_id",
                "relative_npz_path",
                "bird_id_norm",
                "regime",
                "split",
                "dph",
                "tutor_start_day",
                "days_since_tutor",
                "n_windows",
                "z_dim",
                "has_logvar",
                "has_energy",
                "window_length_sec",
                "hop_length_sec",
            ],
        )

    if args.measures_csv is not None:
        csv_measures = []
        for measure in measures:
            row = dict(measure)
            row["z_dims"] = ",".join(str(value) for value in measure["z_dims"])
            row["clip_ids"] = ",".join(measure["clip_ids"])
            csv_measures.append(row)
        _write_csv(
            args.measures_csv,
            csv_measures,
            fieldnames=[
                "measure_id",
                "bird_id_norm",
                "regime",
                "split",
                "dph",
                "tutor_start_day",
                "days_since_tutor",
                "n_clips",
                "total_windows",
                "z_dims",
                "all_have_logvar",
                "any_have_energy",
                "clip_ids",
            ],
        )

    print(
        f"Indexed {len(clips)} clips into {len(measures)} measures "
        f"from {args.latent_dir.as_posix()}"
    )


if __name__ == "__main__":
    main()
