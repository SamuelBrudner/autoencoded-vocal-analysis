#!/usr/bin/env python3
"""Generate bird-stratified train/test manifests for birdsong experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _load_metadata(path: Path):
    pandas_error = None
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None
    if pd is not None:
        try:
            return pd.read_parquet(path.as_posix())
        except Exception as exc:  # pragma: no cover - best-effort fallback
            pandas_error = exc
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        pq = None
        pyarrow_error = exc
    else:
        pyarrow_error = None
    if pq is not None:
        return pq.read_table(path.as_posix()).to_pandas()
    detail = ""
    if pandas_error:
        detail = f" pandas error: {pandas_error}"
    message = (
        "Failed to load parquet metadata. Install pandas/pyarrow to read "
        f"{path}.{detail}"
    )
    raise RuntimeError(message) from pyarrow_error


def _require_columns(df, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Metadata missing columns: {missing}")


def _resolve_dir(root: Path, rel: str) -> str:
    if rel in (".", ""):
        return root.as_posix()
    return (root / rel).as_posix()


def _split_birds(
    birds: list[str],
    rng: random.Random,
    train_fraction: float,
    train_count: Optional[int],
    test_count: Optional[int],
) -> tuple[list[str], list[str]]:
    rng.shuffle(birds)
    if train_count is not None or test_count is not None:
        if train_count is None:
            train_count = len(birds) - (test_count or 0)
        if test_count is None:
            test_count = len(birds) - train_count
        if train_count < 0 or test_count < 0:
            raise ValueError("train/test bird counts must be non-negative.")
        if train_count + test_count > len(birds):
            raise ValueError(
                "Requested more birds than available for a regime."
            )
        return birds[:train_count], birds[train_count : train_count + test_count]
    if len(birds) <= 1:
        return birds, []
    n_train = int(round(train_fraction * len(birds)))
    n_train = max(1, min(len(birds) - 1, n_train))
    return birds[:n_train], birds[n_train:]


def _summarize(entries: list[dict]) -> dict:
    by_regime = defaultdict(lambda: {"birds": set(), "directories": 0, "files": 0})
    total_files = 0
    for entry in entries:
        regime = entry.get("regime", "unknown")
        by_regime[regime]["directories"] += 1
        by_regime[regime]["files"] += int(entry.get("num_files", 0))
        bird = entry.get("bird_id_norm")
        if bird:
            by_regime[regime]["birds"].add(bird)
        total_files += int(entry.get("num_files", 0))
    by_regime_out = {}
    for regime, stats in by_regime.items():
        by_regime_out[regime] = {
            "birds": len(stats["birds"]),
            "directories": stats["directories"],
            "files": stats["files"],
        }
    return {
        "birds": len({entry.get("bird_id_norm") for entry in entries if entry.get("bird_id_norm")}),
        "directories": len(entries),
        "files": total_files,
        "by_regime": by_regime_out,
    }


def build_manifest(
    metadata_path: Path,
    root: Path,
    roi_root: Path,
    out_path: Path,
    seed: int,
    train_fraction: float,
    birds_per_regime: Optional[int],
    train_birds_per_regime: Optional[int],
    test_birds_per_regime: Optional[int],
    min_files_per_dir: int,
    max_dirs_per_bird: Optional[int],
    regimes: Optional[list[str]],
) -> dict:
    df = _load_metadata(metadata_path)
    required = [
        "audio_dir_rel",
        "bird_id_norm",
        "bird_id_raw",
        "regime",
        "top_dir",
        "pre_bird_path",
        "dph",
        "session_label",
        "file_name",
        "is_audio",
        "is_appledouble",
    ]
    _require_columns(df, required)

    df = (
        df.query(
            "is_audio and not is_appledouble and bird_id_norm == bird_id_norm "
            "and regime == regime"
        )
        .groupby("audio_dir_rel", as_index=False)
        .agg(
            num_files=("file_name", "size"),
            bird_id_norm=("bird_id_norm", "first"),
            bird_id_raw=("bird_id_raw", "first"),
            regime=("regime", "first"),
            top_dir=("top_dir", "first"),
            pre_bird_path=("pre_bird_path", "first"),
            dph=("dph", "first"),
            session_label=("session_label", "first"),
        )
    )

    if regimes:
        normed = [regime.lower() for regime in regimes]
        df = df.query("regime in @normed")

    if min_files_per_dir > 1:
        df = df.query("num_files >= @min_files_per_dir")

    if max_dirs_per_bird is not None:
        rng = np.random.default_rng(seed)
        df = (
            df.assign(_rand=rng.random(len(df)))
            .sort_values(["bird_id_norm", "_rand"])
            .groupby("bird_id_norm", group_keys=False)
            .head(max_dirs_per_bird)
            .drop(columns=["_rand"])
        )

    if df.empty:
        raise ValueError("No audio directories remain after filtering.")

    bird_regime_counts = df.groupby("bird_id_norm")["regime"].nunique()
    if (bird_regime_counts > 1).any():
        raise ValueError(
            "Some birds appear in multiple regimes; filter metadata to avoid "
            "leakage before splitting."
        )

    audio_dir_abs = [
        _resolve_dir(root, rel) for rel in df["audio_dir_rel"].tolist()
    ]
    roi_dir_abs = [
        _resolve_dir(roi_root, rel) for rel in df["audio_dir_rel"].tolist()
    ]
    df = df.assign(audio_dir=audio_dir_abs, roi_dir=roi_dir_abs)

    rng = random.Random(seed)
    train_birds: set[str] = set()
    test_birds: set[str] = set()
    bird_sets = {}

    for regime, group in df.groupby("regime", sort=True):
        birds = sorted(group["bird_id_norm"].dropna().unique().tolist())
        if birds_per_regime is not None:
            rng.shuffle(birds)
            birds = birds[: birds_per_regime]
        if not birds:
            continue
        train, test = _split_birds(
            birds=birds,
            rng=rng,
            train_fraction=train_fraction,
            train_count=train_birds_per_regime,
            test_count=test_birds_per_regime,
        )
        train_birds.update(train)
        test_birds.update(test)
        bird_sets[regime] = {"train": train, "test": test}

    overlap = train_birds.intersection(test_birds)
    if overlap:
        raise ValueError(
            f"Bird overlap detected in train/test split: {sorted(overlap)}"
        )

    base_cols = [
        "audio_dir_rel",
        "audio_dir",
        "roi_dir",
        "bird_id_norm",
        "bird_id_raw",
        "regime",
        "top_dir",
        "pre_bird_path",
        "dph",
        "session_label",
        "num_files",
    ]

    train_entries = (
        df.query("bird_id_norm in @train_birds")
        .assign(split="train")
        .sort_values(["regime", "bird_id_norm", "audio_dir_rel"])
        .loc[:, base_cols + ["split"]]
        .to_dict(orient="records")
    )
    test_entries = (
        df.query("bird_id_norm in @test_birds")
        .assign(split="test")
        .sort_values(["regime", "bird_id_norm", "audio_dir_rel"])
        .loc[:, base_cols + ["split"]]
        .to_dict(orient="records")
    )

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata_path": metadata_path.as_posix(),
        "root": root.as_posix(),
        "roi_root": roi_root.as_posix(),
        "seed": seed,
        "train_fraction": train_fraction,
        "filters": {
            "birds_per_regime": birds_per_regime,
            "train_birds_per_regime": train_birds_per_regime,
            "test_birds_per_regime": test_birds_per_regime,
            "min_files_per_dir": min_files_per_dir,
            "max_dirs_per_bird": max_dirs_per_bird,
            "regimes": regimes,
        },
        "bird_sets": bird_sets,
        "summary": {
            "train": _summarize(train_entries),
            "test": _summarize(test_entries),
        },
        "train": train_entries,
        "test": test_entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bird-stratified manifests from birdsong metadata."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to birdsong parquet metadata.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Volumes/samsung_ssd/data/birdsong"),
        help="Root birdsong directory.",
    )
    parser.add_argument(
        "--roi-root",
        required=True,
        type=Path,
        help="Root directory for ROI outputs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output manifest JSON path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of birds assigned to train when using split mode.",
    )
    parser.add_argument("--birds-per-regime", type=int, default=None)
    parser.add_argument("--train-birds-per-regime", type=int, default=None)
    parser.add_argument("--test-birds-per-regime", type=int, default=None)
    parser.add_argument("--min-files-per-dir", type=int, default=1)
    parser.add_argument("--max-dirs-per-bird", type=int, default=None)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")
    if args.min_files_per_dir <= 0:
        raise ValueError("--min-files-per-dir must be positive.")
    if args.birds_per_regime is not None and args.birds_per_regime <= 0:
        raise ValueError("--birds-per-regime must be positive.")
    if args.train_birds_per_regime is not None and args.train_birds_per_regime < 0:
        raise ValueError("--train-birds-per-regime must be >= 0.")
    if args.test_birds_per_regime is not None and args.test_birds_per_regime < 0:
        raise ValueError("--test-birds-per-regime must be >= 0.")

    manifest = build_manifest(
        metadata_path=args.metadata,
        root=args.root,
        roi_root=args.roi_root,
        out_path=args.out,
        seed=args.seed,
        train_fraction=args.train_fraction,
        birds_per_regime=args.birds_per_regime,
        train_birds_per_regime=args.train_birds_per_regime,
        test_birds_per_regime=args.test_birds_per_regime,
        min_files_per_dir=args.min_files_per_dir,
        max_dirs_per_bird=args.max_dirs_per_bird,
        regimes=args.regimes,
    )

    print("Manifest summary:")
    print(json.dumps(manifest["summary"], indent=2))
    if args.dry_run:
        print("Dry run requested; manifest written but no further action taken.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
