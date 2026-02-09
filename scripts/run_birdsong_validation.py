#!/usr/bin/env python3
"""Run a local validation training loop for fixed-window birdsong."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.lightning_vae import train_vae
from ava.models.shotgun_vae_dataset import get_fixed_shotgun_data_loaders
from ava.models.utils import _get_wavs_from_dir


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_entry_paths(
    entry: dict,
    audio_root: Optional[Path],
    roi_root: Optional[Path],
) -> tuple[str, str]:
    audio_dir = entry.get("audio_dir")
    roi_dir = entry.get("roi_dir")
    if not audio_dir:
        rel = entry.get("audio_dir_rel")
        if rel is None or audio_root is None:
            raise ValueError("Manifest entry missing audio_dir and audio_root.")
        audio_dir = (audio_root if rel in (".", "") else audio_root / rel).as_posix()
    if not roi_dir:
        rel = entry.get("audio_dir_rel")
        if rel is None or roi_root is None:
            raise ValueError("Manifest entry missing roi_dir and roi_root.")
        roi_dir = (roi_root if rel in (".", "") else roi_root / rel).as_posix()
    return audio_dir, roi_dir


def _roi_has_data(filename: str) -> bool:
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return True
    except FileNotFoundError:
        return False
    return False


def _collect_files(
    entries: list[dict],
    audio_root: Optional[Path],
    roi_root: Optional[Path],
    max_files: Optional[int],
) -> tuple[list[str], list[str], int, int]:
    audio_files = []
    roi_files = []
    missing = 0
    empty = 0
    for entry in entries:
        audio_dir, roi_dir = _resolve_entry_paths(entry, audio_root, roi_root)
        wavs = _get_wavs_from_dir(audio_dir)
        for wav in wavs:
            roi_path = os.path.join(roi_dir, f"{Path(wav).stem}.txt")
            if not os.path.exists(roi_path):
                missing += 1
                continue
            if not _roi_has_data(roi_path):
                empty += 1
                continue
            audio_files.append(wav)
            roi_files.append(roi_path)
            if max_files is not None and len(audio_files) >= max_files:
                break
        if max_files is not None and len(audio_files) >= max_files:
            break
    return audio_files, roi_files, missing, empty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a short fixed-window validation training loop."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-spec-cache", action="store_true")
    parser.add_argument("--spec-cache-dir", type=Path, default=None)

    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    train_entries = manifest.get("train", [])
    test_entries = manifest.get("test", [])

    train_audio, train_rois, train_missing, train_empty = _collect_files(
        train_entries, args.audio_root, args.roi_root, args.max_files
    )
    test_audio, test_rois, test_missing, test_empty = _collect_files(
        test_entries, args.audio_root, args.roi_root, args.max_files
    )

    if not train_audio:
        raise ValueError("No training audio files found after filtering.")

    partition = {
        "train": {"audio": np.array(train_audio), "rois": np.array(train_rois)},
        "test": {},
    }
    if test_audio:
        partition["test"] = {
            "audio": np.array(test_audio),
            "rois": np.array(test_rois),
        }

    if train_missing or test_missing:
        print(
            f"Skipped {train_missing + test_missing} files with missing ROI files."
        )
    if train_empty or test_empty:
        print(
            f"Skipped {train_empty + test_empty} files with empty ROI files."
        )

    config = FixedWindowExperimentConfig.from_yaml(args.config)
    params = config.preprocess.to_params()
    data_config = config.data
    train_config = config.training

    loader_kwargs = data_config.to_loader_kwargs()
    if args.batch_size is not None:
        loader_kwargs["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_kwargs["num_workers"] = args.num_workers
    if args.disable_spec_cache:
        loader_kwargs["spec_cache_dir"] = None
    if args.spec_cache_dir is not None:
        loader_kwargs["spec_cache_dir"] = args.spec_cache_dir.as_posix()

    loaders = get_fixed_shotgun_data_loaders(
        partition,
        params,
        augmentations=config.augmentations,
        **loader_kwargs,
    )

    train_kwargs = train_config.to_train_kwargs()
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    trainer_kwargs = dict(train_kwargs.get("trainer_kwargs") or {})
    if args.cpu:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1, "precision": 32})
    train_kwargs["trainer_kwargs"] = trainer_kwargs

    args.save_dir.mkdir(parents=True, exist_ok=True)
    model, trainer = train_vae(
        loaders,
        save_dir=args.save_dir.as_posix(),
        **train_kwargs,
    )

    print(
        f"Training complete. Train files: {len(train_audio)} Test files: {len(test_audio)}"
    )
    print(f"Artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
