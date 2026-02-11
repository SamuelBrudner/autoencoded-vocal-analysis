#!/usr/bin/env python3
"""Evaluate latent invariance and self-retrieval for a trained VAE."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.latent_metrics import evaluate_latent_metrics, load_vae_from_checkpoint
from ava.models.shotgun_vae_dataset import get_fixed_shotgun_data_loaders, get_shotgun_partition
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


def _select_loaders(loaders: dict, split: str) -> list:
    if split == "train":
        return [loaders["train"]]
    if split == "test":
        if loaders.get("test") is None:
            raise ValueError("Requested test split but no test loader is available.")
        return [loaders["test"]]
    selected = [loaders["train"]]
    if loaders.get("test") is not None:
        selected.append(loaders["test"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate latent invariance and self-retrieval."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--audio-dir", action="append", default=None)
    parser.add_argument("--roi-dir", action="append", default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")

    if args.manifest:
        manifest = _load_manifest(args.manifest)
        train_entries = manifest.get("train", [])
        test_entries = manifest.get("test", [])

        train_audio, train_rois, train_missing, train_empty = _collect_files(
            train_entries, args.audio_root, args.roi_root, args.max_files
        )
        test_audio, test_rois, test_missing, test_empty = _collect_files(
            test_entries, args.audio_root, args.roi_root, args.max_files
        )

        if not train_audio and not test_audio:
            raise ValueError("No audio files found after filtering manifest entries.")

        partition = {"train": {"audio": np.array(train_audio), "rois": np.array(train_rois)}, "test": {}}
        if test_audio:
            partition["test"] = {"audio": np.array(test_audio), "rois": np.array(test_rois)}

        if train_missing or test_missing:
            print(f"Skipped {train_missing + test_missing} files with missing ROI files.")
        if train_empty or test_empty:
            print(f"Skipped {train_empty + test_empty} files with empty ROI files.")
    else:
        if not args.audio_dir or not args.roi_dir:
            raise ValueError("Provide --audio-dir and --roi-dir or a --manifest.")
        if len(args.audio_dir) != len(args.roi_dir):
            raise ValueError("--audio-dir and --roi-dir must be provided in pairs.")
        partition = get_shotgun_partition(
            args.audio_dir,
            args.roi_dir,
            split=args.train_fraction,
            shuffle=True,
        )

    config = FixedWindowExperimentConfig.from_yaml(args.config)
    params = config.preprocess.to_params()
    data_config = config.data
    loader_kwargs = data_config.to_loader_kwargs()
    if args.batch_size is not None:
        loader_kwargs["batch_size"] = args.batch_size
    if args.num_workers is not None:
        loader_kwargs["num_workers"] = args.num_workers

    loaders = get_fixed_shotgun_data_loaders(
        partition,
        params,
        augmentations=config.augmentations,
        augmentations_eval=True,
        return_pair=True,
        pair_with_original=True,
        **loader_kwargs,
    )

    eval_loaders = _select_loaders(loaders, args.split)
    model = load_vae_from_checkpoint(args.checkpoint.as_posix(), device=args.device)

    results = evaluate_latent_metrics(
        model,
        eval_loaders,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        device=model.device,
    )
    results.update(
        {
            "split": args.split,
            "checkpoint": args.checkpoint.as_posix(),
            "config": args.config.as_posix(),
            "max_samples": args.max_samples,
        }
    )

    payload = json.dumps(results, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
