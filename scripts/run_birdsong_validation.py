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

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.lightning_vae import train_vae
from ava.models.roi_preflight import (
    assert_window_length_compatible,
    assert_window_length_compatible_parquet_sample,
)
from ava.models.shotgun_vae_dataset import (
    get_fixed_shotgun_data_loaders,
    get_manifest_fixed_window_data_loaders,
)
from ava.models.utils import _get_wavs_from_dir
from ava.data.manifest_paths import resolve_manifest_entry_paths


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
        audio_dir, roi_dir = resolve_manifest_entry_paths(
            entry, audio_root=audio_root, roi_root=roi_root
        )
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


def _resolve_entries(
    entries: list[dict],
    audio_root: Optional[Path],
    roi_root: Optional[Path],
) -> list[dict]:
    resolved = []
    for entry in entries:
        audio_dir, roi_dir = resolve_manifest_entry_paths(
            entry, audio_root=audio_root, roi_root=roi_root
        )
        payload = dict(entry)
        payload["audio_dir"] = audio_dir
        payload["roi_dir"] = roi_dir
        resolved.append(payload)
    return resolved


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
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use manifest streaming dataset (does not enumerate all wav/ROI files).",
    )
    parser.add_argument(
        "--roi-format",
        choices=["txt", "parquet"],
        default="txt",
        help="ROI storage format under each roi_dir (streaming mode).",
    )
    parser.add_argument(
        "--roi-parquet-name",
        type=str,
        default="roi.parquet",
        help="Filename for per-directory ROI parquet bundle (streaming mode).",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        default=256,
        help="Arbitrary dataset length controlling batches/epoch for window sampling.",
    )
    parser.add_argument(
        "--roi-cache-size",
        type=int,
        default=16,
        help="Number of directories whose ROI bundles are cached per worker (streaming mode).",
    )
    parser.add_argument(
        "--preflight-sample-dirs",
        type=int,
        default=10,
        help="Number of ROI parquet bundles to sample for duration preflight (streaming parquet).",
    )
    parser.add_argument(
        "--preflight-sample-segments",
        type=int,
        default=2000,
        help="Number of ROI segments to sample for duration preflight (streaming parquet).",
    )
    parser.add_argument(
        "--preflight-seed",
        type=int,
        default=0,
        help="RNG seed for the sampled duration preflight (streaming parquet).",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable-spec-cache", action="store_true")
    parser.add_argument("--spec-cache-dir", type=Path, default=None)

    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    train_entries = manifest.get("train", [])
    test_entries = manifest.get("test", [])

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

    use_pairs = train_config.invariance_weight > 0
    if args.streaming:
        resolved_train = _resolve_entries(train_entries, args.audio_root, args.roi_root)
        resolved_test = _resolve_entries(test_entries, args.audio_root, args.roi_root)
        if not resolved_train:
            raise ValueError("No manifest entries available for validation training.")

        if str(args.roi_format) == "parquet":
            parquet_paths = [
                os.path.join(entry["roi_dir"], str(args.roi_parquet_name))
                for entry in (resolved_train + resolved_test)
            ]
            preflight_stats = assert_window_length_compatible_parquet_sample(
                parquet_paths,
                window_length=float(params["window_length"]),
                sample_dirs=int(args.preflight_sample_dirs),
                sample_segments=int(args.preflight_sample_segments),
                seed=int(args.preflight_seed),
            )
            print(
                "ROI duration preflight (parquet sample): "
                f"{json.dumps(preflight_stats, sort_keys=True)}"
            )
        else:
            print("Note: streaming mode skips ROI duration preflight for txt ROIs.")

        loaders = get_manifest_fixed_window_data_loaders(
            resolved_train,
            resolved_test if resolved_test else None,
            params,
            roi_format=str(args.roi_format),
            roi_parquet_name=str(args.roi_parquet_name),
            dataset_length=int(args.dataset_length),
            roi_cache_size=int(args.roi_cache_size),
            augmentations=config.augmentations,
            return_pair=use_pairs,
            pair_with_original=use_pairs,
            **loader_kwargs,
        )
        train_audio = []
        test_audio = []
    else:
        train_audio, train_rois, train_missing, train_empty = _collect_files(
            train_entries, args.audio_root, args.roi_root, args.max_files
        )
        test_audio, test_rois, test_missing, test_empty = _collect_files(
            test_entries, args.audio_root, args.roi_root, args.max_files
        )

        if not train_audio:
            raise ValueError("No training audio files found after filtering.")

        preflight_stats = assert_window_length_compatible(
            train_rois + test_rois,
            window_length=params["window_length"],
        )
        print(f"ROI duration preflight: {json.dumps(preflight_stats, sort_keys=True)}")

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

        loaders = get_fixed_shotgun_data_loaders(
            partition,
            params,
            augmentations=config.augmentations,
            return_pair=use_pairs,
            pair_with_original=use_pairs,
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
        config_path=args.config.as_posix(),
        manifest_path=args.manifest.as_posix(),
        **train_kwargs,
    )

    if args.streaming:
        print(
            f"Training complete. Train dirs: {len(train_entries)} Test dirs: {len(test_entries)}"
        )
    else:
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
