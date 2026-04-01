#!/usr/bin/env python3
"""Launcher for manifest-driven recurrent sequence-VAE birdsong training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.data.manifest_paths import resolve_manifest_entry_paths
from ava.models.fixed_window_config import FixedWindowExperimentConfig
from ava.models.sequence_window_dataset import get_sequence_window_data_loaders


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
    audio_files: list[str] = []
    roi_files: list[str] = []
    missing = 0
    empty = 0
    for entry in entries:
        audio_dir, roi_dir = resolve_manifest_entry_paths(
            entry,
            audio_root=audio_root,
            roi_root=roi_root,
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


def _parse_trainer_overrides(payload: Optional[str]) -> dict:
    if not payload:
        return {}
    overrides = json.loads(payload)
    if not isinstance(overrides, dict):
        raise ValueError("--trainer-kwargs-json must parse to a JSON object.")
    return overrides


def _get_wavs_from_dir(path: Path) -> list[str]:
    return [
        (path / filename).as_posix()
        for filename in sorted(os.listdir(path))
        if filename.endswith(".wav")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch contiguous whole-file sequence-VAE birdsong training."
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
        "--trainer-kwargs-json",
        type=str,
        default=None,
        help="JSON object merged into Lightning Trainer kwargs.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from ava.models.lightning_sequence_vae import train_sequence_vae


    manifest = _load_manifest(args.manifest)
    train_entries = manifest.get("train", [])
    test_entries = manifest.get("test", [])
    config = FixedWindowExperimentConfig.from_yaml(args.config)
    params = config.preprocess.to_params()
    data_config = config.data
    train_config = config.training

    train_audio, train_rois, train_missing, train_empty = _collect_files(
        train_entries,
        args.audio_root,
        args.roi_root,
        args.max_files,
    )
    test_audio, test_rois, test_missing, test_empty = _collect_files(
        test_entries,
        args.audio_root,
        args.roi_root,
        args.max_files,
    )
    if not train_audio:
        raise ValueError("No training audio files found after filtering.")

    if train_missing or test_missing:
        print(f"Skipped {train_missing + test_missing} files with missing ROI files.")
    if train_empty or test_empty:
        print(f"Skipped {train_empty + test_empty} files with empty ROI files.")

    if args.dry_run:
        print(f"Planned train files: {len(train_audio)}")
        print(f"Planned test files: {len(test_audio)}")
        return

    partition = {
        "train": {"audio": train_audio, "rois": train_rois},
        "test": {"audio": test_audio, "rois": test_rois},
    }
    batch_size = args.batch_size if args.batch_size is not None else data_config.batch_size
    num_workers = args.num_workers if args.num_workers is not None else data_config.num_workers
    loaders = get_sequence_window_data_loaders(
        partition,
        params,
        batch_size=batch_size,
        shuffle=(data_config.shuffle_train, data_config.shuffle_test),
        num_workers=num_workers,
        audio_cache_size=data_config.audio_cache_size,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers,
        prefetch_factor=data_config.prefetch_factor,
    )

    train_kwargs = train_config.to_sequence_train_kwargs()
    sequence_hop_length_sec = params.get("sequence_hop_length")
    if sequence_hop_length_sec is None:
        sequence_hop_length_sec = float(params["window_length"]) / 2.0
    train_kwargs.setdefault("sequence_hop_length_sec", float(sequence_hop_length_sec))
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    trainer_kwargs = dict(train_kwargs.get("trainer_kwargs") or {})
    trainer_kwargs.update(_parse_trainer_overrides(args.trainer_kwargs_json))
    if args.cpu:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1, "precision": 32})
    train_kwargs["trainer_kwargs"] = trainer_kwargs

    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_sequence_vae(
        loaders,
        save_dir=args.save_dir.as_posix(),
        input_shape=(int(params["num_freq_bins"]), int(params["num_time_bins"])),
        config_path=args.config.as_posix(),
        manifest_path=args.manifest.as_posix(),
        **train_kwargs,
    )

    print(
        f"Training complete. Train files: {len(train_audio)} Test files: {len(test_audio)}"
    )
    print(f"Artifacts saved to: {args.save_dir}")
    print(f"Dashboard HTML: {args.save_dir / 'training_dashboard.html'}")
    print(f"Dashboard JSON: {args.save_dir / 'training_dashboard.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
