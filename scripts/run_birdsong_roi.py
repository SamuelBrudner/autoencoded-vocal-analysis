#!/usr/bin/env python3
"""Generate ROI files for birdsong audio directories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import repeat
from pathlib import Path
from typing import Iterable, Optional

import yaml
from joblib import Parallel, delayed

from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.segment import segment
from ava.models.utils import _get_wavs_from_dir


ALGORITHMS = {
    "amplitude": get_onsets_offsets,
    "amplitude_segmentation": get_onsets_offsets,
}


def _load_segment_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    params = payload.get("segment", payload)
    if not isinstance(params, dict):
        raise ValueError("Segment config must be a mapping.")
    params = dict(params)
    algorithm = params.get("algorithm")
    if callable(algorithm):
        params["algorithm"] = algorithm
    else:
        algo_key = (algorithm or "amplitude")
        if isinstance(algo_key, str):
            algo_key = algo_key.lower()
        if algo_key not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available: {sorted(ALGORITHMS)}"
            )
        params["algorithm"] = ALGORITHMS[algo_key]
    return params


def _validate_segment_params(params: dict) -> None:
    required = {
        "fs",
        "min_freq",
        "max_freq",
        "nperseg",
        "noverlap",
        "spec_min_val",
        "spec_max_val",
        "th_1",
        "th_2",
        "th_3",
        "min_dur",
        "max_dur",
        "smoothing_timescale",
        "softmax",
        "temperature",
        "algorithm",
    }
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(f"Segment config missing keys: {missing}")


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


def _find_audio_dirs(root: Path) -> list[str]:
    audio_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(name.lower().endswith(".wav") for name in filenames):
            audio_dirs.append(Path(dirpath).as_posix())
    return sorted(set(audio_dirs))


def _should_skip(audio_dir: str, roi_dir: str, force: bool, skip_existing: bool) -> bool:
    if force or not skip_existing:
        return False
    if not os.path.isdir(roi_dir):
        return False
    wav_count = len(_get_wavs_from_dir(audio_dir))
    if wav_count == 0:
        return True
    roi_count = len([f for f in os.listdir(roi_dir) if f.endswith(".txt")])
    return roi_count >= wav_count


def _run_segment(audio_dir: str, roi_dir: str, params: dict) -> None:
    segment(audio_dir, roi_dir, params, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ROI generation over birdsong audio directories."
    )
    parser.add_argument("--segment-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument("--max-dirs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    params = _load_segment_params(args.segment_config)
    _validate_segment_params(params)

    if args.manifest:
        manifest = _load_manifest(args.manifest)
        entries = []
        if args.split in ("train", "all"):
            entries.extend(manifest.get("train", []))
        if args.split in ("test", "all"):
            entries.extend(manifest.get("test", []))
        audio_dirs = []
        roi_dirs = []
        for entry in entries:
            audio_dir, roi_dir = _resolve_entry_paths(
                entry,
                audio_root=args.audio_root,
                roi_root=args.roi_root,
            )
            audio_dirs.append(audio_dir)
            roi_dirs.append(roi_dir)
    else:
        if args.audio_root is None or args.roi_root is None:
            raise ValueError("Provide --audio-root and --roi-root without a manifest.")
        audio_dirs = _find_audio_dirs(args.audio_root)
        roi_dirs = [
            (args.roi_root if rel == "." else args.roi_root / rel).as_posix()
            for rel in [
                Path(audio_dir).relative_to(args.audio_root).as_posix()
                for audio_dir in audio_dirs
            ]
        ]

    if args.max_dirs is not None:
        audio_dirs = audio_dirs[: args.max_dirs]
        roi_dirs = roi_dirs[: args.max_dirs]

    tasks = []
    for audio_dir, roi_dir in zip(audio_dirs, roi_dirs):
        if _should_skip(audio_dir, roi_dir, args.force, args.skip_existing):
            continue
        tasks.append((audio_dir, roi_dir))

    if args.dry_run:
        print(f"Planned ROI directories: {len(tasks)}")
        for audio_dir, roi_dir in tasks[:10]:
            print(f"  {audio_dir} -> {roi_dir}")
        return

    if not tasks:
        print("No ROI tasks to run. Check filters or --skip-existing.")
        return

    if args.jobs is None:
        cpu_count = os.cpu_count() or 1
        args.jobs = max(1, min(len(tasks), cpu_count - 1))

    print(f"Running ROI generation for {len(tasks)} directories.")
    gen = zip([t[0] for t in tasks], [t[1] for t in tasks], repeat(params))
    Parallel(n_jobs=args.jobs)(delayed(_run_segment)(*args) for args in gen)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
