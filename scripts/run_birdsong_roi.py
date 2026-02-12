#!/usr/bin/env python3
"""Generate ROI files for birdsong audio directories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import repeat
from pathlib import Path
from typing import Iterable, Optional

import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.segment import segment


ALGORITHMS = {
    "amplitude": get_onsets_offsets,
    "amplitude_segmentation": get_onsets_offsets,
}

APPLEDOUBLE_PREFIX = "._"


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
        if any(
            name.lower().endswith(".wav") and not name.startswith(APPLEDOUBLE_PREFIX)
            for name in filenames
        ):
            audio_dirs.append(Path(dirpath).as_posix())
    return sorted(set(audio_dirs))


def _count_audio_wavs(audio_dir: str) -> int:
    try:
        filenames = os.listdir(audio_dir)
    except FileNotFoundError:
        return 0
    return len(
        [
            name
            for name in filenames
            if name.lower().endswith(".wav") and not name.startswith(APPLEDOUBLE_PREFIX)
        ]
    )


def _count_roi_txts(roi_dir: str) -> int:
    if not os.path.isdir(roi_dir):
        return 0
    return len(
        [
            name
            for name in os.listdir(roi_dir)
            if name.lower().endswith(".txt") and not name.startswith(APPLEDOUBLE_PREFIX)
        ]
    )


def _should_skip(audio_dir: str, roi_dir: str, force: bool, skip_existing: bool) -> bool:
    if force or not skip_existing:
        return False
    if not os.path.isdir(roi_dir):
        return False
    wav_count = _count_audio_wavs(audio_dir)
    if wav_count == 0:
        return True
    roi_count = _count_roi_txts(roi_dir)
    return roi_count >= wav_count


def _select_shard(tasks: list[tuple[str, str]], num_shards: int, shard_index: int) -> list[tuple[str, str]]:
    if num_shards <= 1:
        return tasks
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    return [task for idx, task in enumerate(tasks) if (idx % num_shards) == shard_index]


def _run_segment_task(
    audio_dir: str,
    roi_dir: str,
    params: dict,
    max_retries: int,
) -> dict:
    attempts = 0
    start = time.time()
    last_error = None
    while attempts <= max_retries:
        attempts += 1
        try:
            segment(audio_dir, roi_dir, params, verbose=True)
            return {
                "audio_dir": audio_dir,
                "roi_dir": roi_dir,
                "status": "ok",
                "attempts": attempts,
                "elapsed_sec": float(time.time() - start),
            }
        except Exception as exc:  # pragma: no cover - exercised via integration runs
            last_error = str(exc)
    return {
        "audio_dir": audio_dir,
        "roi_dir": roi_dir,
        "status": "error",
        "attempts": attempts,
        "elapsed_sec": float(time.time() - start),
        "error": last_error,
    }


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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")
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

    planned_tasks = list(zip(audio_dirs, roi_dirs))

    if args.max_dirs is not None:
        planned_tasks = planned_tasks[: args.max_dirs]

    planned_tasks = _select_shard(planned_tasks, args.num_shards, args.shard_index)

    tasks = []
    skipped_existing = 0
    for audio_dir, roi_dir in planned_tasks:
        if _should_skip(audio_dir, roi_dir, args.force, args.skip_existing):
            skipped_existing += 1
            continue
        tasks.append((audio_dir, roi_dir))

    if args.dry_run:
        print(
            "Planned ROI directories: "
            f"{len(tasks)} (skipped_existing={skipped_existing} shard={args.shard_index}/{args.num_shards})"
        )
        for audio_dir, roi_dir in tasks[:10]:
            print(f"  {audio_dir} -> {roi_dir}")
        return

    if not tasks:
        print("No ROI tasks to run. Check filters or --skip-existing.")
        return

    if args.jobs is None:
        cpu_count = os.cpu_count() or 1
        args.jobs = max(1, min(len(tasks), cpu_count - 1))

    start_time = time.time()
    total = len(tasks)
    print(
        "Running ROI generation for "
        f"{total} directories (skipped_existing={skipped_existing} shard={args.shard_index}/{args.num_shards})."
    )

    gen = zip([t[0] for t in tasks], [t[1] for t in tasks], repeat(params), repeat(int(args.max_retries)))
    results = Parallel(n_jobs=args.jobs)(
        delayed(_run_segment_task)(*args) for args in gen
    )

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    elapsed = max(1e-9, time.time() - start_time)
    rate = total / elapsed
    print(
        f"Done. total={total} ok={len(ok)} failed={len(failed)} ({rate:.2f} dirs/s, jobs={args.jobs})"
    )

    if args.summary_out is not None:
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_path": args.manifest.as_posix() if args.manifest else None,
            "segment_config": args.segment_config.as_posix(),
            "split": args.split,
            "audio_root": args.audio_root.as_posix() if args.audio_root else None,
            "roi_root": args.roi_root.as_posix() if args.roi_root else None,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "planned_total": int(len(planned_tasks)),
            "skipped_existing": int(skipped_existing),
            "total": int(total),
            "ok": int(len(ok)),
            "failed": int(len(failed)),
            "elapsed_sec": float(elapsed),
            "dirs_per_sec": float(rate),
            "errors": failed,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
