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
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import warnings

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.segmenting.amplitude_segmentation import get_onsets_offsets
from ava.segmenting.segment import segment
from ava.data.manifest_paths import resolve_manifest_entry_paths


ALGORITHMS = {
    "amplitude": get_onsets_offsets,
    "amplitude_segmentation": get_onsets_offsets,
}

APPLEDOUBLE_PREFIX = "._"
DEFAULT_ROI_PARQUET_NAME = "roi.parquet"


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


def _should_skip_txt(audio_dir: str, roi_dir: str, force: bool, skip_existing: bool) -> bool:
    if force or not skip_existing:
        return False
    if not os.path.isdir(roi_dir):
        return False
    wav_count = _count_audio_wavs(audio_dir)
    if wav_count == 0:
        return True
    roi_count = _count_roi_txts(roi_dir)
    return roi_count >= wav_count


def _should_skip_parquet(
    roi_dir: str, parquet_name: str, force: bool, skip_existing: bool
) -> bool:
    if force or not skip_existing:
        return False
    path = Path(roi_dir) / parquet_name
    return path.exists()


def _select_shard(tasks: list[tuple[str, str]], num_shards: int, shard_index: int) -> list[tuple[str, str]]:
    if num_shards <= 1:
        return tasks
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    return [task for idx, task in enumerate(tasks) if (idx % num_shards) == shard_index]


def _list_wavs(audio_dir: str) -> list[str]:
    try:
        filenames = os.listdir(audio_dir)
    except FileNotFoundError:
        return []
    return [
        os.path.join(audio_dir, name)
        for name in sorted(filenames)
        if name.lower().endswith(".wav") and not name.startswith(APPLEDOUBLE_PREFIX)
    ]


def _segment_directory_to_parquet(
    audio_dir: str,
    roi_dir: str,
    params: dict,
    parquet_name: str,
) -> tuple[int, int]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet output requires pyarrow. Install pyarrow or use --roi-output-format txt."
        ) from exc

    output_dir = Path(roi_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / parquet_name
    tmp_path = output_dir / f".{parquet_name}.tmp"

    schema = pa.schema(
        [
            pa.field("clip_stem", pa.string()),
            pa.field("onsets_sec", pa.list_(pa.float32())),
            pa.field("offsets_sec", pa.list_(pa.float32())),
        ]
    )

    wavs = _list_wavs(audio_dir)
    segments_total = 0
    clips_total = 0

    try:
        with pq.ParquetWriter(str(tmp_path), schema=schema) as writer:
            batch_clip_stems: list[str] = []
            batch_onsets: list[list[float]] = []
            batch_offsets: list[list[float]] = []

            def flush_batch() -> None:
                if not batch_clip_stems:
                    return
                table = pa.Table.from_arrays(
                    [
                        pa.array(batch_clip_stems, type=pa.string()),
                        pa.array(batch_onsets, type=pa.list_(pa.float32())),
                        pa.array(batch_offsets, type=pa.list_(pa.float32())),
                    ],
                    names=["clip_stem", "onsets_sec", "offsets_sec"],
                )
                writer.write_table(table)
                batch_clip_stems.clear()
                batch_onsets.clear()
                batch_offsets.clear()

            for wav_path in wavs:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=WavFileWarning)
                    fs, audio = wavfile.read(wav_path)
                expected_fs = params.get("fs")
                local_params = params
                if expected_fs is not None and fs != expected_fs:
                    # Keep parquet mode robust across mixed sample rates while
                    # warning loudly. The legacy txt pipeline does not enforce
                    # sample rate consistency either.
                    print(
                        f"Warning: sample rate mismatch for {wav_path}: found {fs}, expected {expected_fs}. Using {fs}.",
                        file=sys.stderr,
                    )
                    local_params = dict(params)
                    local_params["fs"] = fs

                onsets, offsets = local_params["algorithm"](audio, local_params)
                onsets = [float(x) for x in onsets]
                offsets = [float(x) for x in offsets]
                if len(onsets) != len(offsets):
                    raise ValueError(
                        f"Algorithm returned mismatched onsets/offsets lengths for {wav_path}."
                    )

                batch_clip_stems.append(Path(wav_path).stem)
                batch_onsets.append(onsets)
                batch_offsets.append(offsets)

                clips_total += 1
                segments_total += len(onsets)

                if len(batch_clip_stems) >= 512:
                    flush_batch()

            flush_batch()
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return clips_total, segments_total


def _run_segment_task(
    audio_dir: str,
    roi_dir: str,
    params: dict,
    roi_output_format: str,
    roi_parquet_name: str,
    max_retries: int,
) -> dict:
    attempts = 0
    start = time.time()
    last_error = None
    clips_total = None
    segments_total = None
    while attempts <= max_retries:
        attempts += 1
        try:
            if roi_output_format == "txt":
                segment(audio_dir, roi_dir, params, verbose=True)
            else:
                clips_total, segments_total = _segment_directory_to_parquet(
                    audio_dir,
                    roi_dir,
                    params,
                    parquet_name=roi_parquet_name,
                )
            return {
                "audio_dir": audio_dir,
                "roi_dir": roi_dir,
                "status": "ok",
                "attempts": attempts,
                "elapsed_sec": float(time.time() - start),
                "clips_total": clips_total,
                "segments_total": segments_total,
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
        "clips_total": clips_total,
        "segments_total": segments_total,
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
    parser.add_argument(
        "--roi-output-format",
        choices=["txt", "parquet"],
        default="txt",
        help="ROI output format. 'txt' writes one <clip>.txt per wav; 'parquet' writes one parquet bundle per audio directory.",
    )
    parser.add_argument(
        "--roi-parquet-name",
        type=str,
        default=DEFAULT_ROI_PARQUET_NAME,
        help="Filename for per-directory parquet bundles (when --roi-output-format=parquet).",
    )
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
            audio_dir, roi_dir = resolve_manifest_entry_paths(
                entry, audio_root=args.audio_root, roi_root=args.roi_root
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
        if args.roi_output_format == "txt":
            should_skip = _should_skip_txt(
                audio_dir, roi_dir, args.force, args.skip_existing
            )
        else:
            should_skip = _should_skip_parquet(
                roi_dir,
                parquet_name=str(args.roi_parquet_name),
                force=args.force,
                skip_existing=args.skip_existing,
            )
        if should_skip:
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

    gen = zip(
        [t[0] for t in tasks],
        [t[1] for t in tasks],
        repeat(params),
        repeat(str(args.roi_output_format)),
        repeat(str(args.roi_parquet_name)),
        repeat(int(args.max_retries)),
    )
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
