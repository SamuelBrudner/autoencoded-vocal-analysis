#!/usr/bin/env python3
"""Summarize ROI coverage across a birdsong manifest.

This script inspects ROI outputs for each leaf audio directory referenced by a
birdsong manifest (schema: docs/birdsong_manifest.md), producing:
- per-directory coverage stats (missing/empty ROI files, segment counts, duration stats)
- aggregated coverage stats grouped by bird/regime/dph/session

Outputs are written to an output directory as CSV + (best-effort) Parquet, plus
a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.data.manifest_paths import resolve_manifest_entry_paths

APPLEDOUBLE_PREFIX = "._"
DEFAULT_ROI_PARQUET_NAME = "roi.parquet"


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_entries(manifest: dict, split: str) -> list[dict]:
    if split == "train":
        return list(manifest.get("train", []))
    if split == "test":
        return list(manifest.get("test", []))
    entries: list[dict] = []
    entries.extend(manifest.get("train", []))
    entries.extend(manifest.get("test", []))
    return entries


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


def _parse_roi_file(path: str) -> Tuple[int, float, List[float], int]:
    """Return (segments, total_duration_sec, durations_sec, parse_errors)."""
    durations: list[float] = []
    parse_errors = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    parse_errors += 1
                    continue
                try:
                    onset = float(parts[0])
                    offset = float(parts[1])
                except ValueError:
                    parse_errors += 1
                    continue
                dur = float(offset - onset)
                if not np.isfinite(dur) or dur <= 0:
                    parse_errors += 1
                    continue
                durations.append(dur)
    except FileNotFoundError:
        return 0, 0.0, [], 0

    total = float(np.sum(durations)) if durations else 0.0
    return len(durations), total, durations, parse_errors


def _load_roi_parquet(path: str) -> dict[str, tuple[list[float], list[float]]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Parquet ROI coverage requires pyarrow. Install pyarrow or use --roi-format txt."
        ) from exc

    table = pq.read_table(path, columns=["clip_stem", "onsets_sec", "offsets_sec"])
    payload = table.to_pydict()
    stems = payload.get("clip_stem") or []
    onsets = payload.get("onsets_sec") or []
    offsets = payload.get("offsets_sec") or []
    if not (len(stems) == len(onsets) == len(offsets)):
        raise ValueError("Malformed ROI parquet payload (column lengths differ).")
    out: dict[str, tuple[list[float], list[float]]] = {}
    for stem, ons, offs in zip(stems, onsets, offsets):
        out[str(stem)] = ([float(x) for x in (ons or [])], [float(x) for x in (offs or [])])
    return out


def _select_shard(entries: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards <= 1:
        return entries
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    return [entry for idx, entry in enumerate(entries) if (idx % num_shards) == shard_index]


def _minimal_entry_metadata(entry: dict) -> Dict[str, Any]:
    keys = [
        "audio_dir_rel",
        "bird_id_norm",
        "bird_id_raw",
        "regime",
        "dph",
        "session_label",
        "split",
        "num_files",
        "top_dir",
        "pre_bird_path",
    ]
    return {key: entry.get(key) for key in keys if key in entry}


def coverage_for_entry(
    entry: dict,
    audio_root: Optional[Path],
    roi_root: Optional[Path],
    roi_format: str,
    roi_parquet_name: str,
    max_files_per_dir: Optional[int],
) -> dict:
    audio_dir, roi_dir = resolve_manifest_entry_paths(
        entry, audio_root=audio_root, roi_root=roi_root
    )

    wavs = _list_wavs(audio_dir)
    if max_files_per_dir is not None:
        wavs = wavs[: int(max_files_per_dir)]

    out: dict[str, Any] = {
        **_minimal_entry_metadata(entry),
        "audio_dir": audio_dir,
        "roi_dir": roi_dir,
        "wav_files_checked": int(len(wavs)),
        "missing_roi_dir": False,
        "missing_roi_files": 0,
        "empty_roi_files": 0,
        "roi_files_present": 0,
        "roi_parse_errors": 0,
        "segments_total": 0,
        "segment_duration_total_sec": 0.0,
        "segment_duration_mean_sec": None,
        "segment_duration_min_sec": None,
        "segment_duration_p05_sec": None,
        "segment_duration_p50_sec": None,
        "segment_duration_p95_sec": None,
        "segment_duration_max_sec": None,
    }

    roi_index = None
    roi_parquet_path = None
    if roi_format == "parquet":
        roi_parquet_path = os.path.join(roi_dir, roi_parquet_name)
        if os.path.exists(roi_parquet_path):
            roi_index = _load_roi_parquet(roi_parquet_path)

    durations_all: list[float] = []
    missing = 0
    empty = 0
    present = 0
    parse_errors = 0
    segments_total = 0
    duration_total = 0.0

    if roi_format == "txt" and not os.path.isdir(roi_dir):
        out["missing_roi_dir"] = True
        out["missing_roi_files"] = int(len(wavs))
        return out

    if roi_format == "parquet" and roi_index is None:
        out["missing_roi_dir"] = True
        out["missing_roi_files"] = int(len(wavs))
        return out

    for wav in wavs:
        if roi_format == "txt":
            roi_path = os.path.join(roi_dir, f"{Path(wav).stem}.txt")
            if not os.path.exists(roi_path):
                missing += 1
                continue
            present += 1
            segs, dur_sum, durations, errs = _parse_roi_file(roi_path)
            parse_errors += int(errs)
            if segs == 0:
                empty += 1
                continue
            segments_total += int(segs)
            duration_total += float(dur_sum)
            durations_all.extend(durations)
            continue

        stem = Path(wav).stem
        record = roi_index.get(stem) if roi_index is not None else None
        if record is None:
            missing += 1
            continue
        present += 1
        onsets, offsets = record
        if len(onsets) != len(offsets):
            parse_errors += 1
            segs = min(len(onsets), len(offsets))
            onsets = onsets[:segs]
            offsets = offsets[:segs]
        segs = len(onsets)
        if segs == 0:
            empty += 1
            continue
        durations = []
        dur_sum = 0.0
        for onset, offset in zip(onsets, offsets):
            dur = float(offset - onset)
            if not np.isfinite(dur) or dur <= 0:
                parse_errors += 1
                continue
            durations.append(dur)
            dur_sum += dur
        if not durations:
            empty += 1
            continue
        segments_total += int(len(durations))
        duration_total += float(dur_sum)
        durations_all.extend(durations)

    out["missing_roi_files"] = int(missing)
    out["empty_roi_files"] = int(empty)
    out["roi_files_present"] = int(present)
    out["roi_parse_errors"] = int(parse_errors)
    out["segments_total"] = int(segments_total)
    out["segment_duration_total_sec"] = float(duration_total)

    if durations_all:
        arr = np.asarray(durations_all, dtype=float)
        out["segment_duration_mean_sec"] = float(duration_total / max(1, segments_total))
        out["segment_duration_min_sec"] = float(np.min(arr))
        out["segment_duration_p05_sec"] = float(np.quantile(arr, 0.05))
        out["segment_duration_p50_sec"] = float(np.quantile(arr, 0.50))
        out["segment_duration_p95_sec"] = float(np.quantile(arr, 0.95))
        out["segment_duration_max_sec"] = float(np.max(arr))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize ROI coverage across a birdsong manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument(
        "--audio-dir-rel-prefix",
        type=str,
        default=None,
        help="Only include entries whose audio_dir_rel starts with this prefix.",
    )
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--roi-root", type=Path, default=None)
    parser.add_argument(
        "--roi-format",
        choices=["txt", "parquet"],
        default="txt",
        help="ROI storage format under each roi_dir.",
    )
    parser.add_argument(
        "--roi-parquet-name",
        type=str,
        default=DEFAULT_ROI_PARQUET_NAME,
        help="Filename for per-directory ROI parquet bundle (when --roi-format=parquet).",
    )
    parser.add_argument("--max-dirs", type=int, default=None)
    parser.add_argument("--max-files-per-dir", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.max_dirs is not None and args.max_dirs <= 0:
        raise ValueError("--max-dirs must be positive.")
    if args.max_files_per_dir is not None and args.max_files_per_dir <= 0:
        raise ValueError("--max-files-per-dir must be positive.")

    manifest = _load_manifest(args.manifest)
    entries = _select_entries(manifest, args.split)
    if not entries:
        raise ValueError("No manifest entries found for the requested split.")

    # Stable ordering for deterministic sharding.
    entries = sorted(
        entries,
        key=lambda e: (str(e.get("audio_dir_rel", "")), str(e.get("audio_dir", ""))),
    )

    if args.audio_dir_rel_prefix:
        prefix = str(args.audio_dir_rel_prefix)
        entries = [
            entry
            for entry in entries
            if str(entry.get("audio_dir_rel") or "").startswith(prefix)
        ]
        if not entries:
            raise ValueError("No manifest entries match --audio-dir-rel-prefix.")

    if args.max_dirs is not None:
        entries = entries[: int(args.max_dirs)]

    entries = _select_shard(entries, args.num_shards, args.shard_index)

    if args.dry_run:
        print(
            f"Planned directories: {len(entries)} (split={args.split} shard={args.shard_index}/{args.num_shards})"
        )
        for entry in entries[:10]:
            audio_dir_rel = entry.get("audio_dir_rel")
            _, roi_dir = resolve_manifest_entry_paths(
                entry, audio_root=args.audio_root, roi_root=args.roi_root
            )
            print(f"  {audio_dir_rel} -> {roi_dir}")
        return

    if args.jobs is None:
        cpu_count = os.cpu_count() or 1
        args.jobs = max(1, min(len(entries), cpu_count - 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    results = Parallel(n_jobs=args.jobs)(
        delayed(coverage_for_entry)(
            entry,
            audio_root=args.audio_root,
            roi_root=args.roi_root,
            roi_format=str(args.roi_format),
            roi_parquet_name=str(args.roi_parquet_name),
            max_files_per_dir=args.max_files_per_dir,
        )
        for entry in entries
    )
    elapsed = max(1e-9, time.time() - start)

    missing_dirs = sum(1 for r in results if r.get("missing_roi_dir"))
    missing_files = sum(int(r.get("missing_roi_files") or 0) for r in results)
    empty_files = sum(int(r.get("empty_roi_files") or 0) for r in results)
    parse_errors = sum(int(r.get("roi_parse_errors") or 0) for r in results)
    segments_total = sum(int(r.get("segments_total") or 0) for r in results)
    duration_total = float(
        np.sum([float(r.get("segment_duration_total_sec") or 0.0) for r in results])
    )

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": args.manifest.as_posix(),
        "split": args.split,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "directories": int(len(results)),
        "missing_roi_dirs": int(missing_dirs),
        "missing_roi_files": int(missing_files),
        "empty_roi_files": int(empty_files),
        "roi_parse_errors": int(parse_errors),
        "segments_total": int(segments_total),
        "segment_duration_total_sec": float(duration_total),
        "elapsed_sec": float(elapsed),
        "dirs_per_sec": float(len(results) / elapsed),
    }

    # Always write JSON outputs (no optional deps required).
    (args.out_dir / "per_directory.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Best-effort tabular outputs for convenient analysis.
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(results)
        df.to_csv(args.out_dir / "per_directory.csv", index=False)
        try:
            df.to_parquet(args.out_dir / "per_directory.parquet", index=False)
        except Exception:
            pass

        group_keys = ["bird_id_norm", "regime", "dph", "session_label", "split"]
        group_keys = [k for k in group_keys if k in df.columns]
        agg = (
            df.groupby(group_keys, dropna=False)
            .agg(
                directories=("audio_dir_rel", "count"),
                wav_files_checked=("wav_files_checked", "sum"),
                roi_files_present=("roi_files_present", "sum"),
                missing_roi_files=("missing_roi_files", "sum"),
                empty_roi_files=("empty_roi_files", "sum"),
                segments_total=("segments_total", "sum"),
                segment_duration_total_sec=("segment_duration_total_sec", "sum"),
                roi_parse_errors=("roi_parse_errors", "sum"),
            )
            .reset_index()
        )
        if "segments_total" in agg.columns and "segment_duration_total_sec" in agg.columns:
            agg["segment_duration_mean_sec_weighted"] = (
                agg["segment_duration_total_sec"]
                / agg["segments_total"].replace(0, np.nan)
            )

        agg.to_csv(args.out_dir / "by_bird.csv", index=False)
        try:
            agg.to_parquet(args.out_dir / "by_bird.parquet", index=False)
        except Exception:
            pass
    except ImportError:
        pass

    print(json.dumps(summary, indent=2))

    # Non-zero exit when missing ROI outputs (and optionally empties) exist.
    if summary["missing_roi_dirs"] or summary["missing_roi_files"] or summary["roi_parse_errors"]:
        sys.exit(1)
    if args.fail_on_empty and summary["empty_roi_files"]:
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
