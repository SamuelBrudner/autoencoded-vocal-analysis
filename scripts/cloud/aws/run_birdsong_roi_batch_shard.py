#!/usr/bin/env python3
"""AWS Batch shard runner for birdsong ROI extraction (S3 -> local -> S3).

This script is intended to run inside an AWS Batch *array job* where
`AWS_BATCH_JOB_ARRAY_INDEX` maps to `--shard-index`, and `--num-shards` matches
the array size.

Workflow (per shard):
1. Download manifest + segment config from S3.
2. Deterministically select this shard's manifest entries.
3. Optionally skip entries whose ROI parquet already exists in S3.
4. Download each entry's audio directory from S3.
5. Run `scripts/run_birdsong_roi.py` over the subset.
6. Upload per-directory ROI parquet bundles back to S3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.cloud.manifest_sharding import (
    apply_max_dirs,
    iter_manifest_entry_pairs,
    pairs_to_manifest,
    select_shard,
)


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_int(name: str, default: int | None = None) -> int | None:
    val = _env(name)
    if val is None:
        return default
    return int(val)


def _join_s3_uri(root_uri: str, rel: str) -> str:
    root_uri = str(root_uri).rstrip("/")
    rel = str(rel or ".").lstrip("/")
    if rel in (".", ""):
        return root_uri
    return f"{root_uri}/{rel}"


def _require_aws_cli() -> str:
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError(
            "aws CLI is required but was not found on PATH. "
            "Install awscli (or AWS CLI v2) and configure credentials."
        )
    return aws


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}")
    return proc


def _s3_object_exists(aws: str, uri: str) -> bool:
    # `aws s3 ls s3://bucket/key` is a convenient existence check for a single object.
    proc = subprocess.run([aws, "s3", "ls", uri], capture_output=True, text=True)
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _download_one_dir(
    aws: str,
    s3_src: str,
    local_dst: Path,
    include_patterns: list[str],
    quiet: bool,
) -> dict:
    start = time.time()
    local_dst.mkdir(parents=True, exist_ok=True)
    cmd = [aws, "s3", "sync", str(s3_src), str(local_dst), "--exclude", "*"]
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])
    if quiet:
        cmd.append("--only-show-errors")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return {
        "s3_src": str(s3_src),
        "local_dst": local_dst.as_posix(),
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - start),
        "output": output.strip() if proc.returncode != 0 else None,
        "cmd": cmd if proc.returncode != 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one shard of ROI extraction inside an AWS Batch array job."
    )
    parser.add_argument(
        "--manifest-s3-uri",
        type=str,
        default=_env("AVA_MANIFEST_S3_URI"),
        required=False,
        help="S3 URI to the manifest JSON, e.g. s3://bucket/path/birdsong_manifest.json",
    )
    parser.add_argument(
        "--segment-config-s3-uri",
        type=str,
        default=_env("AVA_SEGMENT_CONFIG_S3_URI"),
        required=False,
        help="S3 URI to the segment config YAML, e.g. s3://bucket/path/segment.yaml",
    )
    parser.add_argument(
        "--s3-audio-root",
        type=str,
        default=_env("AVA_S3_AUDIO_ROOT"),
        required=False,
        help="S3 URI root for audio directories, e.g. s3://bucket/birdsong/audio",
    )
    parser.add_argument(
        "--s3-roi-root",
        type=str,
        default=_env("AVA_S3_ROI_ROOT"),
        required=False,
        help="S3 URI root for ROI output directories, e.g. s3://bucket/birdsong/roi",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default=_env("AVA_SPLIT", "all"))
    parser.add_argument("--num-shards", type=int, default=_env_int("AVA_NUM_SHARDS"))
    parser.add_argument(
        "--shard-index",
        type=int,
        default=_env_int("AWS_BATCH_JOB_ARRAY_INDEX", 0),
        help="Shard index. Defaults to AWS_BATCH_JOB_ARRAY_INDEX (or 0).",
    )
    parser.add_argument("--max-dirs", type=int, default=None)
    parser.add_argument(
        "--roi-output-format",
        choices=["parquet"],
        default=_env("AVA_ROI_OUTPUT_FORMAT", "parquet"),
    )
    parser.add_argument("--roi-parquet-name", type=str, default=_env("AVA_ROI_PARQUET_NAME", "roi.parquet"))
    parser.add_argument("--download-jobs", type=int, default=_env_int("AVA_DOWNLOAD_JOBS", 8))
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--skip-existing", action="store_true", default=bool(int(_env("AVA_SKIP_EXISTING", "1"))))
    parser.add_argument("--jobs", type=int, default=_env_int("AVA_JOBS"))
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(_env("AVA_WORKDIR", "/tmp/ava_roi_workdir")),
        help="Local scratch directory for downloads and outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--s3-summary-root",
        type=str,
        default=_env("AVA_S3_SUMMARY_ROOT"),
        help="Optional S3 URI root to upload per-shard summaries, e.g. s3://bucket/birdsong/roi/_summaries",
    )

    args = parser.parse_args()

    if not args.manifest_s3_uri:
        raise ValueError("--manifest-s3-uri (or AVA_MANIFEST_S3_URI) is required.")
    if not args.segment_config_s3_uri:
        raise ValueError("--segment-config-s3-uri (or AVA_SEGMENT_CONFIG_S3_URI) is required.")
    if not args.s3_audio_root:
        raise ValueError("--s3-audio-root (or AVA_S3_AUDIO_ROOT) is required.")
    if not args.s3_roi_root:
        raise ValueError("--s3-roi-root (or AVA_S3_ROI_ROOT) is required.")
    if args.num_shards is None:
        raise ValueError("--num-shards (or AVA_NUM_SHARDS) is required for deterministic array sharding.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if args.download_jobs <= 0:
        raise ValueError("--download-jobs must be positive.")

    include_patterns = args.include or ["*.wav", "*.WAV"]

    # We still compute the shard plan in dry-run mode without requiring awscli.
    local_manifest_path = args.workdir / "inputs" / "manifest.json"
    local_segment_config_path = args.workdir / "inputs" / "segment_config.yaml"
    local_manifest_subset_path = args.workdir / "inputs" / "manifest_subset.json"
    local_audio_root = args.workdir / "audio"
    local_roi_root = args.workdir / "roi"
    local_roi_summary_path = args.workdir / "out" / "roi_summary.json"
    local_job_summary_path = args.workdir / "out" / "job_summary.json"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_s3_uri": str(args.manifest_s3_uri),
                    "segment_config_s3_uri": str(args.segment_config_s3_uri),
                    "s3_audio_root": str(args.s3_audio_root),
                    "s3_roi_root": str(args.s3_roi_root),
                    "split": str(args.split),
                    "num_shards": int(args.num_shards),
                    "shard_index": int(args.shard_index),
                    "max_dirs": args.max_dirs,
                    "workdir": args.workdir.as_posix(),
                    "local_audio_root": local_audio_root.as_posix(),
                    "local_roi_root": local_roi_root.as_posix(),
                },
                indent=2,
            )
        )
        return

    aws = _require_aws_cli()

    args.workdir.mkdir(parents=True, exist_ok=True)

    # Download inputs.
    local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _run([aws, "s3", "cp", str(args.manifest_s3_uri), str(local_manifest_path)])
    _run([aws, "s3", "cp", str(args.segment_config_s3_uri), str(local_segment_config_path)])

    manifest = _load_json(local_manifest_path)
    pairs = iter_manifest_entry_pairs(manifest, split=str(args.split))
    pairs = apply_max_dirs(pairs, max_dirs=args.max_dirs)
    planned_total = len(pairs)
    pairs = select_shard(pairs, num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    shard_total = len(pairs)

    # Filter entries that already have parquet output in S3.
    skipped_existing = 0
    if args.skip_existing:
        remaining: list[tuple[str, dict]] = []
        for split_name, entry in pairs:
            rel = entry.get("audio_dir_rel")
            if rel is None:
                raise ValueError("Manifest entry missing audio_dir_rel (required for S3 layout).")
            out_uri = _join_s3_uri(str(args.s3_roi_root), f"{rel}/{args.roi_parquet_name}")
            if _s3_object_exists(aws, out_uri):
                skipped_existing += 1
                continue
            remaining.append((split_name, entry))
        pairs = remaining
    to_process = len(pairs)

    if not pairs:
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "nothing_to_do",
            "manifest_s3_uri": str(args.manifest_s3_uri),
            "split": str(args.split),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "planned_total": int(planned_total),
            "selected_for_shard": int(shard_total),
            "skipped_existing": int(skipped_existing),
            "to_process": int(to_process),
        }
        _write_json(local_job_summary_path, summary)
        print(json.dumps(summary, indent=2))
        if args.s3_summary_root:
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_job_summary_path),
                    _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                ]
            )
        return

    subset_manifest = pairs_to_manifest(pairs)
    _write_json(local_manifest_subset_path, subset_manifest)

    # Download audio directories for this shard.
    quiet = True
    download_failures = 0
    download_results: list[dict] = []
    start_dl = time.time()
    with ThreadPoolExecutor(max_workers=int(args.download_jobs)) as pool:
        futures = []
        for _, entry in pairs:
            rel = entry.get("audio_dir_rel")
            if rel is None:
                raise ValueError("Manifest entry missing audio_dir_rel (required for S3 layout).")
            s3_src = _join_s3_uri(str(args.s3_audio_root), str(rel))
            local_dst = local_audio_root / str(rel)
            futures.append(
                pool.submit(
                    _download_one_dir,
                    aws,
                    s3_src=s3_src,
                    local_dst=local_dst,
                    include_patterns=include_patterns,
                    quiet=quiet,
                )
            )
        for fut in as_completed(futures):
            res = fut.result()
            download_results.append(res)
            if res.get("status") != "ok":
                download_failures += 1
    elapsed_dl = max(1e-9, time.time() - start_dl)

    if download_failures:
        # Fail fast before segmentation to make retries clean.
        summary = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "download_failed",
            "download_failures": int(download_failures),
            "download_total": int(len(download_results)),
            "download_elapsed_sec": float(elapsed_dl),
            "download_tasks_per_sec": float(len(download_results) / elapsed_dl),
            "errors": [r for r in download_results if r.get("status") != "ok"],
        }
        _write_json(local_job_summary_path, summary)
        print(json.dumps(summary, indent=2))
        sys.exit(1)

    # Run ROI extraction on the subset manifest (no further sharding).
    roi_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_birdsong_roi.py"),
        "--segment-config",
        str(local_segment_config_path),
        "--manifest",
        str(local_manifest_subset_path),
        "--split",
        str(args.split),
        "--audio-root",
        str(local_audio_root),
        "--roi-root",
        str(local_roi_root),
        "--roi-output-format",
        str(args.roi_output_format),
        "--roi-parquet-name",
        str(args.roi_parquet_name),
        "--num-shards",
        "1",
        "--shard-index",
        "0",
        "--skip-existing",
        "--summary-out",
        str(local_roi_summary_path),
    ]
    if args.jobs is not None:
        roi_cmd.extend(["--jobs", str(int(args.jobs))])

    _run(roi_cmd, check=True)

    # Upload ROI parquet bundles.
    sync_cmd = [
        aws,
        "s3",
        "sync",
        str(local_roi_root),
        str(args.s3_roi_root),
        "--exclude",
        "*",
        "--include",
        f"*{args.roi_parquet_name}",
        "--only-show-errors",
    ]
    _run(sync_cmd, check=True)

    # Final summary.
    roi_summary = _load_json(local_roi_summary_path) if local_roi_summary_path.exists() else {}
    job_summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "manifest_s3_uri": str(args.manifest_s3_uri),
        "segment_config_s3_uri": str(args.segment_config_s3_uri),
        "s3_audio_root": str(args.s3_audio_root),
        "s3_roi_root": str(args.s3_roi_root),
        "split": str(args.split),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "max_dirs": args.max_dirs,
        "planned_total": int(planned_total),
        "selected_for_shard": int(shard_total),
        "skipped_existing": int(skipped_existing),
        "to_process": int(to_process),
        "download_jobs": int(args.download_jobs),
        "download_total": int(len(download_results)),
        "download_elapsed_sec": float(elapsed_dl),
        "roi_summary": roi_summary,
    }
    _write_json(local_job_summary_path, job_summary)
    print(json.dumps(job_summary, indent=2))

    if args.s3_summary_root:
        _run(
            [
                aws,
                "s3",
                "cp",
                str(local_job_summary_path),
                _join_s3_uri(str(args.s3_summary_root), f"job_summary_shard_{args.shard_index}.json"),
                "--only-show-errors",
            ]
        )
        if local_roi_summary_path.exists():
            _run(
                [
                    aws,
                    "s3",
                    "cp",
                    str(local_roi_summary_path),
                    _join_s3_uri(str(args.s3_summary_root), f"roi_summary_shard_{args.shard_index}.json"),
                    "--only-show-errors",
                ]
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
