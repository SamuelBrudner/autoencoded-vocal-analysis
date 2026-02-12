#!/usr/bin/env python3
"""Submit an AWS Batch array job for birdsong ROI extraction.

This script generates (and optionally submits) an `aws batch submit-job`
payload that runs `scripts/cloud/aws/run_birdsong_roi_batch_shard.py` as an
array job.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _require_aws_cli() -> str:
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError(
            "aws CLI is required but was not found on PATH. "
            "Install awscli (or AWS CLI v2) and configure credentials."
        )
    return aws


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}")
    return proc


def _payload_env(name: str, value: str | int | None) -> dict | None:
    if value is None:
        return None
    return {"name": str(name), "value": str(value)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an AWS Batch array job for ROI extraction.")
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--job-queue", type=str, required=True)
    parser.add_argument("--job-definition", type=str, required=True)
    parser.add_argument("--array-size", type=int, required=True, help="Number of array children (num_shards).")

    parser.add_argument("--manifest-s3-uri", type=str, required=True)
    parser.add_argument("--segment-config-s3-uri", type=str, required=True)
    parser.add_argument("--s3-audio-root", type=str, required=True)
    parser.add_argument("--s3-roi-root", type=str, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
    parser.add_argument("--download-jobs", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=None, help="Parallelism inside ROI script (optional).")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--s3-summary-root",
        type=str,
        default=None,
        help="Optional S3 URI root for per-shard summaries.",
    )

    parser.add_argument(
        "--emit-json",
        type=Path,
        default=None,
        help="If set, write the submit-job payload JSON to this path.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually run `aws batch submit-job`. Without this flag, prints the JSON payload.",
    )

    args = parser.parse_args()

    if args.array_size <= 0:
        raise ValueError("--array-size must be positive.")
    if args.download_jobs <= 0:
        raise ValueError("--download-jobs must be positive.")

    env_items = [
        _payload_env("AVA_MANIFEST_S3_URI", args.manifest_s3_uri),
        _payload_env("AVA_SEGMENT_CONFIG_S3_URI", args.segment_config_s3_uri),
        _payload_env("AVA_S3_AUDIO_ROOT", args.s3_audio_root),
        _payload_env("AVA_S3_ROI_ROOT", args.s3_roi_root),
        _payload_env("AVA_SPLIT", args.split),
        _payload_env("AVA_NUM_SHARDS", int(args.array_size)),
        _payload_env("AVA_ROI_OUTPUT_FORMAT", "parquet"),
        _payload_env("AVA_ROI_PARQUET_NAME", args.roi_parquet_name),
        _payload_env("AVA_DOWNLOAD_JOBS", int(args.download_jobs)),
        _payload_env("AVA_SKIP_EXISTING", "1" if args.skip_existing else "0"),
        _payload_env("AVA_JOBS", args.jobs),
        _payload_env("AVA_S3_SUMMARY_ROOT", args.s3_summary_root),
    ]
    env_items = [item for item in env_items if item is not None]

    payload = {
        "jobName": str(args.job_name),
        "jobQueue": str(args.job_queue),
        "jobDefinition": str(args.job_definition),
        "arrayProperties": {"size": int(args.array_size)},
        "containerOverrides": {
            "command": ["python", "scripts/cloud/aws/run_birdsong_roi_batch_shard.py"],
            "environment": env_items,
        },
    }

    rendered = json.dumps(payload, indent=2)
    if args.emit_json is not None:
        args.emit_json.parent.mkdir(parents=True, exist_ok=True)
        args.emit_json.write_text(rendered, encoding="utf-8")

    if not args.submit:
        print(rendered)
        return

    aws = _require_aws_cli()
    proc = _run([aws, "batch", "submit-job", "--cli-input-json", rendered])
    print(proc.stdout.strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

