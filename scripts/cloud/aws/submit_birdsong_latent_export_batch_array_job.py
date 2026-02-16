#!/usr/bin/env python3
"""Submit an AWS Batch array job for birdsong latent export.

This script generates (and optionally submits) an `aws batch submit-job`
payload for a latent-export array job.

By default this script does *not* override the container command. Use
`--override-command` only if your job definition expects an explicit command.
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


def _payload_env(name: str, value: str | int | float | None) -> dict | None:
    if value is None:
        return None
    return {"name": str(name), "value": str(value)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an AWS Batch array job for latent export.")
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--job-queue", type=str, required=True)
    parser.add_argument("--job-definition", type=str, required=True)
    parser.add_argument("--array-size", type=int, required=True, help="Number of array children (num_shards).")

    parser.add_argument("--manifest-s3-uri", type=str, required=True)
    parser.add_argument("--config-s3-uri", type=str, required=True)
    parser.add_argument("--checkpoint-s3-uri", type=str, required=True)
    parser.add_argument("--s3-audio-root", type=str, required=True)
    parser.add_argument("--s3-roi-root", type=str, default=None)
    parser.add_argument("--s3-latent-root", type=str, required=True)

    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--max-dirs", type=int, default=None)
    parser.add_argument("--max-files-per-dir", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)

    parser.add_argument("--download-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--roi-format", choices=["txt", "parquet"], default="parquet")
    parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
    parser.add_argument("--no-rois", action="store_true")

    parser.add_argument("--hop-length-sec", type=float, default=None)
    parser.add_argument("--start-time-sec", type=float, default=0.0)
    parser.add_argument("--end-time-sec", type=float, default=None)

    parser.add_argument("--export-energy", action="store_true")
    parser.add_argument("--audio-sha256", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--report-every", type=int, default=25)

    parser.add_argument(
        "--override-command",
        action="store_true",
        help="Override the container command to run the latent shard runner explicitly.",
    )
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
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.no_rois and args.s3_roi_root:
        # Harmless, but keep payload clean.
        args.s3_roi_root = None
    if (not args.no_rois) and not args.s3_roi_root:
        raise ValueError("--s3-roi-root is required unless --no-rois is set.")

    env_items = [
        _payload_env("AVA_MANIFEST_S3_URI", args.manifest_s3_uri),
        _payload_env("AVA_CONFIG_S3_URI", args.config_s3_uri),
        _payload_env("AVA_CHECKPOINT_S3_URI", args.checkpoint_s3_uri),
        _payload_env("AVA_S3_AUDIO_ROOT", args.s3_audio_root),
        _payload_env("AVA_S3_ROI_ROOT", args.s3_roi_root),
        _payload_env("AVA_S3_LATENT_ROOT", args.s3_latent_root),
        _payload_env("AVA_SPLIT", args.split),
        _payload_env("AVA_NUM_SHARDS", int(args.array_size)),
        _payload_env("AVA_MAX_DIRS", args.max_dirs),
        _payload_env("AVA_MAX_FILES_PER_DIR", args.max_files_per_dir),
        _payload_env("AVA_MAX_CLIPS", args.max_clips),
        _payload_env("AVA_DOWNLOAD_JOBS", int(args.download_jobs)),
        _payload_env("AVA_BATCH_SIZE", int(args.batch_size)),
        _payload_env("AVA_DEVICE", args.device),
        _payload_env("AVA_ROI_FORMAT", args.roi_format),
        _payload_env("AVA_ROI_PARQUET_NAME", args.roi_parquet_name),
        _payload_env("AVA_NO_ROIS", "1" if args.no_rois else "0"),
        _payload_env("AVA_HOP_LENGTH_SEC", args.hop_length_sec),
        _payload_env("AVA_START_TIME_SEC", args.start_time_sec),
        _payload_env("AVA_END_TIME_SEC", args.end_time_sec),
        _payload_env("AVA_EXPORT_ENERGY", "1" if args.export_energy else "0"),
        _payload_env("AVA_AUDIO_SHA256", "1" if args.audio_sha256 else "0"),
        _payload_env("AVA_SKIP_EXISTING", "1" if args.skip_existing else "0"),
        _payload_env("AVA_CONTINUE_ON_ERROR", "1" if args.continue_on_error else "0"),
        _payload_env("AVA_REPORT_EVERY", int(args.report_every)),
        _payload_env("AVA_S3_SUMMARY_ROOT", args.s3_summary_root),
    ]
    env_items = [item for item in env_items if item is not None]

    container_overrides: dict = {"environment": env_items}
    if args.override_command:
        container_overrides["command"] = [
            "python",
            "scripts/cloud/aws/run_birdsong_latent_export_batch_shard.py",
        ]

    payload = {
        "jobName": str(args.job_name),
        "jobQueue": str(args.job_queue),
        "jobDefinition": str(args.job_definition),
        "arrayProperties": {"size": int(args.array_size)},
        "containerOverrides": container_overrides,
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
