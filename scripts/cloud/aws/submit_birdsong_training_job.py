#!/usr/bin/env python3
"""Submit a single AWS Batch job for birdsong training."""

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
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}"
        )
    return proc


def _payload_env(name: str, value: str | int | float | None) -> dict | None:
    if value is None:
        return None
    return {"name": str(name), "value": str(value)}


def build_payload(args: argparse.Namespace) -> dict:
    env_items = [
        _payload_env("AVA_MANIFEST_S3_URI", args.manifest_s3_uri),
        _payload_env("AVA_CONFIG_S3_URI", args.config_s3_uri),
        _payload_env("AVA_S3_AUDIO_ROOT", args.s3_audio_root),
        _payload_env("AVA_S3_ROI_ROOT", args.s3_roi_root),
        _payload_env("AVA_S3_RUN_ROOT", args.s3_run_root),
        _payload_env("AVA_RUN_NAME", args.run_name),
        _payload_env("AVA_ROI_FORMAT", args.roi_format),
        _payload_env("AVA_ROI_PARQUET_NAME", args.roi_parquet_name),
        _payload_env("AVA_DOWNLOAD_JOBS", int(args.download_jobs)),
        _payload_env("AVA_BATCH_SIZE", int(args.batch_size) if args.batch_size is not None else None),
        _payload_env("AVA_NUM_WORKERS", int(args.num_workers) if args.num_workers is not None else None),
        _payload_env("AVA_EPOCHS", int(args.epochs) if args.epochs is not None else None),
        _payload_env("AVA_TRAIN_DATASET_LENGTH", int(args.train_dataset_length) if args.train_dataset_length is not None else None),
        _payload_env("AVA_TEST_DATASET_LENGTH", int(args.test_dataset_length) if args.test_dataset_length is not None else None),
        _payload_env("AVA_SPEC_CACHE_DIR", args.spec_cache_dir),
        _payload_env("AVA_TRAINER_KWARGS_JSON", args.trainer_kwargs_json),
        _payload_env("AVA_PREFLIGHT_SAMPLE_DIRS", int(args.preflight_sample_dirs)),
        _payload_env("AVA_PREFLIGHT_SAMPLE_SEGMENTS", int(args.preflight_sample_segments)),
        _payload_env("AVA_PREFLIGHT_SEED", int(args.preflight_seed)),
        _payload_env("AVA_MAX_EMPTY_FRACTION", float(args.max_empty_fraction)),
        _payload_env(
            "AVA_DISK_TELEMETRY_EVERY_N_EPOCHS",
            int(args.disk_telemetry_every_n_epochs)
            if args.disk_telemetry_every_n_epochs is not None
            else None,
        ),
        _payload_env("AVA_WORKDIR", args.workdir),
    ]
    env_items = [item for item in env_items if item is not None]

    payload: dict = {
        "jobName": str(args.job_name),
        "jobQueue": str(args.job_queue),
        "jobDefinition": str(args.job_definition),
        "containerOverrides": {"environment": env_items},
    }
    if args.depends_on_job_id:
        payload["dependsOn"] = [{"jobId": str(job_id)} for job_id in args.depends_on_job_id]
    if args.timeout_seconds is not None:
        payload["timeout"] = {"attemptDurationSeconds": int(args.timeout_seconds)}
    if args.override_command:
        payload["containerOverrides"]["command"] = [
            "python",
            "scripts/cloud/aws/run_birdsong_training_batch_job.py",
        ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a single AWS Batch job for birdsong training.")
    parser.add_argument("--job-name", type=str, required=True)
    parser.add_argument("--job-queue", type=str, required=True)
    parser.add_argument("--job-definition", type=str, required=True)

    parser.add_argument("--manifest-s3-uri", type=str, required=True)
    parser.add_argument("--config-s3-uri", type=str, required=True)
    parser.add_argument("--s3-audio-root", type=str, required=True)
    parser.add_argument("--s3-roi-root", type=str, required=True)
    parser.add_argument("--s3-run-root", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--roi-format", choices=["txt", "parquet"], default="parquet")
    parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
    parser.add_argument("--download-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-dataset-length", type=int, default=None)
    parser.add_argument("--test-dataset-length", type=int, default=None)
    parser.add_argument("--spec-cache-dir", type=str, default="/mnt/ava_cache/spec_cache")
    parser.add_argument("--trainer-kwargs-json", type=str, default=None)
    parser.add_argument("--preflight-sample-dirs", type=int, default=25)
    parser.add_argument("--preflight-sample-segments", type=int, default=5000)
    parser.add_argument("--preflight-seed", type=int, default=0)
    parser.add_argument("--max-empty-fraction", type=float, default=0.01)
    parser.add_argument("--disk-telemetry-every-n-epochs", type=int, default=5)
    parser.add_argument("--workdir", type=str, default="/mnt/ava_cache/ava_train_workdir")
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--depends-on-job-id", action="append", default=None)
    parser.add_argument(
        "--override-command",
        action="store_true",
        help="Override the container command to run the training runner explicitly.",
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

    if args.download_jobs <= 0:
        raise ValueError("--download-jobs must be positive.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.train_dataset_length is not None and args.train_dataset_length <= 0:
        raise ValueError("--train-dataset-length must be positive.")
    if args.test_dataset_length is not None and args.test_dataset_length <= 0:
        raise ValueError("--test-dataset-length must be positive.")
    if args.preflight_sample_dirs <= 0:
        raise ValueError("--preflight-sample-dirs must be positive.")
    if args.preflight_sample_segments <= 0:
        raise ValueError("--preflight-sample-segments must be positive.")
    if not (0.0 <= float(args.max_empty_fraction) <= 1.0):
        raise ValueError("--max-empty-fraction must be in [0, 1].")
    if args.disk_telemetry_every_n_epochs is not None and args.disk_telemetry_every_n_epochs <= 0:
        raise ValueError("--disk-telemetry-every-n-epochs must be positive.")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")

    payload = build_payload(args)
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
