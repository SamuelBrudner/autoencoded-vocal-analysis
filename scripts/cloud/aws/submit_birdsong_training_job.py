#!/usr/bin/env python3
"""Submit one AWS Batch GPU job for fixed-window shotgun VAE training."""

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
	parser = argparse.ArgumentParser(description="Submit an AWS Batch job for shotgun VAE training.")
	parser.add_argument("--job-name", type=str, required=True)
	parser.add_argument("--job-queue", type=str, required=True)
	parser.add_argument("--job-definition", type=str, required=True)

	parser.add_argument("--manifest-s3-uri", type=str, required=True)
	parser.add_argument("--config-s3-uri", type=str, required=True)
	parser.add_argument("--s3-audio-root", type=str, required=True)
	parser.add_argument("--s3-roi-root", type=str, required=True)
	parser.add_argument("--s3-output-root", type=str, required=True)
	parser.add_argument("--run-name", type=str, default=None)

	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--dataset-length", type=int, default=200_000)
	parser.add_argument("--roi-cache-size", type=int, default=32)
	parser.add_argument("--preflight-sample-dirs", type=int, default=100)
	parser.add_argument("--preflight-sample-segments", type=int, default=50_000)
	parser.add_argument("--preflight-seed", type=int, default=0)
	parser.add_argument("--download-jobs", type=int, default=8)
	parser.add_argument("--roi-format", choices=["txt", "parquet"], default="parquet")
	parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
	parser.add_argument("--trainer-kwargs-json", type=str, default=None)

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
	if args.epochs <= 0:
		raise ValueError("--epochs must be positive.")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be positive.")
	if args.num_workers < 0:
		raise ValueError("--num-workers must be non-negative.")
	if args.dataset_length <= 0:
		raise ValueError("--dataset-length must be positive.")
	if args.roi_cache_size <= 0:
		raise ValueError("--roi-cache-size must be positive.")
	if args.download_jobs <= 0:
		raise ValueError("--download-jobs must be positive.")

	env_items = [
		_payload_env("AVA_TRAIN_MANIFEST_S3_URI", args.manifest_s3_uri),
		_payload_env("AVA_TRAIN_CONFIG_S3_URI", args.config_s3_uri),
		_payload_env("AVA_TRAIN_S3_AUDIO_ROOT", args.s3_audio_root),
		_payload_env("AVA_TRAIN_S3_ROI_ROOT", args.s3_roi_root),
		_payload_env("AVA_TRAIN_S3_OUTPUT_ROOT", args.s3_output_root),
		_payload_env("AVA_TRAIN_RUN_NAME", args.run_name),
		_payload_env("AVA_TRAIN_EPOCHS", int(args.epochs)),
		_payload_env("AVA_TRAIN_BATCH_SIZE", int(args.batch_size)),
		_payload_env("AVA_TRAIN_NUM_WORKERS", int(args.num_workers)),
		_payload_env("AVA_TRAIN_DATASET_LENGTH", int(args.dataset_length)),
		_payload_env("AVA_TRAIN_ROI_CACHE_SIZE", int(args.roi_cache_size)),
		_payload_env("AVA_TRAIN_PREFLIGHT_SAMPLE_DIRS", int(args.preflight_sample_dirs)),
		_payload_env("AVA_TRAIN_PREFLIGHT_SAMPLE_SEGMENTS", int(args.preflight_sample_segments)),
		_payload_env("AVA_TRAIN_PREFLIGHT_SEED", int(args.preflight_seed)),
		_payload_env("AVA_TRAIN_DOWNLOAD_JOBS", int(args.download_jobs)),
		_payload_env("AVA_TRAIN_ROI_FORMAT", args.roi_format),
		_payload_env("AVA_TRAIN_ROI_PARQUET_NAME", args.roi_parquet_name),
		_payload_env("AVA_TRAIN_TRAINER_KWARGS_JSON", args.trainer_kwargs_json),
	]
	env_items = [item for item in env_items if item is not None]

	container_overrides: dict = {"environment": env_items}
	if args.override_command:
		container_overrides["command"] = [
			"python",
			"scripts/cloud/aws/run_birdsong_training_job.py",
		]

	payload = {
		"jobName": str(args.job_name),
		"jobQueue": str(args.job_queue),
		"jobDefinition": str(args.job_definition),
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
