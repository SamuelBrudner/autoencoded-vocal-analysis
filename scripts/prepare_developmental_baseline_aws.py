#!/usr/bin/env python3
"""Prepare AWS dry-run artifacts for 11-bird developmental baseline staging."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.cloud.developmental_baseline_aws import (  # noqa: E402
	DEFAULT_AUDIO_ROOT,
	DEFAULT_COHORT_MANIFEST_REL,
	DEFAULT_PK249_LATENT_ROOT,
	DEFAULT_SEGMENT_CONFIG_REL,
	DEFAULT_STAGING_BEAD,
	run_preparation,
)


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-developmental-baseline-aws-plan", time.localtime())


def _env_or_required(name: str, value: str | None, arg_name: str) -> str:
	value = value or os.environ.get(name)
	if not value:
		raise ValueError(f"{arg_name} is required or set {name}.")
	return value


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Write AWS dry-run payloads and a reproducible staging report for the "
			"fixed 11-bird developmental baseline cohort. This command never submits "
			"AWS jobs or uploads audio."
		)
	)
	parser.add_argument(
		"--cohort-manifest",
		type=Path,
		default=ROOT / DEFAULT_COHORT_MANIFEST_REL,
	)
	parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
	parser.add_argument("--segment-config", type=Path, default=ROOT / DEFAULT_SEGMENT_CONFIG_REL)
	parser.add_argument("--pk249-latent-root", type=Path, default=DEFAULT_PK249_LATENT_ROOT)
	parser.add_argument("--s3-root", type=str, default=None, help="Defaults to AVA_S3_ROOT.")
	parser.add_argument("--roi-job-queue", type=str, default=None, help="Defaults to AVA_ROI_JOB_QUEUE.")
	parser.add_argument(
		"--roi-job-definition",
		type=str,
		default=None,
		help="Defaults to AVA_ROI_JOB_DEFINITION.",
	)
	parser.add_argument(
		"--latent-job-queue",
		type=str,
		default=None,
		help="Defaults to AVA_LATENT_JOB_QUEUE.",
	)
	parser.add_argument(
		"--latent-job-definition",
		type=str,
		default=None,
		help="Defaults to AVA_LATENT_JOB_DEFINITION.",
	)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--array-size", type=int, default=None)
	parser.add_argument("--smoke-max-dirs", type=int, default=2)
	parser.add_argument("--upload-jobs", type=int, default=8)
	parser.add_argument("--roi-download-jobs", type=int, default=8)
	parser.add_argument("--roi-jobs", type=int, default=8)
	parser.add_argument("--latent-download-jobs", type=int, default=8)
	parser.add_argument("--latent-batch-size", type=int, default=64)
	parser.add_argument("--latent-device", choices=["auto", "cpu", "cuda"], default="cpu")
	parser.add_argument("--run-aws-preflight", action="store_true")
	parser.add_argument("--aws-profile", type=str, default=None)
	parser.add_argument("--bead-id", type=str, default=DEFAULT_STAGING_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--report-out", type=Path, default=None)

	args = parser.parse_args()
	s3_root = _env_or_required("AVA_S3_ROOT", args.s3_root, "--s3-root")
	roi_job_queue = _env_or_required("AVA_ROI_JOB_QUEUE", args.roi_job_queue, "--roi-job-queue")
	roi_job_definition = _env_or_required(
		"AVA_ROI_JOB_DEFINITION", args.roi_job_definition, "--roi-job-definition"
	)
	latent_job_queue = _env_or_required(
		"AVA_LATENT_JOB_QUEUE", args.latent_job_queue, "--latent-job-queue"
	)
	latent_job_definition = _env_or_required(
		"AVA_LATENT_JOB_DEFINITION", args.latent_job_definition, "--latent-job-definition"
	)
	run_name = args.run_name or _default_run_name()
	out_dir = (
		args.out_dir
		if args.out_dir is not None
		else ROOT / "docs" / "runs" / "artifacts" / str(args.bead_id) / run_name
	)
	report_path = (
		args.report_out
		if args.report_out is not None
		else ROOT / "docs" / "runs" / f"{run_name}-{args.bead_id}.md"
	)
	plan = run_preparation(
		repo_root=ROOT,
		out_dir=out_dir,
		report_path=report_path,
		s3_root=s3_root,
		roi_job_queue=roi_job_queue,
		roi_job_definition=roi_job_definition,
		latent_job_queue=latent_job_queue,
		latent_job_definition=latent_job_definition,
		cohort_manifest_path=args.cohort_manifest,
		audio_root=args.audio_root,
		segment_config_path=args.segment_config,
		pk249_latent_root=args.pk249_latent_root,
		split=args.split,
		array_size=args.array_size,
		smoke_max_dirs=args.smoke_max_dirs,
		upload_jobs=args.upload_jobs,
		roi_download_jobs=args.roi_download_jobs,
		roi_jobs=args.roi_jobs,
		latent_download_jobs=args.latent_download_jobs,
		latent_batch_size=args.latent_batch_size,
		latent_device=args.latent_device,
		run_aws_preflight=args.run_aws_preflight,
		aws_profile=args.aws_profile,
	)
	print(f"Wrote report: {report_path.as_posix()}")
	print(f"Wrote artifacts: {out_dir.as_posix()}")
	print(f"Cohort directories: {plan['cohort_summary']['row_count']}")
	print(f"Batch array size: {plan['array_size']}")
	print("AWS jobs submitted: no")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
