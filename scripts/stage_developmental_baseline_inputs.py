#!/usr/bin/env python3
"""Stage small AWS input files for developmental baseline replication.

By default this command only writes a staging manifest. Passing --execute uploads
the files to S3 with aws s3 cp. It never submits Batch jobs or uploads audio.
"""

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
	DEFAULT_COHORT_MANIFEST_REL,
	DEFAULT_SEGMENT_CONFIG_REL,
	build_input_staging_plan,
	upload_input_staging_plan,
	write_json,
)


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-developmental-baseline-inputs", time.localtime())


def _env_or_required(name: str, value: str | None, arg_name: str) -> str:
	value = value or os.environ.get(name)
	if not value:
		raise ValueError(f"{arg_name} is required or set {name}.")
	return value


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Write or execute the small-file S3 input staging plan for the fixed "
			"11-bird developmental baseline replication. Default mode is dry-run."
		)
	)
	parser.add_argument(
		"--cohort-manifest",
		type=Path,
		default=ROOT / DEFAULT_COHORT_MANIFEST_REL,
	)
	parser.add_argument(
		"--segment-config",
		type=Path,
		default=ROOT / DEFAULT_SEGMENT_CONFIG_REL,
	)
	parser.add_argument(
		"--ava-config-source",
		type=Path,
		default=os.environ.get("AVA_CONFIG_PATH"),
		help="Local recovered AVA config.yaml. Defaults to AVA_CONFIG_PATH.",
	)
	parser.add_argument(
		"--ava-checkpoint-source",
		type=Path,
		default=os.environ.get("AVA_CHECKPOINT_PATH"),
		help="Local recovered checkpoint_050.tar. Defaults to AVA_CHECKPOINT_PATH.",
	)
	parser.add_argument("--s3-root", type=str, default=None, help="Defaults to AVA_S3_ROOT.")
	parser.add_argument("--aws-profile", type=str, default=None)
	parser.add_argument("--execute", action="store_true", help="Actually upload files to S3.")
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Output JSON path. Defaults under docs/runs/artifacts/autoencoded-vocal-analysis-obi.4.1.",
	)

	args = parser.parse_args()
	s3_root = _env_or_required("AVA_S3_ROOT", args.s3_root, "--s3-root")
	ava_config = Path(
		_env_or_required(
			"AVA_CONFIG_PATH",
			str(args.ava_config_source) if args.ava_config_source else None,
			"--ava-config-source",
		)
	)
	ava_checkpoint = Path(
		_env_or_required(
			"AVA_CHECKPOINT_PATH",
			str(args.ava_checkpoint_source) if args.ava_checkpoint_source else None,
			"--ava-checkpoint-source",
		)
	)
	run_name = args.run_name or _default_run_name()
	out_path = (
		args.out
		if args.out is not None
		else ROOT
		/ "docs"
		/ "runs"
		/ "artifacts"
		/ "autoencoded-vocal-analysis-obi.4.1"
		/ run_name
		/ "input_staging_manifest.json"
	)

	plan = build_input_staging_plan(
		s3_root=s3_root,
		cohort_manifest_path=args.cohort_manifest,
		segment_config_path=args.segment_config,
		ava_config_path=ava_config,
		ava_checkpoint_path=ava_checkpoint,
	)
	if args.execute:
		plan["safety"]["uploads_to_s3"] = True
		plan["execution"] = upload_input_staging_plan(plan, aws_profile=args.aws_profile)
	else:
		plan["execution"] = {
			"status": "dry_run",
			"uploads_to_s3": False,
			"message": "Pass --execute to upload these files with aws s3 cp.",
		}
	write_json(out_path, plan)
	print(f"Wrote input staging manifest: {out_path.as_posix()}")
	print(f"Files: {len(plan['files'])}")
	print(f"Uploaded to S3: {'yes' if args.execute else 'no'}")
	if args.execute and plan["execution"]["status"] != "ok":
		sys.exit(1)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
