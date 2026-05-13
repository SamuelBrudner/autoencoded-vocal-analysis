#!/usr/bin/env python3
"""Compare baseline AVA and shotgun VAE developmental replication metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.analysis.shotgun_development import (  # noqa: E402
	DEFAULT_SHOTGUN_BEAD,
	SHOTGUN_MODEL_ID,
	compare_replication_metrics,
	write_comparison_artifacts,
)


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-shotgun-comparison", time.localtime())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Write a report comparing baseline AVA and shotgun VAE developmental replication metrics."
	)
	parser.add_argument("--baseline-per-bird", type=Path, required=True)
	parser.add_argument("--shotgun-per-bird", type=Path, required=True)
	parser.add_argument("--baseline-label", type=str, default="ava_latent")
	parser.add_argument("--shotgun-label", type=str, default=SHOTGUN_MODEL_ID)
	parser.add_argument("--bead-id", type=str, default=DEFAULT_SHOTGUN_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--report-out", type=Path, default=None)

	args = parser.parse_args()
	baseline = json.loads(args.baseline_per_bird.read_text(encoding="utf-8"))
	shotgun = json.loads(args.shotgun_per_bird.read_text(encoding="utf-8"))
	comparison = compare_replication_metrics(
		baseline,
		shotgun,
		baseline_label=args.baseline_label,
		shotgun_label=args.shotgun_label,
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
	written = write_comparison_artifacts(comparison, out_dir, report_path)
	print(f"Wrote report: {written['report_path']}")
	print(f"Wrote artifacts: {out_dir.as_posix()}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
