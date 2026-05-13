#!/usr/bin/env python3
"""Inventory local inputs for the fixed developmental replication cohort."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.analysis.developmental_input_inventory import (  # noqa: E402
	DEFAULT_AUDIO_ROOT,
	DEFAULT_INVENTORY_BEAD,
	DEFAULT_PK249_LATENT_ROOT,
	DEFAULT_PK249_ROI_ROOT,
	DEFAULT_ROI_ROOT,
	build_developmental_input_inventory,
	write_inventory_artifacts,
)
from ava.analysis.developmental_replication import DEFAULT_BIRD_IDS  # noqa: E402


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-developmental-input-inventory", time.localtime())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Inventory local audio/ROI/latent coverage for developmental branch replication."
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "data" / "manifests" / "birdsong_manifest.json",
	)
	parser.add_argument("--bird-ids", type=str, default=",".join(DEFAULT_BIRD_IDS))
	parser.add_argument("--dph-min", type=float, default=33)
	parser.add_argument("--dph-max", type=float, default=90)
	parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
	parser.add_argument(
		"--roi-root",
		action="append",
		type=Path,
		default=[DEFAULT_PK249_ROI_ROOT, DEFAULT_ROI_ROOT],
		help="Local ROI root. May be repeated; first matching root wins.",
	)
	parser.add_argument("--roi-format", choices=["auto", "parquet", "txt"], default="auto")
	parser.add_argument(
		"--latent-root",
		action="append",
		type=Path,
		default=[DEFAULT_PK249_LATENT_ROOT],
		help="Local latent sequence root. May be repeated.",
	)
	parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
	parser.add_argument("--no-count-files", action="store_true")
	parser.add_argument("--bead-id", type=str, default=DEFAULT_INVENTORY_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--report-out", type=Path, default=None)

	args = parser.parse_args()
	bird_ids = [item.strip().upper() for item in args.bird_ids.split(",") if item.strip()]
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
	inventory = build_developmental_input_inventory(
		manifest_path=args.manifest,
		bird_ids=bird_ids,
		dph_min=args.dph_min,
		dph_max=args.dph_max,
		audio_root=args.audio_root,
		roi_roots=args.roi_root,
		latent_roots=args.latent_root,
		roi_parquet_name=args.roi_parquet_name,
		roi_format=args.roi_format,
		count_files=not args.no_count_files,
	)
	artifacts = write_inventory_artifacts(inventory, out_dir, report_path)
	print(f"Wrote report: {report_path.as_posix()}")
	print(f"Wrote artifacts: {out_dir.as_posix()}")
	print(f"Ready birds: {inventory['summary']['birds_ready_for_full_rebuild']} / {inventory['summary']['birds_requested']}")
	print("Missing latent birds: " + ",".join(inventory["summary"]["birds_missing_latents"]))
	print(f"Inventory JSON: {artifacts['input_inventory']}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
