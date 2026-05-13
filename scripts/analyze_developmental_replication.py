#!/usr/bin/env python3
"""Run multi-bird developmental branch-commitment replication."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.analysis.developmental_replication import (  # noqa: E402
	DEFAULT_BIRD_IDS,
	DEFAULT_REPLICATION_BEAD,
	run_developmental_replication_analysis,
	select_top_longitudinal_birds,
)
from ava.analysis.hyperbolic_development import load_manifest  # noqa: E402


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-developmental-replication", time.localtime())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Multi-bird developmental branch-commitment replication for AVA event latents."
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "data" / "manifests" / "birdsong_manifest.json",
	)
	parser.add_argument(
		"--cohort",
		choices=["top-longitudinal", "explicit"],
		default="top-longitudinal",
	)
	parser.add_argument(
		"--bird-ids",
		type=str,
		default=",".join(DEFAULT_BIRD_IDS),
		help="Comma-separated bird ids. Defaults to PK249 plus the fixed top-longitudinal cohort.",
	)
	parser.add_argument(
		"--event-table",
		action="append",
		default=[],
		help="Explicit per-bird event table as BIRD=/path/event_latents.parquet. May be repeated.",
	)
	parser.add_argument(
		"--event-table-root",
		action="append",
		type=Path,
		default=[ROOT / "docs" / "runs" / "artifacts"],
		help="Root to scan recursively for event_latents.parquet. May be repeated.",
	)
	parser.add_argument("--latent-root", type=Path, default=None)
	parser.add_argument("--audio-root", type=Path, default=None)
	parser.add_argument("--roi-root", type=Path, default=None)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--dph-min", type=float, default=33)
	parser.add_argument("--dph-max", type=float, default=90)
	parser.add_argument("--early-dph-max", type=float, default=45)
	parser.add_argument("--late-dph-min", type=float, default=80)
	parser.add_argument("--roi-format", choices=["parquet", "txt"], default="parquet")
	parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
	parser.add_argument("--max-events-per-dph", type=int, default=2000)
	parser.add_argument("--cluster-min-k", type=int, default=4)
	parser.add_argument("--cluster-max-k", type=int, default=12)
	parser.add_argument("--bootstrap-iterations", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--bead-id", type=str, default=DEFAULT_REPLICATION_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--report-out", type=Path, default=None)
	parser.add_argument("--use-energy-weighted", action="store_true")
	parser.add_argument("--embedding-max-events", type=int, default=6000)
	parser.add_argument("--embedding-knn", type=int, default=10)
	parser.add_argument("--embedding-epochs", type=int, default=300)
	parser.add_argument("--embedding-lr", type=float, default=0.01)
	parser.add_argument("--embedding-max-edges", type=int, default=200_000)
	parser.add_argument("--skip-radius", action="store_true")
	parser.add_argument("--skip-bias-sensitivity", action="store_true")

	args = parser.parse_args()
	if args.max_events_per_dph is not None and args.max_events_per_dph <= 0:
		raise ValueError("--max-events-per-dph must be positive.")
	if args.cluster_min_k <= 0 or args.cluster_max_k <= 0:
		raise ValueError("cluster K bounds must be positive.")
	if args.cluster_max_k < args.cluster_min_k:
		raise ValueError("--cluster-max-k must be >= --cluster-min-k.")
	if args.bootstrap_iterations < 0:
		raise ValueError("--bootstrap-iterations must be non-negative.")
	if args.embedding_max_events <= 1:
		raise ValueError("--embedding-max-events must be > 1.")
	if args.embedding_epochs < 0:
		raise ValueError("--embedding-epochs must be non-negative.")

	event_tables = _parse_event_tables(args.event_table)
	bird_ids = _parse_bird_ids(args.bird_ids)
	if args.cohort == "top-longitudinal" and not args.bird_ids:
		bird_ids = select_top_longitudinal_birds(load_manifest(args.manifest))

	run_name = args.run_name or _default_run_name()
	if args.out_dir is None:
		artifact_dir = ROOT / "docs" / "runs" / "artifacts" / str(args.bead_id) / run_name
	else:
		artifact_dir = args.out_dir
	if args.report_out is None:
		report_path = ROOT / "docs" / "runs" / f"{run_name}-{args.bead_id}.md"
	else:
		report_path = args.report_out

	metrics = run_developmental_replication_analysis(
		artifact_dir=artifact_dir,
		report_path=report_path,
		manifest_path=args.manifest,
		bird_ids=bird_ids,
		cohort=args.cohort,
		event_tables=event_tables,
		event_table_roots=args.event_table_root,
		latent_root=args.latent_root,
		audio_root=args.audio_root,
		roi_root=args.roi_root,
		split=args.split,
		dph_min=args.dph_min,
		dph_max=args.dph_max,
		early_dph_max=args.early_dph_max,
		late_dph_min=args.late_dph_min,
		roi_format=args.roi_format,
		roi_parquet_name=args.roi_parquet_name,
		max_events_per_dph=args.max_events_per_dph,
		cluster_min_k=args.cluster_min_k,
		cluster_max_k=args.cluster_max_k,
		bootstrap_iterations=args.bootstrap_iterations,
		seed=args.seed,
		use_energy_weighted=args.use_energy_weighted,
		embedding_max_events=args.embedding_max_events,
		embedding_knn=args.embedding_knn,
		embedding_epochs=args.embedding_epochs,
		embedding_lr=args.embedding_lr,
		embedding_max_edges=args.embedding_max_edges,
		skip_radius=args.skip_radius,
		skip_bias_sensitivity=args.skip_bias_sensitivity,
	)
	print(f"Wrote report: {metrics['report_path']}")
	print(f"Wrote artifacts: {artifact_dir.as_posix()}")


def _parse_bird_ids(value: str) -> list[str]:
	return [item.strip().upper() for item in str(value).split(",") if item.strip()]


def _parse_event_tables(values: list[str]) -> dict[str, Path]:
	out = {}
	for value in values:
		if "=" not in value:
			raise ValueError("--event-table entries must be formatted as BIRD=/path")
		bird, path = value.split("=", 1)
		bird = bird.strip().upper()
		if not bird:
			raise ValueError("--event-table bird id cannot be empty")
		out[bird] = Path(path)
	return out


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
