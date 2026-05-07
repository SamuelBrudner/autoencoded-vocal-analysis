#!/usr/bin/env python3
"""Analyze hyperbolic developmental geometry in AVA latent exports."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.analysis.hyperbolic_development import (  # noqa: E402
	DEFAULT_RUN_BEAD,
	run_hyperbolic_development_analysis,
)


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-hyperbolic-development", time.localtime())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Post-hoc hyperbolic developmental analysis for AVA latent exports."
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "data" / "manifests" / "birdsong_manifest.json",
	)
	parser.add_argument("--latent-root", type=Path, required=True)
	parser.add_argument("--audio-root", type=Path, default=None)
	parser.add_argument("--roi-root", type=Path, default=None)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--bird-id", type=str, default="PK249")
	parser.add_argument("--dph-min", type=float, default=33)
	parser.add_argument("--dph-max", type=float, default=90)
	parser.add_argument("--early-dph-max", type=float, default=45)
	parser.add_argument("--late-dph-min", type=float, default=80)
	parser.add_argument("--roi-format", choices=["parquet", "txt"], default="parquet")
	parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
	parser.add_argument("--max-events-per-dph", type=int, default=2000)
	parser.add_argument("--embedding-max-events", type=int, default=6000)
	parser.add_argument("--knn", type=int, default=10)
	parser.add_argument("--embedding-epochs", type=int, default=300)
	parser.add_argument("--embedding-lr", type=float, default=0.05)
	parser.add_argument("--cluster-min-k", type=int, default=4)
	parser.add_argument("--cluster-max-k", type=int, default=12)
	parser.add_argument("--bootstrap-iterations", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--bead-id", type=str, default=DEFAULT_RUN_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)
	parser.add_argument("--report-out", type=Path, default=None)
	parser.add_argument("--use-energy-weighted", action="store_true")
	parser.add_argument("--skip-umap", action="store_true")
	parser.add_argument("--no-sensitivity", action="store_true")

	args = parser.parse_args()
	if args.max_events_per_dph is not None and args.max_events_per_dph <= 0:
		raise ValueError("--max-events-per-dph must be positive.")
	if args.embedding_max_events <= 0:
		raise ValueError("--embedding-max-events must be positive.")
	if args.knn <= 0:
		raise ValueError("--knn must be positive.")
	if args.embedding_epochs <= 0:
		raise ValueError("--embedding-epochs must be positive.")
	if args.bootstrap_iterations < 0:
		raise ValueError("--bootstrap-iterations must be non-negative.")

	run_name = args.run_name or _default_run_name()
	if args.out_dir is None:
		artifact_dir = (
			ROOT / "docs" / "runs" / "artifacts" / str(args.bead_id) / run_name
		)
	else:
		artifact_dir = args.out_dir
	if args.report_out is None:
		report_path = ROOT / "docs" / "runs" / f"{run_name}-{args.bead_id}.md"
	else:
		report_path = args.report_out

	metrics = run_hyperbolic_development_analysis(
		manifest_path=args.manifest,
		latent_root=args.latent_root,
		artifact_dir=artifact_dir,
		report_path=report_path,
		bird_id=args.bird_id,
		dph_min=args.dph_min,
		dph_max=args.dph_max,
		early_dph_max=args.early_dph_max,
		late_dph_min=args.late_dph_min,
		split=args.split,
		audio_root=args.audio_root,
		roi_root=args.roi_root,
		roi_format=args.roi_format,
		roi_parquet_name=args.roi_parquet_name,
		max_events_per_dph=args.max_events_per_dph,
		embedding_max_events=args.embedding_max_events,
		knn=args.knn,
		embedding_epochs=args.embedding_epochs,
		embedding_lr=args.embedding_lr,
		cluster_min_k=args.cluster_min_k,
		cluster_max_k=args.cluster_max_k,
		bootstrap_iterations=args.bootstrap_iterations,
		seed=args.seed,
		use_energy_weighted=args.use_energy_weighted,
		compute_umap=not args.skip_umap,
		run_sensitivity=not args.no_sensitivity,
	)
	print(f"Wrote report: {metrics['report_path']}")
	print(f"Wrote artifacts: {artifact_dir.as_posix()}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
