#!/usr/bin/env python3
"""Build fixed developmental shotgun VAE pilot/cohort manifests and configs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.analysis.developmental_replication import DEFAULT_BIRD_IDS  # noqa: E402
from ava.analysis.shotgun_development import (  # noqa: E402
	DEFAULT_SHOTGUN_BEAD,
	write_shotgun_config,
	write_shotgun_manifests,
)


def _default_run_name() -> str:
	return time.strftime("%Y%m%d-%H%M%S-shotgun-cohort", time.localtime())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Write PK249 pilot and fixed 11-bird cohort manifests for shotgun VAE training/export."
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		default=ROOT / "data" / "manifests" / "birdsong_manifest.json",
	)
	parser.add_argument(
		"--base-config",
		type=Path,
		default=ROOT / "examples" / "configs" / "fixed_window_finch_30ms_44k.yaml",
	)
	parser.add_argument("--pilot-bird", type=str, default="PK249")
	parser.add_argument("--bird-ids", type=str, default=",".join(DEFAULT_BIRD_IDS))
	parser.add_argument("--dph-min", type=float, default=33)
	parser.add_argument("--dph-max", type=float, default=90)
	parser.add_argument("--pilot-epochs", type=int, default=10)
	parser.add_argument("--cohort-epochs", type=int, default=100)
	parser.add_argument("--min-freq", type=float, default=300.0)
	parser.add_argument("--spec-min-val", type=float, default=1.0)
	parser.add_argument("--kl-beta", type=float, default=1.0)
	parser.add_argument("--kl-warmup-epochs", type=int, default=20)
	parser.add_argument("--bead-id", type=str, default=DEFAULT_SHOTGUN_BEAD)
	parser.add_argument("--run-name", type=str, default=None)
	parser.add_argument("--out-dir", type=Path, default=None)

	args = parser.parse_args()
	if args.pilot_epochs <= 0 or args.cohort_epochs <= 0:
		raise ValueError("epoch counts must be positive.")

	run_name = args.run_name or _default_run_name()
	out_dir = (
		args.out_dir
		if args.out_dir is not None
		else ROOT / "docs" / "runs" / "artifacts" / str(args.bead_id) / run_name
	)
	birds = [item.strip().upper() for item in args.bird_ids.split(",") if item.strip()]
	summary = write_shotgun_manifests(
		manifest_path=args.manifest,
		out_dir=out_dir,
		pilot_bird=args.pilot_bird,
		cohort_birds=birds,
		dph_min=args.dph_min,
		dph_max=args.dph_max,
	)
	pilot_config = write_shotgun_config(
		base_config=args.base_config,
		out_path=out_dir / "shotgun_pk249_pilot_config.yaml",
		epochs=args.pilot_epochs,
		min_freq=args.min_freq,
		spec_min_val=args.spec_min_val,
		kl_beta=args.kl_beta,
		kl_warmup_epochs=args.kl_warmup_epochs,
	)
	cohort_config = write_shotgun_config(
		base_config=args.base_config,
		out_path=out_dir / "shotgun_fixed_11bird_config.yaml",
		epochs=args.cohort_epochs,
		min_freq=args.min_freq,
		spec_min_val=args.spec_min_val,
		kl_beta=args.kl_beta,
		kl_warmup_epochs=args.kl_warmup_epochs,
	)
	print(f"Wrote manifests: {out_dir.as_posix()}")
	print(f"Pilot manifest: {summary['artifacts']['pilot_manifest']}")
	print(f"Cohort manifest: {summary['artifacts']['cohort_manifest']}")
	print(f"Pilot config: {pilot_config['config']}")
	print(f"Cohort config: {cohort_config['config']}")


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
