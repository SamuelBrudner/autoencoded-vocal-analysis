#!/usr/bin/env python3
"""Render or backfill a training dashboard for an AVA run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.models.training_dashboard import backfill_training_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a lightweight AVA training dashboard from a run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing run_metadata.json and/or lightning_logs.",
    )
    parser.add_argument(
        "--status",
        choices=["unknown", "running", "completed", "failed"],
        default="unknown",
        help="Run status used when backfilling from older artifacts.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=30,
        help="Auto-refresh cadence for the generated HTML while status=running.",
    )
    args = parser.parse_args()

    payload, json_path, html_path = backfill_training_dashboard(
        save_dir=args.run_dir.as_posix(),
        status=str(args.status),
        refresh_seconds=int(args.refresh_seconds),
    )
    print(f"Dashboard status: {payload.get('status', 'unknown')}")
    print(f"Dashboard JSON: {json_path}")
    print(f"Dashboard HTML: {html_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
