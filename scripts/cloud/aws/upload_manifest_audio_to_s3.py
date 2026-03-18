#!/usr/bin/env python3
"""Upload the manifest audio subset to S3.

This script syncs each manifest entry's audio directory to an S3 prefix while
preserving the `audio_dir_rel` layout.

It intentionally uses `aws s3 sync` for:
- resume/idempotency (safe to rerun)
- multipart/concurrency handled by AWS CLI
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ava.cloud.manifest_sharding import (
    apply_max_dirs,
    iter_manifest_entry_pairs,
    select_shard,
)
from ava.data.manifest_paths import resolve_manifest_entry_paths


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _join_s3_uri(root_uri: str, rel: str) -> str:
    root_uri = str(root_uri).rstrip("/")
    rel = str(rel or ".").lstrip("/")
    if rel in (".", ""):
        return root_uri
    return f"{root_uri}/{rel}"


def _require_aws_cli() -> str:
    aws = shutil.which("aws")
    if not aws:
        raise RuntimeError(
            "aws CLI is required but was not found on PATH. "
            "Install awscli (or AWS CLI v2) and configure credentials."
        )
    return aws


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return int(proc.returncode), output.strip()


def _sync_one(
    aws: str,
    audio_dir: str,
    s3_target: str,
    include_patterns: list[str],
    quiet: bool,
) -> dict:
    start = time.time()
    # Include patterns only take effect when exclude-all is set first.
    cmd = [aws, "s3", "sync", audio_dir, s3_target, "--exclude", "*"]
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])
    if quiet:
        cmd.append("--only-show-errors")

    code, output = _run(cmd)
    return {
        "audio_dir": audio_dir,
        "s3_target": s3_target,
        "status": "ok" if code == 0 else "error",
        "returncode": int(code),
        "elapsed_sec": float(time.time() - start),
        "output": output if code != 0 else None,
        "cmd": cmd if code != 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the manifest audio subset to S3 (directory-wise sync)."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument(
        "--s3-audio-root",
        type=str,
        required=True,
        help="S3 URI root, e.g. s3://my-bucket/birdsong/audio",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-dirs", type=int, default=None)
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="File patterns to include (repeatable). Default: *.wav and *.WAV",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-quiet", action="store_true")
    parser.add_argument("--summary-out", type=Path, default=None)

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")

    manifest = _load_manifest(args.manifest)
    pairs = iter_manifest_entry_pairs(manifest, split=str(args.split))
    pairs = apply_max_dirs(pairs, max_dirs=args.max_dirs)
    pairs = select_shard(pairs, num_shards=int(args.num_shards), shard_index=int(args.shard_index))
    entries = [entry for _, entry in pairs]

    include_patterns = args.include or ["*.wav", "*.WAV"]

    tasks: list[tuple[str, str]] = []
    for entry in entries:
        audio_dir, _ = resolve_manifest_entry_paths(
            entry, audio_root=args.audio_root, roi_root=None
        )
        rel = entry.get("audio_dir_rel")
        if rel is None:
            raise ValueError("Manifest entry missing audio_dir_rel (required for S3 layout).")
        target = _join_s3_uri(args.s3_audio_root, str(rel))
        tasks.append((audio_dir, target))

    if not tasks:
        raise ValueError("No manifest entries selected.")

    if args.dry_run:
        print(f"Planned sync tasks: {len(tasks)} (shard={args.shard_index}/{args.num_shards})")
        for audio_dir, target in tasks[:10]:
            print(f"  {audio_dir} -> {target}")
        return

    aws = _require_aws_cli()

    start = time.time()
    results = []
    failures = 0
    quiet = not bool(args.no_quiet)
    with ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
        futures = [
            pool.submit(
                _sync_one,
                aws,
                audio_dir,
                target,
                include_patterns=include_patterns,
                quiet=quiet,
            )
            for audio_dir, target in tasks
        ]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.get("status") != "ok":
                failures += 1

    elapsed = max(1e-9, time.time() - start)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": args.manifest.as_posix(),
        "split": args.split,
        "audio_root": args.audio_root.as_posix() if args.audio_root else None,
        "s3_audio_root": str(args.s3_audio_root),
        "jobs": int(args.jobs),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "tasks": int(len(tasks)),
        "failures": int(failures),
        "elapsed_sec": float(elapsed),
        "tasks_per_sec": float(len(tasks) / elapsed),
        "errors": [r for r in results if r.get("status") != "ok"],
    }

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
