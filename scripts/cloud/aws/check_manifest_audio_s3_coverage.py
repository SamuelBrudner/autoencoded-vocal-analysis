#!/usr/bin/env python3
"""Check that manifest audio directories have WAV coverage in S3."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.cloud.manifest_s3_coverage import (  # noqa: E402
	build_manifest_audio_prefixes,
	parse_aws_s3_ls_key,
	summarize_manifest_audio_coverage,
)


def _load_manifest(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _iter_keys_from_listing_file(path: Path) -> Iterable[str]:
	with open(path, "r", encoding="utf-8") as handle:
		for line in handle:
			key = parse_aws_s3_ls_key(line)
			if key is not None:
				yield key


def _iter_keys_from_s3(s3_audio_root: str, aws_profile: str | None) -> Iterable[str]:
	aws = shutil.which("aws")
	if not aws:
		raise RuntimeError("aws CLI is required but was not found on PATH.")
	cmd = [aws]
	if aws_profile:
		cmd.extend(["--profile", str(aws_profile)])
	cmd.extend(["s3", "ls", str(s3_audio_root).rstrip("/") + "/", "--recursive"])
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	assert proc.stdout is not None
	for line in proc.stdout:
		key = parse_aws_s3_ls_key(line)
		if key is not None:
			yield key
	stderr = ""
	if proc.stderr is not None:
		stderr = proc.stderr.read()
	code = proc.wait()
	if code != 0:
		raise RuntimeError(
			f"aws s3 ls failed with exit code {code}: {stderr.strip()[-2000:]}"
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Check S3 WAV coverage for every manifest audio_dir_rel entry."
	)
	parser.add_argument("--manifest", type=Path, required=True)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--s3-audio-root", type=str, required=True)
	parser.add_argument("--max-dirs", type=int, default=None)
	parser.add_argument("--listing-file", type=Path, default=None)
	parser.add_argument("--aws-profile", type=str, default=None)
	parser.add_argument("--out", type=Path, default=None)
	parser.add_argument("--fail-on-missing", action="store_true")
	parser.add_argument("--fail-on-count-mismatch", action="store_true")
	args = parser.parse_args()

	start = time.time()
	manifest = _load_manifest(args.manifest)
	prefixes = build_manifest_audio_prefixes(
		manifest,
		split=str(args.split),
		s3_audio_root=str(args.s3_audio_root),
		max_dirs=args.max_dirs,
	)
	keys = (
		_iter_keys_from_listing_file(args.listing_file)
		if args.listing_file is not None
		else _iter_keys_from_s3(args.s3_audio_root, aws_profile=args.aws_profile)
	)
	summary = summarize_manifest_audio_coverage(prefixes, keys)
	summary.update(
		{
			"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
			"manifest_path": args.manifest.as_posix(),
			"split": str(args.split),
			"s3_audio_root": str(args.s3_audio_root).rstrip("/"),
			"listing_file": args.listing_file.as_posix() if args.listing_file else None,
			"elapsed_sec": float(time.time() - start),
		}
	)

	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

	print(
		json.dumps(
			{
				"status": summary["status"],
				"manifest_entries": summary["manifest_entries"],
				"observed_dirs": summary["observed_dirs"],
				"missing_dirs": summary["missing_dirs"],
				"count_mismatch_dirs": summary["count_mismatch_dirs"],
				"expected_wav_files": summary["expected_wav_files"],
				"observed_wav_files": summary["observed_wav_files"],
				"unmatched_wav_keys": summary["unmatched_wav_keys"],
				"out": args.out.as_posix() if args.out else None,
			},
			indent=2,
			sort_keys=True,
		)
	)

	if args.fail_on_missing and summary["missing_dirs"]:
		sys.exit(1)
	if args.fail_on_count_mismatch and summary["count_mismatch_dirs"]:
		sys.exit(1)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
