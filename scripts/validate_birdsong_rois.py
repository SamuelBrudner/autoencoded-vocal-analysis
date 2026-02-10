#!/usr/bin/env python3
"""Validate ROI directory layout for birdsong workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _load_manifest(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _resolve_entry_paths(
	entry: dict,
	audio_root: Optional[Path],
	roi_root: Optional[Path],
) -> tuple[str, str]:
	audio_dir = entry.get("audio_dir")
	roi_dir = entry.get("roi_dir")
	if not audio_dir:
		rel = entry.get("audio_dir_rel")
		if rel is None or audio_root is None:
			raise ValueError("Manifest entry missing audio_dir and audio_root.")
		audio_dir = (audio_root if rel in (".", "") else audio_root / rel).as_posix()
	if not roi_dir:
		rel = entry.get("audio_dir_rel")
		if rel is None or roi_root is None:
			raise ValueError("Manifest entry missing roi_dir and roi_root.")
		roi_dir = (roi_root if rel in (".", "") else roi_root / rel).as_posix()
	return audio_dir, roi_dir


def _list_wavs(audio_dir: str) -> list[str]:
	try:
		filenames = os.listdir(audio_dir)
	except FileNotFoundError:
		return []
	return [
		os.path.join(audio_dir, name)
		for name in sorted(filenames)
		if name.lower().endswith(".wav")
	]


def _roi_has_data(path: str) -> bool:
	try:
		with open(path, "r", encoding="utf-8") as handle:
			for line in handle:
				stripped = line.strip()
				if stripped and not stripped.startswith("#"):
					return True
	except FileNotFoundError:
		return False
	return False


def _select_entries(manifest: dict, split: str) -> list[dict]:
	if split == "train":
		return list(manifest.get("train", []))
	if split == "test":
		return list(manifest.get("test", []))
	entries = []
	entries.extend(manifest.get("train", []))
	entries.extend(manifest.get("test", []))
	return entries


def validate_rois(
	entries: list[dict],
	audio_root: Optional[Path] = None,
	roi_root: Optional[Path] = None,
	max_dirs: Optional[int] = None,
	max_files_per_dir: Optional[int] = None,
	check_empty: bool = True,
) -> dict:
	missing_roi_dirs = 0
	missing_roi_files = 0
	empty_roi_files = 0
	total_wavs = 0
	total_dirs = 0
	per_dir = []

	for entry in entries[: max_dirs]:
		audio_dir, roi_dir = _resolve_entry_paths(entry, audio_root, roi_root)
		wavs = _list_wavs(audio_dir)
		if max_files_per_dir is not None:
			wavs = wavs[: int(max_files_per_dir)]
		total_dirs += 1
		total_wavs += len(wavs)

		if not os.path.isdir(roi_dir):
			missing_roi_dirs += 1
			missing_roi_files += len(wavs)
			per_dir.append(
				{
					"audio_dir": audio_dir,
					"roi_dir": roi_dir,
					"wav_files": len(wavs),
					"missing_roi_dir": True,
					"missing_roi_files": len(wavs),
					"empty_roi_files": 0,
				}
			)
			continue

		dir_missing = 0
		dir_empty = 0
		for wav in wavs:
			roi_path = os.path.join(roi_dir, f"{Path(wav).stem}.txt")
			if not os.path.exists(roi_path):
				dir_missing += 1
				continue
			if check_empty and not _roi_has_data(roi_path):
				dir_empty += 1
		missing_roi_files += dir_missing
		empty_roi_files += dir_empty
		per_dir.append(
			{
				"audio_dir": audio_dir,
				"roi_dir": roi_dir,
				"wav_files": len(wavs),
				"missing_roi_dir": False,
				"missing_roi_files": dir_missing,
				"empty_roi_files": dir_empty,
			}
		)

	return {
		"directories_checked": int(total_dirs),
		"wav_files_checked": int(total_wavs),
		"missing_roi_dirs": int(missing_roi_dirs),
		"missing_roi_files": int(missing_roi_files),
		"empty_roi_files": int(empty_roi_files),
		"per_directory": per_dir,
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Validate ROI directory layout (missing/empty ROI files)."
	)
	parser.add_argument("--manifest", type=Path, required=True)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--audio-root", type=Path, default=None)
	parser.add_argument("--roi-root", type=Path, default=None)
	parser.add_argument("--max-dirs", type=int, default=None)
	parser.add_argument("--max-files-per-dir", type=int, default=None)
	parser.add_argument("--no-check-empty", action="store_true")
	parser.add_argument(
		"--fail-on-empty",
		action="store_true",
		help="Exit non-zero when empty ROI files are detected.",
	)
	parser.add_argument("--output", type=Path, default=None)

	args = parser.parse_args()

	manifest = _load_manifest(args.manifest)
	entries = _select_entries(manifest, args.split)
	if not entries:
		raise ValueError("No manifest entries found for the requested split.")

	report = validate_rois(
		entries,
		audio_root=args.audio_root,
		roi_root=args.roi_root,
		max_dirs=args.max_dirs,
		max_files_per_dir=args.max_files_per_dir,
		check_empty=not args.no_check_empty,
	)
	report.update(
		{
			"manifest": args.manifest.as_posix(),
			"split": args.split,
		}
	)
	payload = json.dumps(report, indent=2, sort_keys=False)
	print(payload)
	if args.output is not None:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(payload, encoding="utf-8")

	if report["missing_roi_dirs"] or report["missing_roi_files"]:
		sys.exit(2)
	if args.fail_on_empty and report["empty_roi_files"]:
		sys.exit(3)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)

