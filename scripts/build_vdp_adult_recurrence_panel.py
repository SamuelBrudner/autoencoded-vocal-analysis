#!/usr/bin/env python3
"""Build the fixed AVN adult panel for the VDP recurrence positive control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _required_text(row: dict[str, str], key: str, *, line_number: int) -> str:
	value = str(row.get(key) or "").strip()
	if not value:
		raise ValueError(f"Manifest line {line_number} has no {key!r} value.")
	return value


def _explicit_dph(row: dict[str, str], *, line_number: int) -> Optional[float]:
	value = str(row.get("age_dph") or "").strip()
	if not value:
		return None
	try:
		dph = float(value)
	except ValueError as exc:
		raise ValueError(
			f"Manifest line {line_number} has invalid age_dph {value!r}."
		) from exc
	if not math.isfinite(dph) or dph < 0:
		raise ValueError(
			f"Manifest line {line_number} has invalid age_dph {value!r}."
		)
	return dph


def _portable_clip(
	value: str,
	*,
	manifest_dir: Path,
	audio_root: Path,
	line_number: int,
) -> tuple[str, str]:
	path = Path(value)
	resolved = (manifest_dir / path).resolve() if not path.is_absolute() else path.resolve()
	try:
		relative = resolved.relative_to(audio_root.resolve())
	except ValueError as exc:
		raise ValueError(
			f"Manifest line {line_number} clip is outside the declared audio root."
		) from exc
	if not resolved.is_file():
		raise FileNotFoundError(f"Manifest line {line_number} clip does not exist: {resolved}")
	return relative.parent.as_posix(), relative.name


def build_panel(
	*,
	source_manifest_path: Path,
	audio_root: Path,
	clips_per_bird: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	"""Return a portable AVA manifest and exact member allowlist.

	The lexicographically first unique clips per bird reproduce the historical
	trajectory-control panel rule while keeping selection independent of the new
	encoder's outputs.
	"""
	if clips_per_bird <= 0:
		raise ValueError("clips_per_bird must be positive.")
	if not audio_root.is_dir():
		raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

	candidates: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
	with source_manifest_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		required = {
			"bird_id",
			"age_dph",
			"experimental_condition",
			"clip_a",
			"clip_b",
			"clip_a_id",
			"clip_b_id",
		}
		missing = required - set(reader.fieldnames or ())
		if missing:
			raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
		for line_number, row in enumerate(reader, start=2):
			bird = _required_text(row, "bird_id", line_number=line_number)
			dph = _explicit_dph(row, line_number=line_number)
			regime = str(row.get("experimental_condition") or "").strip() or None
			for clip_column, id_column in (
				("clip_a", "clip_a_id"),
				("clip_b", "clip_b_id"),
			):
				rel, filename = _portable_clip(
					_required_text(row, clip_column, line_number=line_number),
					manifest_dir=source_manifest_path.parent,
					audio_root=audio_root,
					line_number=line_number,
				)
				recording_id = _required_text(row, id_column, line_number=line_number)
				key = (rel, filename)
				candidate = {
					"split": "test",
					"bird_id": bird,
					"audio_dir_rel": rel,
					"filename": filename,
					"recording_id": recording_id,
					"dph": dph,
					"regime": regime,
					"tutor_start_dph": None,
					"panel": "adult_positive_control",
				}
				previous = candidates[bird].get(key)
				if previous is not None and previous != candidate:
					raise ValueError(f"Conflicting metadata for {rel}/{filename}.")
				candidates[bird][key] = candidate

	if not candidates:
		raise ValueError("Source manifest contains no clips.")

	selected: list[dict[str, Any]] = []
	for bird in sorted(candidates):
		bird_rows = sorted(
			candidates[bird].values(),
			key=lambda row: (str(row["audio_dir_rel"]), str(row["filename"])),
		)
		if len(bird_rows) < clips_per_bird:
			raise ValueError(
				f"Bird {bird} has {len(bird_rows)} unique clips; {clips_per_bird} required."
			)
		selected.extend(bird_rows[:clips_per_bird])

	by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in selected:
		by_dir[str(row["audio_dir_rel"])].append(row)

	entries: list[dict[str, Any]] = []
	for audio_dir_rel in sorted(by_dir):
		rows = by_dir[audio_dir_rel]
		metadata = {
			(
				str(row["bird_id"]),
				None if row["dph"] is None else float(row["dph"]),
				None if row["regime"] is None else str(row["regime"]),
			)
			for row in rows
		}
		if len(metadata) != 1:
			raise ValueError(f"Directory {audio_dir_rel!r} has conflicting metadata.")
		bird, dph, regime = next(iter(metadata))
		entries.append(
			{
				"audio_dir_rel": audio_dir_rel,
				"bird_id_norm": bird,
				"bird_id_raw": bird,
				"regime": regime,
				"dph": dph,
				"num_files": len(rows),
				"split": "test",
				"recording_id": None,
				"tutor_start_dph": None,
				"panel": "adult_positive_control",
			}
		)

	manifest = {
		"schema_version": "vdp_adult_recurrence_panel_v1",
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"source_manifest_sha256": _sha256_file(source_manifest_path),
		"selection": {
			"rule": "lexicographically first unique source-manifest clips per bird",
			"clips_per_bird": clips_per_bird,
		},
		"summary": {
			"birds": len(candidates),
			"members": len(selected),
			"panel": "adult_positive_control",
			"birds_with_null_dph": sorted(
				{str(row["bird_id"]) for row in selected if row["dph"] is None}
			),
			"birds_with_null_regime": sorted(
				{str(row["bird_id"]) for row in selected if row["regime"] is None}
			),
		},
		"train": [],
		"test": entries,
	}
	return manifest, selected


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-manifest", type=Path, required=True)
	parser.add_argument("--audio-root", type=Path, required=True)
	parser.add_argument("--out-manifest", type=Path, required=True)
	parser.add_argument("--out-members", type=Path, required=True)
	parser.add_argument("--clips-per-bird", type=int, default=6)
	args = parser.parse_args()

	manifest, members = build_panel(
		source_manifest_path=args.source_manifest,
		audio_root=args.audio_root,
		clips_per_bird=args.clips_per_bird,
	)
	args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
	args.out_members.parent.mkdir(parents=True, exist_ok=True)
	args.out_manifest.write_text(
		json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	args.out_members.write_text(
		"".join(json.dumps(row, sort_keys=True) + "\n" for row in members),
		encoding="utf-8",
	)
	print(
		json.dumps(
			{
				"birds": manifest["summary"]["birds"],
				"members": manifest["summary"]["members"],
				"manifest_sha256": _sha256_file(args.out_manifest),
				"members_sha256": _sha256_file(args.out_members),
			},
			sort_keys=True,
		)
	)


if __name__ == "__main__":
	main()
