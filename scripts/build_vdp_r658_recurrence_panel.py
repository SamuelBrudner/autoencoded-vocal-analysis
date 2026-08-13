#!/usr/bin/env python3
"""Build the fixed R658 developmental recurrence panel from the shared cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _explicit_dph(value: object) -> Optional[int]:
	if value is None:
		return None
	try:
		dph = float(value)
	except (TypeError, ValueError):
		return None
	if not math.isfinite(dph) or dph < 0 or not dph.is_integer():
		return None
	return int(dph)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			if not line.strip():
				continue
			row = json.loads(line)
			for key in ("split", "bird_id", "audio_dir_rel", "filename"):
				if key not in row:
					raise ValueError(f"Member line {line_number} is missing {key!r}.")
			rows.append(row)
	if not rows:
		raise ValueError("Source member manifest is empty.")
	return rows


def _selection_key(row: dict[str, Any], *, seed: int) -> str:
	payload = "\0".join(
		[
			str(seed),
			"r658_developmental_recurrence",
			str(row["dph"]),
			str(row["split"]),
			str(row["audio_dir_rel"]),
			str(row["filename"]),
		]
	)
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tutor_start_by_dir(
	metadata_path: Path, directories: set[str]
) -> dict[str, Optional[float]]:
	try:
		import pyarrow.dataset as ds  # type: ignore
	except ImportError as exc:  # pragma: no cover
		raise ImportError("Panel construction requires pyarrow.") from exc
	dataset = ds.dataset(metadata_path, format="parquet")
	required = {"audio_dir_rel", "tutor_start_day"}
	missing = required - set(dataset.schema.names)
	if missing:
		raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
	table = dataset.to_table(
		columns=["audio_dir_rel", "tutor_start_day"],
		filter=ds.field("audio_dir_rel").isin(sorted(directories)),
	)
	values: dict[str, set[float]] = defaultdict(set)
	seen: set[str] = set()
	for row in table.to_pylist():
		rel = str(row["audio_dir_rel"])
		seen.add(rel)
		value = row.get("tutor_start_day")
		if value is not None:
			values[rel].add(float(value))
	missing_dirs = directories - seen
	if missing_dirs:
		raise ValueError(f"Metadata contains no rows for {sorted(missing_dirs)[0]!r}.")
	result: dict[str, Optional[float]] = {}
	for rel in directories:
		found = values.get(rel, set())
		if len(found) > 1:
			raise ValueError(f"Directory {rel!r} has conflicting tutor-start values.")
		result[rel] = next(iter(found)) if found else None
	return result


def build_panel(
	*,
	source_manifest_path: Path,
	source_members_path: Path,
	metadata_path: Path,
	bird_id: str = "R658",
	max_clips_per_day: int = 6,
	baseline_max_dph: int = 35,
	seed: int = 20260813,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if max_clips_per_day <= 0:
		raise ValueError("max_clips_per_day must be positive.")
	source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
	entries: dict[tuple[str, str], dict[str, Any]] = {}
	for split in ("train", "test"):
		for entry in source_manifest.get(split, []):
			key = (split, str(entry["audio_dir_rel"]))
			if key in entries:
				raise ValueError(f"Duplicate source manifest entry {key}.")
			entries[key] = entry

	by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
	for member in _load_jsonl(source_members_path):
		key = (str(member["split"]), str(member["audio_dir_rel"]))
		entry = entries.get(key)
		if entry is None:
			raise ValueError(f"Source member has no matching manifest entry: {key}.")
		entry_bird = str(entry.get("bird_id_norm") or member["bird_id"])
		if entry_bird != bird_id:
			continue
		dph = _explicit_dph(entry.get("dph"))
		if dph is None:
			raise ValueError(f"R658 member {member['filename']!r} lacks explicit dph.")
		by_day[dph].append(
			dict(
				member,
				bird_id=bird_id,
				dph=dph,
				regime=entry.get("regime"),
				panel="r658_developmental_recurrence",
			)
		)
	if not by_day:
		raise ValueError(f"No source members found for {bird_id}.")

	selected: list[dict[str, Any]] = []
	for dph in sorted(by_day):
		ordered = sorted(by_day[dph], key=lambda row: _selection_key(row, seed=seed))
		selected.extend(ordered[:max_clips_per_day])

	directories = {str(row["audio_dir_rel"]) for row in selected}
	tutor_by_dir = _tutor_start_by_dir(metadata_path, directories)
	for row in selected:
		row["recording_id"] = None
		row["tutor_start_dph"] = tutor_by_dir[str(row["audio_dir_rel"])]

	counts = Counter((str(row["split"]), str(row["audio_dir_rel"])) for row in selected)
	panel_entries: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
	for key in sorted(counts):
		entry = dict(entries[key])
		entry.pop("audio_dir", None)
		entry.pop("roi_dir", None)
		entry["num_files"] = counts[key]
		entry["recording_id"] = None
		entry["tutor_start_dph"] = tutor_by_dir[str(entry["audio_dir_rel"])]
		entry["panel"] = "r658_developmental_recurrence"
		panel_entries[key[0]].append(entry)

	per_day = Counter(int(row["dph"]) for row in selected)
	baseline_members = sum(count for dph, count in per_day.items() if dph <= baseline_max_dph)
	if baseline_members == 0:
		raise ValueError(f"No explicit baseline members at or below {baseline_max_dph} dph.")
	manifest = {
		"schema_version": "vdp_r658_recurrence_panel_v1",
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"source_manifest_sha256": _sha256_file(source_manifest_path),
		"source_member_manifest_sha256": _sha256_file(source_members_path),
		"metadata_sha256": _sha256_file(metadata_path),
		"split_semantics": "descriptive_no_holdout",
		"selection": {
			"bird_id": bird_id,
			"seed": seed,
			"rule": "stable hash-ranked sample capped independently within each explicit dph",
			"max_clips_per_day": max_clips_per_day,
			"baseline_max_dph": baseline_max_dph,
		},
		"summary": {
			"bird_id": bird_id,
			"members": len(selected),
			"days": len(per_day),
			"minimum_dph": min(per_day),
			"maximum_dph": max(per_day),
			"baseline_members": baseline_members,
			"members_per_dph": {str(day): per_day[day] for day in sorted(per_day)},
		},
		"train": panel_entries["train"],
		"test": panel_entries["test"],
	}
	return manifest, selected


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-manifest", type=Path, required=True)
	parser.add_argument("--source-members", type=Path, required=True)
	parser.add_argument("--metadata", type=Path, required=True)
	parser.add_argument("--out-manifest", type=Path, required=True)
	parser.add_argument("--out-members", type=Path, required=True)
	parser.add_argument("--bird", default="R658")
	parser.add_argument("--max-clips-per-day", type=int, default=6)
	parser.add_argument("--baseline-max-dph", type=int, default=35)
	parser.add_argument("--seed", type=int, default=20260813)
	args = parser.parse_args()
	manifest, members = build_panel(
		source_manifest_path=args.source_manifest,
		source_members_path=args.source_members,
		metadata_path=args.metadata,
		bird_id=args.bird,
		max_clips_per_day=args.max_clips_per_day,
		baseline_max_dph=args.baseline_max_dph,
		seed=args.seed,
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
				"days": manifest["summary"]["days"],
				"members": manifest["summary"]["members"],
				"baseline_members": manifest["summary"]["baseline_members"],
				"manifest_sha256": _sha256_file(args.out_manifest),
				"members_sha256": _sha256_file(args.out_members),
			},
			sort_keys=True,
		)
	)


if __name__ == "__main__":
	main()
