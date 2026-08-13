#!/usr/bin/env python3
"""Build the deterministic early/later/adult latent-handoff canary panel."""

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


def _coerce_dph(value: object) -> Optional[int]:
	if value is None:
		return None
	try:
		dph = float(value)
	except (TypeError, ValueError):
		return None
	if not math.isfinite(dph) or dph < 0 or not dph.is_integer():
		return None
	return int(dph)


def _selection_key(row: dict, *, seed: int, panel: str) -> str:
	payload = "\0".join(
		[
			str(seed),
			panel,
			str(row["split"]),
			str(row["audio_dir_rel"]),
			str(row["filename"]),
		]
	)
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
	rows = []
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


def _tutor_start_by_dir(metadata_path: Path, directories: set[str]) -> dict[str, Optional[float]]:
	try:
		import pyarrow.dataset as ds  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
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
	seen_dirs = set()
	for row in table.to_pylist():
		rel = str(row["audio_dir_rel"])
		seen_dirs.add(rel)
		value = row.get("tutor_start_day")
		if value is not None:
			values[rel].add(float(value))
	missing_dirs = sorted(directories - seen_dirs)
	if missing_dirs:
		raise ValueError(
			f"Metadata contains no rows for {len(missing_dirs)} selected directories; "
			f"first is {missing_dirs[0]!r}."
		)
	result: dict[str, Optional[float]] = {}
	for rel in directories:
		found = values.get(rel, set())
		if len(found) > 1:
			raise ValueError(f"Directory {rel!r} has conflicting tutor-start values.")
		result[rel] = next(iter(found)) if found else None
	return result


def _select_exact(
	rows: list[dict],
	*,
	panel: str,
	count: int,
	seed: int,
	context: str,
) -> list[dict]:
	if len(rows) < count:
		raise ValueError(f"{context} has {len(rows)} members; {count} are required.")
	ordered = sorted(rows, key=lambda row: _selection_key(row, seed=seed, panel=panel))
	return [dict(row, panel=panel) for row in ordered[:count]]


def build_panel(
	*,
	source_manifest_path: Path,
	source_members_path: Path,
	metadata_path: Path,
	early_dph: tuple[int, ...] = (33, 34, 35),
	later_dph: tuple[int, ...] = (60, 61, 62),
	r658_bird: str = "R658",
	clips_per_r658_dph: int = 2,
	adult_min_dph: int = 90,
	adult_clips_per_bird: int = 1,
	seed: int = 20260812,
) -> tuple[dict, list[dict]]:
	if clips_per_r658_dph <= 0 or adult_clips_per_bird <= 0:
		raise ValueError("Clip counts must be positive.")
	source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
	entry_map: dict[tuple[str, str], dict] = {}
	for split in ("train", "test"):
		for entry in source_manifest.get(split, []):
			key = (split, str(entry["audio_dir_rel"]))
			if key in entry_map:
				raise ValueError(f"Duplicate source manifest entry {key}.")
			entry_map[key] = entry

	members = _load_jsonl(source_members_path)
	joined = []
	for row in members:
		key = (str(row["split"]), str(row["audio_dir_rel"]))
		if key not in entry_map:
			raise ValueError(f"Source member has no matching manifest entry: {key}.")
		entry = entry_map[key]
		bird = str(entry.get("bird_id_norm") or row["bird_id"])
		dph = _coerce_dph(entry.get("dph"))
		joined.append(dict(row, bird_id=bird, dph=dph))

	selected: list[dict] = []
	for panel, ages in (("early_r658", early_dph), ("later_r658", later_dph)):
		for dph in ages:
			candidates = [
				row
				for row in joined
				if row["bird_id"] == r658_bird and row["dph"] == int(dph)
			]
			selected.extend(
				_select_exact(
					candidates,
					panel=panel,
					count=clips_per_r658_dph,
					seed=seed,
					context=f"{panel} DPH {dph}",
				)
			)

	adult_by_bird: dict[str, list[dict]] = defaultdict(list)
	for row in joined:
		if row["split"] == "test" and row["dph"] is not None and row["dph"] >= adult_min_dph:
			adult_by_bird[row["bird_id"]].append(row)
	if not adult_by_bird:
		raise ValueError("No held-out adult members meet the requested age threshold.")
	for bird in sorted(adult_by_bird):
		bird_rows = adult_by_bird[bird]
		earliest_dph = min(int(row["dph"]) for row in bird_rows)
		candidates = [row for row in bird_rows if int(row["dph"]) == earliest_dph]
		selected.extend(
			_select_exact(
				candidates,
				panel="heldout_adult",
				count=adult_clips_per_bird,
				seed=seed,
				context=f"heldout_adult {bird} DPH {earliest_dph}",
			)
		)

	selected.sort(
		key=lambda row: (
			str(row["panel"]),
			str(row["bird_id"]),
			int(row["dph"]),
			str(row["audio_dir_rel"]),
			str(row["filename"]),
		)
	)
	directories = {str(row["audio_dir_rel"]) for row in selected}
	tutor_by_dir = _tutor_start_by_dir(metadata_path, directories)
	for row in selected:
		row["recording_id"] = None
		row["tutor_start_dph"] = tutor_by_dir[str(row["audio_dir_rel"])]

	counts = Counter((row["split"], row["audio_dir_rel"]) for row in selected)
	panel_by_dir = {
		(row["split"], row["audio_dir_rel"]): row["panel"] for row in selected
	}
	train_entries = []
	test_entries = []
	for key in sorted(counts):
		entry = dict(entry_map[key])
		for field in ("audio_dir", "roi_dir"):
			entry.pop(field, None)
		entry["recording_id"] = None
		entry["tutor_start_dph"] = tutor_by_dir[str(entry["audio_dir_rel"])]
		entry["panel"] = panel_by_dir[key]
		entry["num_files"] = counts[key]
		(train_entries if key[0] == "train" else test_entries).append(entry)

	panel_counts = Counter(row["panel"] for row in selected)
	manifest = {
		"schema_version": "vdp_latent_handoff_panel_v1",
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"source_manifest_sha256": _sha256_file(source_manifest_path),
		"source_member_manifest_sha256": _sha256_file(source_members_path),
		"metadata_sha256": _sha256_file(metadata_path),
		"selection": {
			"seed": int(seed),
			"r658_bird": r658_bird,
			"early_dph": list(early_dph),
			"later_dph": list(later_dph),
			"clips_per_r658_dph": int(clips_per_r658_dph),
			"adult_min_dph": int(adult_min_dph),
			"adult_clips_per_bird": int(adult_clips_per_bird),
			"adult_rule": "earliest available dph at or above threshold per held-out bird",
		},
		"summary": {
			"members": len(selected),
			"panels": dict(sorted(panel_counts.items())),
			"adult_birds": len(adult_by_bird),
		},
		"train": train_entries,
		"test": test_entries,
	}
	return manifest, selected


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source-manifest", type=Path, required=True)
	parser.add_argument("--source-members", type=Path, required=True)
	parser.add_argument("--metadata", type=Path, required=True)
	parser.add_argument("--out-manifest", type=Path, required=True)
	parser.add_argument("--out-members", type=Path, required=True)
	parser.add_argument("--seed", type=int, default=20260812)
	args = parser.parse_args()

	manifest, members = build_panel(
		source_manifest_path=args.source_manifest,
		source_members_path=args.source_members,
		metadata_path=args.metadata,
		seed=args.seed,
	)
	args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
	args.out_members.parent.mkdir(parents=True, exist_ok=True)
	args.out_manifest.write_text(
		json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	with args.out_members.open("w", encoding="utf-8") as handle:
		for row in members:
			handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
	print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
	main()
