#!/usr/bin/env python3
"""Validate and identify a complete ``ava_latent_sequence_v1`` collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.models.latent_sequence_contract import (  # noqa: E402
	validate_latent_sequence_pair,
)


PORTABLE_PATH_FIELDS = (
	"audio_path",
	"roi_path",
	"config_path",
	"checkpoint_path",
	"manifest_path",
)


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
	value = str(value).lower()
	if re.fullmatch(r"[0-9a-f]{64}", value) is None:
		raise ValueError(f"{name} must be a full lowercase SHA-256 value.")
	return value


def validate_collection(
	root: Path,
	*,
	dataset_sha256: str,
	configuration_sha256: str,
	checkpoint_sha256: str,
	code_sha256: str,
	expected_clips: Optional[int] = None,
	require_energy: bool = False,
) -> tuple[list[dict], dict, dict]:
	root = root.resolve()
	npz_paths = sorted(root.rglob("*.npz"))
	json_paths = sorted(root.rglob("*.json"))
	if not npz_paths:
		raise ValueError(f"No NPZ exports found under {root}.")
	if expected_clips is not None and len(npz_paths) != int(expected_clips):
		raise ValueError(
			f"Expected {expected_clips} NPZ exports; found {len(npz_paths)}."
		)
	npz_stems = {path.relative_to(root).with_suffix("") for path in npz_paths}
	json_stems = {path.relative_to(root).with_suffix("") for path in json_paths}
	if npz_stems != json_stems:
		missing_json = sorted(npz_stems - json_stems)
		orphan_json = sorted(json_stems - npz_stems)
		raise ValueError(
			"Collection has unmatched pairs: "
			f"missing_json={len(missing_json)} orphan_json={len(orphan_json)}."
		)

	members = []
	total_windows = 0
	panels: dict[str, int] = {}
	for npz_path in npz_paths:
		artifact = validate_latent_sequence_pair(npz_path)
		if require_energy and artifact.energy is None:
			raise ValueError(f"Required energy array is absent from {npz_path}.")
		metadata = artifact.metadata
		for field in PORTABLE_PATH_FIELDS:
			value = metadata.get(field)
			if isinstance(value, str) and Path(value).is_absolute():
				raise ValueError(
					f"Metadata field {field!r} must be portable in {npz_path}."
				)
		entry = metadata.get("entry") if isinstance(metadata.get("entry"), dict) else {}
		panel = entry.get("panel")
		if panel is not None:
			panels[str(panel)] = panels.get(str(panel), 0) + 1
		npz_rel = npz_path.relative_to(root).as_posix()
		json_path = npz_path.with_suffix(".json")
		member = {
			"clip_id": metadata["clip_id"],
			"panel": panel,
			"bird_id": metadata["bird_id"],
			"dph": metadata["dph"],
			"regime": metadata["regime"],
			"tutor_start_dph": metadata["tutor_start_dph"],
			"recording_id": metadata["recording_id"],
			"audio_sha256": metadata.get("audio_sha256"),
			"npz_path": npz_rel,
			"npz_sha256": _sha256_file(npz_path),
			"json_path": json_path.relative_to(root).as_posix(),
			"json_sha256": _sha256_file(json_path),
			"windows": int(artifact.mu.shape[0]),
			"latent_dimensions": int(artifact.mu.shape[1]),
		}
		members.append(member)
		total_windows += member["windows"]

	lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in members]
	member_bytes = ("\n".join(lines) + "\n").encode("utf-8")
	member_manifest_sha256 = hashlib.sha256(member_bytes).hexdigest()
	created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
	acceptance = {
		"schema_version": "ava_latent_sequence_export_acceptance_v1",
		"created_utc": created_utc,
		"dataset_sha256": _require_sha256(dataset_sha256, "dataset_sha256"),
		"configuration_sha256": _require_sha256(
			configuration_sha256, "configuration_sha256"
		),
		"checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint_sha256"),
		"code_sha256": _require_sha256(code_sha256, "code_sha256"),
		"member_manifest_sha256": member_manifest_sha256,
		"schema_conformance": "pass",
		"clip_count": len(members),
		"total_windows": total_windows,
	}
	summary = {
		"created_utc": created_utc,
		"schema_version": "ava_latent_sequence_v1",
		"ok": True,
		"clip_count": len(members),
		"total_windows": total_windows,
		"panels": dict(sorted(panels.items())),
		"same_stem_pairs": len(members),
		"finite_and_dtype_conformant": len(members),
		"member_manifest_sha256": member_manifest_sha256,
	}
	return members, acceptance, summary


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--root", type=Path, required=True)
	parser.add_argument("--dataset-sha256", required=True)
	parser.add_argument("--configuration-sha256", required=True)
	parser.add_argument("--checkpoint-sha256", required=True)
	parser.add_argument("--code-sha256", required=True)
	parser.add_argument("--expected-clips", type=int, default=None)
	parser.add_argument("--require-energy", action="store_true")
	parser.add_argument("--members-out", type=Path, required=True)
	parser.add_argument("--acceptance-out", type=Path, required=True)
	parser.add_argument("--summary-out", type=Path, required=True)
	args = parser.parse_args()

	members, acceptance, summary = validate_collection(
		args.root,
		dataset_sha256=args.dataset_sha256,
		configuration_sha256=args.configuration_sha256,
		checkpoint_sha256=args.checkpoint_sha256,
		code_sha256=args.code_sha256,
		expected_clips=args.expected_clips,
		require_energy=args.require_energy,
	)
	for path in (args.members_out, args.acceptance_out, args.summary_out):
		path.parent.mkdir(parents=True, exist_ok=True)
	with args.members_out.open("w", encoding="utf-8") as handle:
		for member in members:
			handle.write(json.dumps(member, sort_keys=True, separators=(",", ":")) + "\n")
	args.acceptance_out.write_text(
		json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	args.summary_out.write_text(
		json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
	main()
