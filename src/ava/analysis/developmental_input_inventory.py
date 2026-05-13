"""Inventory local inputs for developmental branch-commitment replication."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from ava.analysis.developmental_replication import DEFAULT_BIRD_IDS
from ava.analysis.hyperbolic_development import write_json


DEFAULT_AUDIO_ROOT = Path("/Volumes/samsung_ssd/data/birdsong")
DEFAULT_ROI_ROOT = Path("/Volumes/samsung_ssd/data/ava_roi_birdsong_full")
DEFAULT_PK249_ROI_ROOT = Path("/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/roi")
DEFAULT_PK249_LATENT_ROOT = Path(
	"/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/latent_sequences"
)
DEFAULT_INVENTORY_BEAD = "autoencoded-vocal-analysis-obi.4.1"


def load_manifest(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def build_developmental_input_inventory(
	manifest_path: Path,
	bird_ids: Sequence[str] = DEFAULT_BIRD_IDS,
	dph_min: float = 33,
	dph_max: float = 90,
	audio_root: Optional[Path] = DEFAULT_AUDIO_ROOT,
	roi_root: Optional[Path] = DEFAULT_ROI_ROOT,
	roi_roots: Optional[Sequence[Path]] = None,
	latent_roots: Optional[Sequence[Path]] = None,
	roi_parquet_name: str = "roi.parquet",
	roi_format: str = "auto",
	count_files: bool = True,
) -> dict:
	"""Inventory manifest, local audio, ROI parquet, and latent-sequence coverage."""
	manifest = load_manifest(manifest_path)
	bird_order = {str(bird).strip().upper(): idx for idx, bird in enumerate(bird_ids)}
	latent_roots = list(latent_roots or [])
	if roi_roots is None:
		roi_roots = [roi_root] if roi_root is not None else []
	roi_roots = [Path(root) for root in roi_roots if root is not None]
	rows = []
	for split in ("train", "test"):
		for entry in manifest.get(split, []):
			bird = str(entry.get("bird_id_norm") or entry.get("bird_id_raw") or "").strip().upper()
			if bird not in bird_order:
				continue
			dph = _optional_float(entry.get("dph"))
			if dph is None or dph < float(dph_min) or dph > float(dph_max):
				continue
			audio_dir_rel = str(entry.get("audio_dir_rel") or "")
			row = _inventory_manifest_row(
				entry=entry,
				split=split,
				bird_id=bird,
				dph=dph,
				audio_dir_rel=audio_dir_rel,
				audio_root=audio_root,
				roi_roots=roi_roots,
				latent_roots=latent_roots,
				roi_parquet_name=roi_parquet_name,
				roi_format=roi_format,
				count_files=count_files,
			)
			rows.append(row)
	rows.sort(
		key=lambda row: (
			bird_order[row["bird_id"]],
			float(row["dph"]),
			str(row["audio_dir_rel"]),
		)
	)
	birds = _summarize_birds(rows, [str(bird).strip().upper() for bird in bird_ids])
	return {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"manifest_path": manifest_path.as_posix(),
		"bird_ids": [str(bird).strip().upper() for bird in bird_ids],
		"dph_min": float(dph_min),
		"dph_max": float(dph_max),
		"audio_root": None if audio_root is None else audio_root.as_posix(),
		"roi_roots": [Path(root).as_posix() for root in roi_roots],
		"latent_roots": [Path(root).as_posix() for root in latent_roots],
		"roi_parquet_name": str(roi_parquet_name),
		"roi_format": str(roi_format),
		"count_files": bool(count_files),
		"rows": rows,
		"birds": birds,
		"summary": _overall_summary(birds),
	}


def write_inventory_artifacts(
	inventory: dict,
	out_dir: Path,
	report_path: Path,
) -> dict:
	"""Write inventory JSON, filtered manifest, and markdown report."""
	out_dir.mkdir(parents=True, exist_ok=True)
	cohort_manifest = _filtered_manifest_from_inventory(inventory)
	artifacts = {
		"input_inventory": (out_dir / "developmental_input_inventory.json").as_posix(),
		"cohort_manifest": (out_dir / "developmental_cohort_manifest.json").as_posix(),
	}
	write_json(Path(artifacts["input_inventory"]), inventory)
	write_json(Path(artifacts["cohort_manifest"]), cohort_manifest)
	write_inventory_report(report_path, inventory, artifacts)
	return artifacts


def write_inventory_report(report_path: Path, inventory: dict, artifacts: dict) -> None:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	summary = inventory["summary"]
	lines = [
		"# Developmental Replication Input Inventory",
		"",
		"## Summary",
		"",
		f"- Birds requested: {summary['birds_requested']}.",
		f"- Birds with audio dirs: {summary['birds_with_audio_dirs']} / {summary['birds_requested']}.",
		f"- Birds with ROI artifacts: {summary['birds_with_roi_artifacts']} / {summary['birds_requested']}.",
		f"- Birds with latent sequences: {summary['birds_with_latents']} / {summary['birds_requested']}.",
		f"- Ready for local full rebuild: {summary['birds_ready_for_full_rebuild']} / {summary['birds_requested']}.",
		"",
		"## Bird Status",
		"",
	]
	for bird, payload in inventory["birds"].items():
		lines.append(
			f"- `{bird}`: {payload['status']}; "
			f"manifest files={payload['manifest_num_files']}, "
			f"audio wavs={payload['audio_wav_count']}, "
			f"ROI dirs={payload['roi_artifact_dirs_present']}/{payload['manifest_rows']}, "
			f"latent npz={payload['latent_npz_count']}."
		)
	lines.extend(
		[
			"",
			"## Next Export Blockers",
			"",
			_next_blocker_text(inventory),
			"",
			"## Artifacts",
			"",
		]
	)
	for name, path in artifacts.items():
		lines.append(f"- `{name}`: `{os.path.relpath(path, report_path.parent.as_posix())}`")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inventory_manifest_row(
	entry: dict,
	split: str,
	bird_id: str,
	dph: float,
	audio_dir_rel: str,
	audio_root: Optional[Path],
	roi_roots: Sequence[Path],
	latent_roots: Sequence[Path],
	roi_parquet_name: str,
	roi_format: str,
	count_files: bool,
) -> dict:
	audio_dir = _join_root(audio_root, audio_dir_rel, entry.get("audio_dir"))
	roi = _roi_counts(
		roi_roots=roi_roots,
		audio_dir_rel=audio_dir_rel,
		fallback=entry.get("roi_dir"),
		roi_parquet_name=roi_parquet_name,
		roi_format=roi_format,
		count_files=count_files,
	)
	latent = _latent_counts(latent_roots, audio_dir_rel, count_files=count_files)
	return {
		"split": str(split),
		"bird_id": bird_id,
		"dph": float(dph),
		"audio_dir_rel": audio_dir_rel,
		"manifest_num_files": int(entry.get("num_files") or 0),
		"regime": entry.get("regime"),
		"audio_dir": None if audio_dir is None else audio_dir.as_posix(),
		"audio_dir_exists": bool(audio_dir is not None and audio_dir.exists()),
		"audio_wav_count": _count_suffix(audio_dir, (".wav", ".WAV")) if count_files else None,
		"roi_root": roi["roi_root"],
		"roi_dir": roi["roi_dir"],
		"roi_format_detected": roi["roi_format_detected"],
		"roi_parquet": roi["roi_parquet"],
		"roi_parquet_exists": roi["roi_parquet_exists"],
		"roi_txt_count": roi["roi_txt_count"],
		"roi_available": roi["roi_available"],
		"latent_root": latent["latent_root"],
		"latent_dir": latent["latent_dir"],
		"latent_dir_exists": latent["latent_dir_exists"],
		"latent_npz_count": latent["npz_count"],
		"latent_json_count": latent["json_count"],
	}


def _summarize_birds(rows: Sequence[dict], requested_bird_ids: Sequence[str]) -> dict:
	grouped: dict[str, list[dict]] = defaultdict(list)
	for row in rows:
		grouped[row["bird_id"]].append(row)
	out = {}
	bird_ids = list(dict.fromkeys(str(bird).strip().upper() for bird in requested_bird_ids))
	for bird in sorted(set(grouped).difference(bird_ids)):
		bird_ids.append(bird)
	for bird in bird_ids:
		bird_rows = grouped.get(bird, [])
		audio_dirs = sum(1 for row in bird_rows if row["audio_dir_exists"])
		roi_dirs = sum(1 for row in bird_rows if row["roi_available"])
		latent_dirs = sum(1 for row in bird_rows if row["latent_dir_exists"])
		latent_npz = sum(int(row.get("latent_npz_count") or 0) for row in bird_rows)
		audio_wav = sum(int(row.get("audio_wav_count") or 0) for row in bird_rows)
		dph_rows = _summarize_dph_rows(bird_rows)
		has_early = any(float(row["dph"]) <= 45 and int(row.get("latent_npz_count") or 0) > 0 for row in bird_rows)
		has_late = any(float(row["dph"]) >= 80 and int(row.get("latent_npz_count") or 0) > 0 for row in bird_rows)
		if not bird_rows:
			status = "missing_manifest_rows"
		elif audio_dirs == 0:
			status = "missing_audio"
		elif roi_dirs == 0:
			status = "missing_roi_artifacts"
		elif latent_npz == 0:
			status = "missing_latents"
		elif not (has_early and has_late):
			status = "missing_early_or_late_latents"
		else:
			status = "ready_for_full_rebuild"
		out[bird] = {
			"bird_id": bird,
			"status": status,
			"manifest_rows": int(len(bird_rows)),
			"manifest_num_files": int(sum(int(row.get("manifest_num_files") or 0) for row in bird_rows)),
			"audio_dirs_present": int(audio_dirs),
			"audio_wav_count": int(audio_wav),
			"roi_artifact_dirs_present": int(roi_dirs),
			"roi_parquet_dirs_present": int(sum(1 for row in bird_rows if row["roi_parquet_exists"])),
			"roi_txt_count": int(sum(int(row.get("roi_txt_count") or 0) for row in bird_rows)),
			"latent_dirs_present": int(latent_dirs),
			"latent_npz_count": int(latent_npz),
			"has_early_latents": bool(has_early),
			"has_late_latents": bool(has_late),
			"by_dph": dph_rows,
		}
	return out


def _summarize_dph_rows(rows: Sequence[dict]) -> dict:
	grouped: dict[float, list[dict]] = defaultdict(list)
	for row in rows:
		grouped[float(row["dph"])].append(row)
	out = {}
	for dph, dph_rows in sorted(grouped.items()):
		out[_format_dph(dph)] = {
			"manifest_num_files": int(sum(int(row.get("manifest_num_files") or 0) for row in dph_rows)),
			"audio_wav_count": int(sum(int(row.get("audio_wav_count") or 0) for row in dph_rows)),
			"roi_artifact_dirs_present": int(sum(1 for row in dph_rows if row["roi_available"])),
			"roi_parquet_dirs_present": int(sum(1 for row in dph_rows if row["roi_parquet_exists"])),
			"roi_txt_count": int(sum(int(row.get("roi_txt_count") or 0) for row in dph_rows)),
			"latent_npz_count": int(sum(int(row.get("latent_npz_count") or 0) for row in dph_rows)),
		}
	return out


def _overall_summary(birds: dict) -> dict:
	return {
		"birds_requested": int(len(birds)),
		"birds_with_audio_dirs": int(sum(payload["audio_dirs_present"] > 0 for payload in birds.values())),
		"birds_with_roi_artifacts": int(sum(payload["roi_artifact_dirs_present"] > 0 for payload in birds.values())),
		"birds_with_roi_parquet": int(sum(payload["roi_parquet_dirs_present"] > 0 for payload in birds.values())),
		"birds_with_latents": int(sum(payload["latent_npz_count"] > 0 for payload in birds.values())),
		"birds_ready_for_full_rebuild": int(sum(payload["status"] == "ready_for_full_rebuild" for payload in birds.values())),
		"birds_missing_manifest_rows": sorted(
			bird for bird, payload in birds.items() if payload["status"] == "missing_manifest_rows"
		),
		"birds_missing_roi_artifacts": sorted(
			bird
			for bird, payload in birds.items()
			if payload["manifest_rows"] > 0 and payload["roi_artifact_dirs_present"] == 0
		),
		"birds_missing_latents": sorted(
			bird
			for bird, payload in birds.items()
			if payload["manifest_rows"] > 0 and payload["latent_npz_count"] == 0
		),
	}


def _filtered_manifest_from_inventory(inventory: dict) -> dict:
	manifest = {"train": [], "test": []}
	for row in inventory["rows"]:
		entry = {
			"audio_dir_rel": row["audio_dir_rel"],
			"bird_id_norm": row["bird_id"],
			"regime": row.get("regime"),
			"dph": row["dph"],
			"num_files": row["manifest_num_files"],
			"split": row["split"],
		}
		manifest[row["split"]].append(entry)
	return manifest


def _next_blocker_text(inventory: dict) -> str:
	missing_manifest = inventory["summary"].get("birds_missing_manifest_rows") or []
	missing_roi = inventory["summary"].get("birds_missing_roi_artifacts") or []
	missing_latents = inventory["summary"].get("birds_missing_latents") or []
	parts = []
	if missing_manifest:
		parts.append(
			"Manifest rows are missing for "
			+ ", ".join(f"`{bird}`" for bird in missing_manifest)
			+ "."
		)
	if missing_roi:
		parts.append(
			"ROI artifacts are missing for "
			+ ", ".join(f"`{bird}`" for bird in missing_roi)
			+ "."
		)
	if missing_latents:
		parts.append(
			"Latent sequences are missing for "
			+ ", ".join(f"`{bird}`" for bird in missing_latents)
			+ "."
		)
	if not parts:
		return "No configured bird is missing ROI artifacts or latent sequences."
	parts.append(
		"Stage ROI artifacts and export latents with the same checkpoint/config used for PK249 before rerunning "
		"`scripts/analyze_developmental_replication.py` in full-rebuild mode."
	)
	return " ".join(parts)


def _join_root(root: Optional[Path], rel: str, fallback: object) -> Optional[Path]:
	if root is not None and rel:
		return Path(root) / rel
	if fallback:
		return Path(str(fallback))
	return None


def _roi_counts(
	roi_roots: Sequence[Path],
	audio_dir_rel: str,
	fallback: object,
	roi_parquet_name: str,
	roi_format: str,
	count_files: bool,
) -> dict:
	candidates = []
	for root in roi_roots:
		candidates.append((Path(root), Path(root) / audio_dir_rel))
	if fallback:
		fallback_path = Path(str(fallback))
		candidates.append((fallback_path.parent, fallback_path))
	for root, roi_dir in candidates:
		if not roi_dir.exists():
			continue
		parquet_path = roi_dir / str(roi_parquet_name)
		txt_count = _count_suffix(roi_dir, (".txt",), exclude_prefix="._") if count_files else None
		parquet_exists = parquet_path.exists()
		if roi_format == "parquet":
			available = parquet_exists
			detected = "parquet" if parquet_exists else None
		elif roi_format == "txt":
			available = bool(txt_count and txt_count > 0)
			detected = "txt" if available else None
		else:
			available = bool(parquet_exists or (txt_count and txt_count > 0))
			detected = "parquet" if parquet_exists else ("txt" if available else None)
		if available:
			return {
				"roi_root": root.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"roi_format_detected": detected,
				"roi_parquet": parquet_path.as_posix(),
				"roi_parquet_exists": bool(parquet_exists),
				"roi_txt_count": txt_count,
				"roi_available": True,
			}
	if candidates:
		root, roi_dir = candidates[0]
		parquet_path = roi_dir / str(roi_parquet_name)
		return {
			"roi_root": root.as_posix(),
			"roi_dir": roi_dir.as_posix(),
			"roi_format_detected": None,
			"roi_parquet": parquet_path.as_posix(),
			"roi_parquet_exists": False,
			"roi_txt_count": 0 if count_files else None,
			"roi_available": False,
		}
	return {
		"roi_root": None,
		"roi_dir": None,
		"roi_format_detected": None,
		"roi_parquet": None,
		"roi_parquet_exists": False,
		"roi_txt_count": 0 if count_files else None,
		"roi_available": False,
	}


def _latent_counts(
	latent_roots: Sequence[Path],
	audio_dir_rel: str,
	count_files: bool,
) -> dict:
	for root in latent_roots:
		latent_dir = Path(root) / audio_dir_rel
		if latent_dir.exists():
			return {
				"latent_root": Path(root).as_posix(),
				"latent_dir": latent_dir.as_posix(),
				"latent_dir_exists": True,
				"npz_count": _count_suffix(latent_dir, (".npz",)) if count_files else None,
				"json_count": _count_suffix(latent_dir, (".json",), exclude_prefix="._") if count_files else None,
			}
	return {
		"latent_root": None,
		"latent_dir": None,
		"latent_dir_exists": False,
		"npz_count": 0 if count_files else None,
		"json_count": 0 if count_files else None,
	}


def _count_suffix(
	path: Optional[Path],
	suffixes: tuple[str, ...],
	exclude_prefix: Optional[str] = None,
) -> int:
	if path is None or not path.exists() or not path.is_dir():
		return 0
	count = 0
	try:
		for name in os.listdir(path.as_posix()):
			if exclude_prefix and name.startswith(exclude_prefix):
				continue
			if name.endswith(suffixes):
				count += 1
	except OSError:
		return 0
	return count


def _optional_float(value: object) -> Optional[float]:
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _format_dph(value: float) -> str:
	return str(int(value)) if float(value).is_integer() else f"{float(value):g}"
