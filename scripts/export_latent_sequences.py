#!/usr/bin/env python3
"""Export per-clip latent sequences from a birdsong manifest.

This script drives `ava.models.latent_sequence` over many wav files, emitting
one `.npz` + one `.json` per clip in the canonical schema documented in
`docs/latent_sequence_export.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from ava.data.manifest_paths import resolve_manifest_entry_paths


def _load_manifest(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _select_entries(manifest: dict, split: str) -> list[dict]:
	if split == "train":
		return list(manifest.get("train", []))
	if split == "test":
		return list(manifest.get("test", []))
	entries: list[dict] = []
	entries.extend(manifest.get("train", []))
	entries.extend(manifest.get("test", []))
	return entries


def _resolve_entry_paths(
	entry: dict,
	audio_root: Optional[Path],
	roi_root: Optional[Path],
) -> Tuple[str, str, str]:
	audio_dir_rel = entry.get("audio_dir_rel")
	if not audio_dir_rel:
		audio_dir_rel = "."
		entry = dict(entry)
		entry["audio_dir_rel"] = audio_dir_rel

	audio_dir, roi_dir = resolve_manifest_entry_paths(
		entry, audio_root=audio_root, roi_root=roi_root
	)
	return str(audio_dir_rel), audio_dir, roi_dir


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


def _clip_id_for_wav(audio_dir_rel: str, wav_path: str) -> str:
	rel = audio_dir_rel if audio_dir_rel not in (None, "", ".") else ""
	stem = Path(wav_path).stem
	if not rel:
		return stem
	return (Path(rel) / stem).as_posix()


def _select_shard(tasks: list[dict], num_shards: int, shard_index: int) -> list[dict]:
	if num_shards <= 1:
		return tasks
	if shard_index < 0 or shard_index >= num_shards:
		raise ValueError("--shard-index must be in [0, --num-shards).")
	return [task for idx, task in enumerate(tasks) if (idx % num_shards) == shard_index]


def _minimal_entry_metadata(entry: dict) -> Dict[str, Any]:
	keys = [
		"audio_dir_rel",
		"bird_id_norm",
		"bird_id_raw",
		"regime",
		"dph",
		"session_label",
		"split",
	]
	return {key: entry.get(key) for key in keys if key in entry}


def _is_complete(out_npz: Path, out_json: Path) -> bool:
	return out_npz.exists() and out_json.exists()


def _load_parquet_roi_index(roi_parquet_path: Path) -> Dict[str, np.ndarray]:
	try:
		import pyarrow.parquet as pq  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError(
			"ROI parquet export requires pyarrow. Install pyarrow or use --roi-format txt."
		) from exc

	if not roi_parquet_path.exists():
		raise FileNotFoundError(roi_parquet_path.as_posix())

	table = pq.read_table(
		roi_parquet_path.as_posix(),
		columns=["clip_stem", "onsets_sec", "offsets_sec"],
	)
	payload = table.to_pydict()
	stems = payload.get("clip_stem") or []
	onsets = payload.get("onsets_sec") or []
	offsets = payload.get("offsets_sec") or []
	if not (len(stems) == len(onsets) == len(offsets)):
		raise ValueError(f"Malformed ROI parquet payload: {roi_parquet_path.as_posix()}")

	index: Dict[str, np.ndarray] = {}
	for stem, ons, offs in zip(stems, onsets, offsets):
		ons = ons or []
		offs = offs or []
		count = min(len(ons), len(offs))
		if count <= 0:
			index[str(stem)] = np.zeros((0, 2), dtype=np.float64)
			continue
		arr = np.stack(
			[
				np.asarray(ons[:count], dtype=np.float64),
				np.asarray(offs[:count], dtype=np.float64),
			],
			axis=1,
		)
		mask = (
			np.isfinite(arr[:, 0])
			& np.isfinite(arr[:, 1])
			& (arr[:, 1] > arr[:, 0])
		)
		index[str(stem)] = arr[mask]
	return index


def _lookup_parquet_rois(
	cache: Dict[str, Dict[str, np.ndarray]],
	roi_dir: str,
	roi_parquet_name: str,
	clip_stem: str,
) -> np.ndarray:
	key = os.path.abspath(str(roi_dir))
	if key not in cache:
		cache[key] = _load_parquet_roi_index(Path(roi_dir) / str(roi_parquet_name))
	return cache[key].get(str(clip_stem), np.zeros((0, 2), dtype=np.float64))


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Export per-clip latent sequences from a birdsong manifest."
	)
	parser.add_argument("--manifest", type=Path, required=True)
	parser.add_argument("--split", choices=["train", "test", "all"], default="all")
	parser.add_argument("--config", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--out-dir", type=Path, required=True)

	parser.add_argument("--audio-root", type=Path, default=None)
	parser.add_argument("--roi-root", type=Path, default=None)
	parser.add_argument("--roi-format", choices=["txt", "parquet"], default="txt")
	parser.add_argument("--roi-parquet-name", type=str, default="roi.parquet")
	parser.add_argument("--max-dirs", type=int, default=None)
	parser.add_argument("--max-files-per-dir", type=int, default=None)
	parser.add_argument("--max-clips", type=int, default=None)

	parser.add_argument("--num-shards", type=int, default=1)
	parser.add_argument("--shard-index", type=int, default=0)

	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--hop-length-sec", type=float, default=None)
	parser.add_argument("--start-time-sec", type=float, default=0.0)
	parser.add_argument("--end-time-sec", type=float, default=None)
	parser.add_argument("--no-rois", action="store_true")
	parser.add_argument(
		"--export-energy",
		action="store_true",
		help="Export per-window RMS energy aligned to latent timestamps.",
	)
	parser.add_argument("--audio-sha256", action="store_true")

	parser.add_argument(
		"--no-skip-existing",
		dest="skip_existing",
		action="store_false",
		help="Recompute outputs even when they already exist.",
	)
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--continue-on-error", action="store_true")
	parser.add_argument("--report-every", type=int, default=25)
	parser.add_argument("--summary-out", type=Path, default=None)
	parser.add_argument("--dry-run", action="store_true")

	parser.set_defaults(skip_existing=True)
	args = parser.parse_args()

	if args.num_shards <= 0:
		raise ValueError("--num-shards must be positive.")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be positive.")
	if args.max_dirs is not None and args.max_dirs <= 0:
		raise ValueError("--max-dirs must be positive.")
	if args.max_files_per_dir is not None and args.max_files_per_dir <= 0:
		raise ValueError("--max-files-per-dir must be positive.")
	if args.max_clips is not None and args.max_clips <= 0:
		raise ValueError("--max-clips must be positive.")

	manifest = _load_manifest(args.manifest)
	entries = _select_entries(manifest, args.split)
	if not entries:
		raise ValueError("No manifest entries found for the requested split.")

	if args.max_dirs is not None:
		entries = entries[: args.max_dirs]

	tasks: list[dict] = []
	for entry in entries:
		audio_dir_rel, audio_dir, roi_dir = _resolve_entry_paths(
			entry,
			audio_root=args.audio_root,
			roi_root=args.roi_root,
		)
		wavs = _list_wavs(audio_dir)
		if args.max_files_per_dir is not None:
			wavs = wavs[: args.max_files_per_dir]
		for wav_path in wavs:
			clip_id = _clip_id_for_wav(audio_dir_rel, wav_path)
			roi_path = None
			roi_parquet_path = None
			if not args.no_rois:
				if args.roi_format == "txt":
					roi_path = os.path.join(roi_dir, f"{Path(wav_path).stem}.txt")
				else:
					roi_parquet_path = os.path.join(roi_dir, str(args.roi_parquet_name))
			tasks.append(
				{
					"clip_id": clip_id,
					"audio_path": wav_path,
					"roi_dir": roi_dir,
					"roi_path": roi_path,
					"roi_parquet_path": roi_parquet_path,
					"entry": entry,
				}
			)

	tasks = sorted(tasks, key=lambda task: (task["clip_id"], task["audio_path"]))

	if args.max_clips is not None:
		tasks = tasks[: args.max_clips]

	tasks = _select_shard(tasks, args.num_shards, args.shard_index)

	if args.dry_run:
		print(
			f"Planned clips: {len(tasks)} (split={args.split} shard={args.shard_index}/{args.num_shards}, roi_format={args.roi_format})"
		)
		for task in tasks[:10]:
			print(f"  {task['clip_id']}  {task['audio_path']}")
		return

	from ava.models.latent_sequence import LatentSequenceEncoder  # noqa: E402

	args.out_dir.mkdir(parents=True, exist_ok=True)

	encoder = LatentSequenceEncoder(
		checkpoint_path=args.checkpoint,
		config=args.config,
		device=args.device,
	)

	start_time = time.time()
	exported = 0
	skipped = 0
	skipped_existing = 0
	skipped_no_roi = 0
	skipped_no_windows = 0
	failed = 0
	errors: list[dict] = []
	parquet_roi_cache: Dict[str, Dict[str, np.ndarray]] = {}

	total = len(tasks)
	print(f"Exporting {total} clips to: {args.out_dir.as_posix()}")

	for idx, task in enumerate(tasks, start=1):
		clip_id = str(task["clip_id"])
		audio_path = str(task["audio_path"])
		roi_path = task.get("roi_path")
		roi_parquet_path = task.get("roi_parquet_path")
		roi_dir = str(task.get("roi_dir") or "")
		entry = task.get("entry") or {}

		out_prefix = args.out_dir / Path(clip_id)
		out_npz = Path(f"{out_prefix.as_posix()}.npz")
		out_json = Path(f"{out_prefix.as_posix()}.json")

		if args.skip_existing and not args.force and _is_complete(out_npz, out_json):
			skipped += 1
			skipped_existing += 1
			continue

		out_prefix.parent.mkdir(parents=True, exist_ok=True)

		try:
			rois = None
			if (not args.no_rois) and args.roi_format == "parquet":
				rois = _lookup_parquet_rois(
					cache=parquet_roi_cache,
					roi_dir=roi_dir,
					roi_parquet_name=str(args.roi_parquet_name),
					clip_stem=Path(audio_path).stem,
				)
				if rois.size == 0:
					skipped += 1
					skipped_no_roi += 1
					continue

			seq = encoder.encode(
				audio_path=audio_path,
				roi_path=roi_path,
				rois=rois,
				batch_size=args.batch_size,
				hop_length_sec=args.hop_length_sec,
				start_time_sec=args.start_time_sec,
				end_time_sec=args.end_time_sec,
				return_energy=args.export_energy,
				compute_audio_sha256=args.audio_sha256,
			)
			seq.metadata["clip_id"] = clip_id
			seq.metadata["manifest_path"] = args.manifest.as_posix()
			seq.metadata["entry"] = _minimal_entry_metadata(entry)
			if roi_parquet_path is not None:
				seq.metadata["roi_path"] = str(roi_parquet_path)
				seq.metadata["roi_source"] = "parquet"

			arrays = seq.to_npz_arrays()
			np.savez_compressed(out_npz.as_posix(), **arrays)
			with open(out_json, "w", encoding="utf-8") as handle:
				json.dump(seq.metadata, handle, indent=2, sort_keys=False)
			exported += 1
		except Exception as exc:
			msg = str(exc)
			if "No windows available after ROI/time filtering." in msg:
				skipped += 1
				skipped_no_windows += 1
				continue
			failed += 1
			errors.append(
				{
					"clip_id": clip_id,
					"audio_path": audio_path,
					"roi_path": roi_path or roi_parquet_path,
					"error": msg,
				}
			)
			print(f"[{idx}/{total}] Failed: {clip_id} ({exc})", file=sys.stderr)
			if not args.continue_on_error:
				break

		if args.report_every and (idx % int(args.report_every) == 0 or idx == total):
			elapsed = max(1e-9, time.time() - start_time)
			rate = idx / elapsed
			print(
				f"[{idx}/{total}] exported={exported} skipped={skipped} failed={failed} ({rate:.2f} clips/s)"
			)

	summary = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"manifest_path": args.manifest.as_posix(),
		"config_path": args.config.as_posix(),
		"checkpoint_path": args.checkpoint.as_posix(),
		"out_dir": args.out_dir.as_posix(),
		"split": args.split,
		"roi_format": args.roi_format,
		"roi_parquet_name": str(args.roi_parquet_name),
		"num_shards": int(args.num_shards),
		"shard_index": int(args.shard_index),
		"total": int(total),
		"exported": int(exported),
		"skipped": int(skipped),
		"skipped_existing": int(skipped_existing),
		"skipped_no_roi": int(skipped_no_roi),
		"skipped_no_windows": int(skipped_no_windows),
		"failed": int(failed),
		"errors": errors,
	}

	print(
		f"Done. total={total} exported={exported} skipped={skipped} failed={failed}"
	)

	if args.summary_out is not None:
		args.summary_out.parent.mkdir(parents=True, exist_ok=True)
		args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	if failed:
		sys.exit(1)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
