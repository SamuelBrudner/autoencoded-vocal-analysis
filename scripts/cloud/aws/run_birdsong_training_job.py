#!/usr/bin/env python3
"""AWS Batch runner for fixed-window shotgun VAE training (S3 -> local -> S3)."""

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


def _env(name: str, default: str | None = None) -> str | None:
	val = os.environ.get(name)
	if val is None or val == "":
		return default
	return val


def _env_int(name: str, default: int | None = None) -> int | None:
	val = _env(name)
	if val is None:
		return default
	return int(val)


def _require_aws_cli() -> str:
	aws = shutil.which("aws")
	if not aws:
		raise RuntimeError(
			"aws CLI is required but was not found on PATH. "
			"Install awscli (or AWS CLI v2) and configure credentials."
		)
	return aws


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if check and proc.returncode != 0:
		output = (proc.stdout or "") + (proc.stderr or "")
		raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n{output}")
	return proc


def _load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _join_s3_uri(root_uri: str, rel: str) -> str:
	root_uri = str(root_uri).rstrip("/")
	rel = str(rel or ".").lstrip("/")
	if rel in ("", "."):
		return root_uri
	return f"{root_uri}/{rel}"


def _iter_manifest_entries(manifest: dict):
	for split in ("train", "test"):
		for row in manifest.get(split, []):
			if not isinstance(row, dict):
				continue
			payload = dict(row)
			payload["split"] = row.get("split", split)
			yield payload


def _entry_rel(entry: dict) -> str:
	rel = entry.get("audio_dir_rel")
	if rel:
		return str(rel)
	audio_dir = entry.get("audio_dir")
	if audio_dir:
		return Path(str(audio_dir)).name
	return "."


def _sync_one_dir(
	aws: str,
	s3_src: str,
	local_dst: Path,
	include_patterns: list[str],
	quiet: bool,
) -> dict:
	start = time.time()
	local_dst.mkdir(parents=True, exist_ok=True)
	cmd = [aws, "s3", "sync", str(s3_src), str(local_dst), "--exclude", "*"]
	for pattern in include_patterns:
		cmd.extend(["--include", pattern])
	if quiet:
		cmd.append("--only-show-errors")
	proc = subprocess.run(cmd, capture_output=True, text=True)
	output = (proc.stdout or "") + (proc.stderr or "")
	return {
		"s3_src": str(s3_src),
		"local_dst": local_dst.as_posix(),
		"status": "ok" if proc.returncode == 0 else "error",
		"returncode": int(proc.returncode),
		"elapsed_sec": float(time.time() - start),
		"output": output.strip() if proc.returncode != 0 else None,
		"cmd": cmd if proc.returncode != 0 else None,
	}


def download_manifest_dirs(
	aws: str,
	manifest: dict,
	s3_audio_root: str,
	s3_roi_root: str,
	local_audio_root: Path,
	local_roi_root: Path,
	roi_format: str,
	roi_parquet_name: str,
	download_jobs: int,
	quiet: bool,
) -> list[dict]:
	"""Download unique audio/ROI directories needed by a manifest."""
	rels = sorted({_entry_rel(entry) for entry in _iter_manifest_entries(manifest)})
	tasks = []
	for rel in rels:
		tasks.append(
			(
				_join_s3_uri(s3_audio_root, rel),
				local_audio_root / rel,
				["*.wav", "*.WAV"],
			)
		)
		roi_patterns = [str(roi_parquet_name)] if roi_format == "parquet" else ["*.txt"]
		tasks.append(
			(
				_join_s3_uri(s3_roi_root, rel),
				local_roi_root / rel,
				roi_patterns,
			)
		)
	results = []
	with ThreadPoolExecutor(max_workers=max(1, int(download_jobs))) as pool:
		futures = [
			pool.submit(_sync_one_dir, aws, src, dst, patterns, quiet)
			for src, dst, patterns in tasks
		]
		for future in as_completed(futures):
			results.append(future.result())
	return sorted(results, key=lambda row: (row["local_dst"], row["s3_src"]))


def build_training_command(
	manifest_path: Path,
	config_path: Path,
	save_dir: Path,
	audio_root: Path,
	roi_root: Path,
	epochs: int,
	batch_size: int,
	num_workers: int,
	dataset_length: int,
	roi_cache_size: int,
	roi_format: str = "parquet",
	roi_parquet_name: str = "roi.parquet",
	preflight_sample_dirs: int = 100,
	preflight_sample_segments: int = 50_000,
	preflight_seed: int = 0,
	trainer_kwargs_json: str | None = None,
) -> list[str]:
	"""Build the local training command executed inside the AWS container."""
	cmd = [
		sys.executable,
		(ROOT / "scripts" / "launch_birdsong_training.py").as_posix(),
		"--manifest",
		manifest_path.as_posix(),
		"--config",
		config_path.as_posix(),
		"--save-dir",
		save_dir.as_posix(),
		"--audio-root",
		audio_root.as_posix(),
		"--roi-root",
		roi_root.as_posix(),
		"--streaming",
		"--roi-format",
		str(roi_format),
		"--roi-parquet-name",
		str(roi_parquet_name),
		"--epochs",
		str(int(epochs)),
		"--batch-size",
		str(int(batch_size)),
		"--num-workers",
		str(int(num_workers)),
		"--dataset-length",
		str(int(dataset_length)),
		"--roi-cache-size",
		str(int(roi_cache_size)),
		"--preflight-sample-dirs",
		str(int(preflight_sample_dirs)),
		"--preflight-sample-segments",
		str(int(preflight_sample_segments)),
		"--preflight-seed",
		str(int(preflight_seed)),
	]
	if trainer_kwargs_json:
		cmd.extend(["--trainer-kwargs-json", str(trainer_kwargs_json)])
	return cmd


def _run_to_log(cmd: list[str], log_path: Path) -> int:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with open(log_path, "w", encoding="utf-8") as handle:
		proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
	return int(proc.returncode)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run one shotgun VAE training job inside AWS Batch.")
	parser.add_argument("--manifest-s3-uri", type=str, default=_env("AVA_TRAIN_MANIFEST_S3_URI"))
	parser.add_argument("--config-s3-uri", type=str, default=_env("AVA_TRAIN_CONFIG_S3_URI"))
	parser.add_argument("--s3-audio-root", type=str, default=_env("AVA_TRAIN_S3_AUDIO_ROOT"))
	parser.add_argument("--s3-roi-root", type=str, default=_env("AVA_TRAIN_S3_ROI_ROOT"))
	parser.add_argument("--s3-output-root", type=str, default=_env("AVA_TRAIN_S3_OUTPUT_ROOT"))
	parser.add_argument("--run-name", type=str, default=_env("AVA_TRAIN_RUN_NAME"))
	parser.add_argument("--workdir", type=Path, default=Path("/tmp/ava_training_job"))
	parser.add_argument("--epochs", type=int, default=_env_int("AVA_TRAIN_EPOCHS", 100))
	parser.add_argument("--batch-size", type=int, default=_env_int("AVA_TRAIN_BATCH_SIZE", 128))
	parser.add_argument("--num-workers", type=int, default=_env_int("AVA_TRAIN_NUM_WORKERS", 4))
	parser.add_argument("--dataset-length", type=int, default=_env_int("AVA_TRAIN_DATASET_LENGTH", 200_000))
	parser.add_argument("--roi-cache-size", type=int, default=_env_int("AVA_TRAIN_ROI_CACHE_SIZE", 32))
	parser.add_argument("--preflight-sample-dirs", type=int, default=_env_int("AVA_TRAIN_PREFLIGHT_SAMPLE_DIRS", 100))
	parser.add_argument("--preflight-sample-segments", type=int, default=_env_int("AVA_TRAIN_PREFLIGHT_SAMPLE_SEGMENTS", 50_000))
	parser.add_argument("--preflight-seed", type=int, default=_env_int("AVA_TRAIN_PREFLIGHT_SEED", 0))
	parser.add_argument("--download-jobs", type=int, default=_env_int("AVA_TRAIN_DOWNLOAD_JOBS", 8))
	parser.add_argument("--roi-format", choices=["txt", "parquet"], default=_env("AVA_TRAIN_ROI_FORMAT", "parquet"))
	parser.add_argument("--roi-parquet-name", type=str, default=_env("AVA_TRAIN_ROI_PARQUET_NAME", "roi.parquet"))
	parser.add_argument("--trainer-kwargs-json", type=str, default=_env("AVA_TRAIN_TRAINER_KWARGS_JSON"))
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument(
		"--verbose-sync",
		action="store_true",
		help="Show full aws s3 sync output instead of --only-show-errors.",
	)

	args = parser.parse_args()
	required = {
		"--manifest-s3-uri": args.manifest_s3_uri,
		"--config-s3-uri": args.config_s3_uri,
		"--s3-audio-root": args.s3_audio_root,
		"--s3-roi-root": args.s3_roi_root,
		"--s3-output-root": args.s3_output_root,
	}
	missing = [name for name, value in required.items() if not value]
	if missing:
		raise ValueError("Missing required arguments: " + ", ".join(missing))
	if args.run_name is None:
		args.run_name = time.strftime("%Y%m%d-%H%M%S-shotgun-training", time.gmtime())

	local_manifest = args.workdir / "in" / "manifest.json"
	local_config = args.workdir / "in" / "config.yaml"
	local_audio = args.workdir / "audio"
	local_roi = args.workdir / "roi"
	save_dir = args.workdir / "out" / "training_run"
	summary_path = args.workdir / "out" / "training_job_summary.json"
	log_path = args.workdir / "out" / "training_stdout.log"
	command = build_training_command(
		manifest_path=local_manifest,
		config_path=local_config,
		save_dir=save_dir,
		audio_root=local_audio,
		roi_root=local_roi,
		epochs=args.epochs,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		dataset_length=args.dataset_length,
		roi_cache_size=args.roi_cache_size,
		roi_format=args.roi_format,
		roi_parquet_name=args.roi_parquet_name,
		preflight_sample_dirs=args.preflight_sample_dirs,
		preflight_sample_segments=args.preflight_sample_segments,
		preflight_seed=args.preflight_seed,
		trainer_kwargs_json=args.trainer_kwargs_json,
	)
	if args.dry_run:
		print(json.dumps({"run_name": args.run_name, "training_command": command}, indent=2))
		return

	aws = _require_aws_cli()
	args.workdir.mkdir(parents=True, exist_ok=True)
	local_manifest.parent.mkdir(parents=True, exist_ok=True)
	_run([aws, "s3", "cp", str(args.manifest_s3_uri), local_manifest.as_posix(), "--only-show-errors"])
	_run([aws, "s3", "cp", str(args.config_s3_uri), local_config.as_posix(), "--only-show-errors"])
	manifest = _load_json(local_manifest)
	download_results = download_manifest_dirs(
		aws=aws,
		manifest=manifest,
		s3_audio_root=str(args.s3_audio_root),
		s3_roi_root=str(args.s3_roi_root),
		local_audio_root=local_audio,
		local_roi_root=local_roi,
		roi_format=str(args.roi_format),
		roi_parquet_name=str(args.roi_parquet_name),
		download_jobs=int(args.download_jobs),
		quiet=not bool(args.verbose_sync),
	)
	failed_downloads = [row for row in download_results if row["status"] != "ok"]
	if failed_downloads:
		_write_json(summary_path, {"status": "failed_download", "downloads": download_results})
		raise RuntimeError(f"{len(failed_downloads)} S3 sync operations failed.")

	start = time.time()
	returncode = _run_to_log(command, log_path)
	status = "ok" if returncode == 0 else "failed_training"
	summary = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"run_name": args.run_name,
		"status": status,
		"returncode": int(returncode),
		"elapsed_sec": float(time.time() - start),
		"manifest_s3_uri": str(args.manifest_s3_uri),
		"config_s3_uri": str(args.config_s3_uri),
		"s3_audio_root": str(args.s3_audio_root),
		"s3_roi_root": str(args.s3_roi_root),
		"s3_output_root": str(args.s3_output_root),
		"training_command": command,
		"downloads": download_results,
	}
	_write_json(summary_path, summary)
	s3_out = _join_s3_uri(str(args.s3_output_root), str(args.run_name))
	_run([aws, "s3", "sync", (args.workdir / "out").as_posix(), s3_out, "--only-show-errors"], check=False)
	if returncode != 0:
		raise RuntimeError(f"Training command failed with exit code {returncode}. See {log_path}")
	print(json.dumps({"status": status, "s3_output": s3_out, "summary": summary_path.as_posix()}, indent=2))


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:  # pragma: no cover - CLI guardrail
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
