"""AWS dry-run preparation for developmental baseline replication inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ava.cloud.manifest_sharding import iter_manifest_entry_pairs


DEFAULT_STAGING_BEAD = "autoencoded-vocal-analysis-obi.4.1"
DEFAULT_COHORT_MANIFEST_REL = Path(
	"docs/runs/artifacts/autoencoded-vocal-analysis-obi.4.1/"
	"20260513-011500-developmental-input-inventory/developmental_cohort_manifest.json"
)
DEFAULT_AUDIO_ROOT = Path("/Volumes/samsung_ssd/data/birdsong")
DEFAULT_SEGMENT_CONFIG_REL = Path("examples/configs/birdsong_roi_medium_pilot.yaml")
DEFAULT_PK249_LATENT_ROOT = Path(
	"/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/latent_sequences"
)
DEFAULT_WINDOW_LENGTH_SEC = 0.03
DEFAULT_HOP_LENGTH_SEC = 0.005804988662131519


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def s3_join(root: str, *parts: str) -> str:
	out = str(root).rstrip("/")
	for part in parts:
		part = str(part).strip("/")
		if part:
			out = f"{out}/{part}"
	return out


def summarize_cohort_manifest(manifest: dict, split: str = "all") -> dict:
	pairs = iter_manifest_entry_pairs(manifest, split=split)
	bird_ids: list[str] = []
	regimes: list[str] = []
	dph_values: list[float] = []
	total_files = 0
	for _, entry in pairs:
		bird = str(entry.get("bird_id_norm") or entry.get("bird_id_raw") or "").strip().upper()
		if bird and bird not in bird_ids:
			bird_ids.append(bird)
		regime = str(entry.get("regime") or "").strip()
		if regime and regime not in regimes:
			regimes.append(regime)
		try:
			dph_values.append(float(entry.get("dph")))
		except (TypeError, ValueError):
			pass
		total_files += int(entry.get("num_files") or 0)
	return {
		"split": split,
		"row_count": int(len(pairs)),
		"train_rows": int(len(manifest.get("train", []) or [])),
		"test_rows": int(len(manifest.get("test", []) or [])),
		"total_files": int(total_files),
		"bird_ids": bird_ids,
		"bird_count": int(len(bird_ids)),
		"regimes": regimes,
		"dph_min": None if not dph_values else float(min(dph_values)),
		"dph_max": None if not dph_values else float(max(dph_values)),
	}


def build_s3_layout(s3_root: str) -> dict:
	return {
		"root": str(s3_root).rstrip("/"),
		"inputs_root": s3_join(s3_root, "inputs"),
		"cohort_manifest": s3_join(s3_root, "inputs", "developmental_cohort_manifest.json"),
		"segment_config": s3_join(s3_root, "inputs", "segment_config.yaml"),
		"ava_config": s3_join(s3_root, "inputs", "config.yaml"),
		"ava_checkpoint": s3_join(s3_root, "inputs", "checkpoint_050.tar"),
		"audio_root": s3_join(s3_root, "audio"),
		"roi_root": s3_join(s3_root, "roi"),
		"latent_root": s3_join(s3_root, "latents", "ava_latent"),
		"roi_summary_root": s3_join(s3_root, "summaries", "roi"),
		"latent_summary_root": s3_join(s3_root, "summaries", "latents"),
	}


def recover_pk249_latent_lineage(latent_root: Path = DEFAULT_PK249_LATENT_ROOT) -> dict:
	root = Path(latent_root)
	if not root.exists():
		raise FileNotFoundError(f"PK249 latent root not found: {root.as_posix()}")
	for json_path in root.rglob("*.json"):
		if json_path.name.startswith("._"):
			continue
		try:
			metadata = load_json(json_path)
		except (OSError, UnicodeDecodeError, json.JSONDecodeError):
			continue
		npz_path = json_path.with_suffix(".npz")
		if not npz_path.exists():
			continue
		with np.load(npz_path) as payload:
			window_length_sec = _optional_scalar(payload, "window_length_sec")
			hop_length_sec = _optional_scalar(payload, "hop_length_sec")
			has_energy = "energy" in payload.files
			latent_dim = None
			if "mu" in payload:
				mu = payload["mu"]
				latent_dim = int(mu.shape[1]) if mu.ndim == 2 else None
		return {
			"source_json": json_path.as_posix(),
			"source_npz": npz_path.as_posix(),
			"schema_version": metadata.get("schema_version"),
			"checkpoint_path": metadata.get("checkpoint_path"),
			"config_path": metadata.get("config_path"),
			"window_length_sec": (
				DEFAULT_WINDOW_LENGTH_SEC if window_length_sec is None else float(window_length_sec)
			),
			"hop_length_sec": DEFAULT_HOP_LENGTH_SEC if hop_length_sec is None else float(hop_length_sec),
			"export_energy": bool(has_energy),
			"latent_dim": latent_dim,
		}
	raise FileNotFoundError(f"No readable PK249 latent metadata+npz pair found under {root.as_posix()}")


def build_upload_audio_dry_run_command(
	repo_root: Path,
	manifest_path: Path,
	audio_root: Path,
	s3_audio_root: str,
	jobs: int = 8,
	split: str = "all",
	max_dirs: Optional[int] = None,
) -> list[str]:
	cmd = [
		sys.executable,
		(repo_root / "scripts" / "cloud" / "aws" / "upload_manifest_audio_to_s3.py").as_posix(),
		"--manifest",
		manifest_path.as_posix(),
		"--split",
		str(split),
		"--audio-root",
		audio_root.as_posix(),
		"--s3-audio-root",
		str(s3_audio_root),
		"--jobs",
		str(int(jobs)),
		"--dry-run",
	]
	if max_dirs is not None:
		cmd.extend(["--max-dirs", str(int(max_dirs))])
	return cmd


def build_roi_submit_command(
	repo_root: Path,
	job_name: str,
	job_queue: str,
	job_definition: str,
	array_size: int,
	layout: dict,
	download_jobs: int = 8,
	jobs: int = 8,
	split: str = "all",
	emit_json: Optional[Path] = None,
	max_dirs: Optional[int] = None,
) -> list[str]:
	cmd = [
		sys.executable,
		(repo_root / "scripts" / "cloud" / "aws" / "submit_birdsong_roi_batch_array_job.py").as_posix(),
		"--job-name",
		str(job_name),
		"--job-queue",
		str(job_queue),
		"--job-definition",
		str(job_definition),
		"--array-size",
		str(int(array_size)),
		"--manifest-s3-uri",
		str(layout["cohort_manifest"]),
		"--segment-config-s3-uri",
		str(layout["segment_config"]),
		"--s3-audio-root",
		str(layout["audio_root"]),
		"--s3-roi-root",
		str(layout["roi_root"]),
		"--split",
		str(split),
		"--skip-existing",
		"--download-jobs",
		str(int(download_jobs)),
		"--jobs",
		str(int(jobs)),
		"--s3-summary-root",
		str(layout["roi_summary_root"]),
	]
	if max_dirs is not None:
		cmd.extend(["--max-dirs", str(int(max_dirs))])
	if emit_json is not None:
		cmd.extend(["--emit-json", emit_json.as_posix()])
	return cmd


def build_latent_submit_command(
	repo_root: Path,
	job_name: str,
	job_queue: str,
	job_definition: str,
	array_size: int,
	layout: dict,
	batch_size: int = 64,
	download_jobs: int = 8,
	device: str = "cpu",
	split: str = "all",
	hop_length_sec: float = DEFAULT_HOP_LENGTH_SEC,
	emit_json: Optional[Path] = None,
	max_dirs: Optional[int] = None,
	max_files_per_dir: Optional[int] = None,
	max_clips: Optional[int] = None,
) -> list[str]:
	cmd = [
		sys.executable,
		(repo_root / "scripts" / "cloud" / "aws" / "submit_birdsong_latent_export_batch_array_job.py").as_posix(),
		"--job-name",
		str(job_name),
		"--job-queue",
		str(job_queue),
		"--job-definition",
		str(job_definition),
		"--array-size",
		str(int(array_size)),
		"--manifest-s3-uri",
		str(layout["cohort_manifest"]),
		"--config-s3-uri",
		str(layout["ava_config"]),
		"--checkpoint-s3-uri",
		str(layout["ava_checkpoint"]),
		"--s3-audio-root",
		str(layout["audio_root"]),
		"--s3-roi-root",
		str(layout["roi_root"]),
		"--s3-latent-root",
		str(layout["latent_root"]),
		"--split",
		str(split),
		"--download-jobs",
		str(int(download_jobs)),
		"--batch-size",
		str(int(batch_size)),
		"--device",
		str(device),
		"--roi-format",
		"parquet",
		"--roi-parquet-name",
		"roi.parquet",
		"--hop-length-sec",
		str(float(hop_length_sec)),
		"--export-energy",
		"--skip-existing",
		"--s3-summary-root",
		str(layout["latent_summary_root"]),
	]
	if max_dirs is not None:
		cmd.extend(["--max-dirs", str(int(max_dirs))])
	if max_files_per_dir is not None:
		cmd.extend(["--max-files-per-dir", str(int(max_files_per_dir))])
	if max_clips is not None:
		cmd.extend(["--max-clips", str(int(max_clips))])
	if emit_json is not None:
		cmd.extend(["--emit-json", emit_json.as_posix()])
	return cmd


def run_preparation(
	repo_root: Path,
	out_dir: Path,
	report_path: Path,
	s3_root: str,
	roi_job_queue: str,
	roi_job_definition: str,
	latent_job_queue: str,
	latent_job_definition: str,
	cohort_manifest_path: Optional[Path] = None,
	audio_root: Path = DEFAULT_AUDIO_ROOT,
	segment_config_path: Optional[Path] = None,
	pk249_latent_root: Path = DEFAULT_PK249_LATENT_ROOT,
	split: str = "all",
	array_size: Optional[int] = None,
	smoke_max_dirs: int = 2,
	upload_jobs: int = 8,
	roi_download_jobs: int = 8,
	roi_jobs: int = 8,
	latent_download_jobs: int = 8,
	latent_batch_size: int = 64,
	latent_device: str = "cpu",
	run_aws_preflight: bool = False,
	aws_profile: Optional[str] = None,
) -> dict:
	repo_root = Path(repo_root)
	cohort_manifest_path = cohort_manifest_path or repo_root / DEFAULT_COHORT_MANIFEST_REL
	segment_config_path = segment_config_path or repo_root / DEFAULT_SEGMENT_CONFIG_REL
	if not str(s3_root).startswith("s3://"):
		raise ValueError("--s3-root must be an s3:// URI.")
	if not cohort_manifest_path.exists():
		raise FileNotFoundError(cohort_manifest_path.as_posix())
	if not segment_config_path.exists():
		raise FileNotFoundError(segment_config_path.as_posix())
	if not Path(audio_root).exists():
		raise FileNotFoundError(Path(audio_root).as_posix())
	if int(smoke_max_dirs) <= 0:
		raise ValueError("smoke_max_dirs must be positive.")
	out_dir.mkdir(parents=True, exist_ok=True)
	report_path.parent.mkdir(parents=True, exist_ok=True)

	manifest = load_json(cohort_manifest_path)
	summary = summarize_cohort_manifest(manifest, split=split)
	array_size = int(array_size or summary["row_count"])
	if array_size <= 0:
		raise ValueError("array_size must be positive.")
	layout = build_s3_layout(s3_root)
	lineage = recover_pk249_latent_lineage(pk249_latent_root)

	paths = {
		"upload_audio_dry_run_stdout": out_dir / "upload_audio_dry_run_stdout.txt",
		"roi_batch_payload": out_dir / "roi_batch_payload.json",
		"latent_batch_payload": out_dir / "latent_batch_payload.json",
		"roi_smoke_batch_payload": out_dir / "roi_smoke_batch_payload.json",
		"latent_smoke_batch_payload": out_dir / "latent_smoke_batch_payload.json",
		"aws_staging_plan": out_dir / "aws_staging_plan.json",
		"aws_preflight": out_dir / "aws_preflight.json",
	}

	upload_cmd = build_upload_audio_dry_run_command(
		repo_root=repo_root,
		manifest_path=cohort_manifest_path,
		audio_root=audio_root,
		s3_audio_root=layout["audio_root"],
		split=split,
		jobs=upload_jobs,
	)
	roi_cmd = build_roi_submit_command(
		repo_root=repo_root,
		job_name="ava-developmental-baseline-roi",
		job_queue=roi_job_queue,
		job_definition=roi_job_definition,
		array_size=array_size,
		layout=layout,
		split=split,
		download_jobs=roi_download_jobs,
		jobs=roi_jobs,
		emit_json=paths["roi_batch_payload"],
	)
	latent_cmd = build_latent_submit_command(
		repo_root=repo_root,
		job_name="ava-developmental-baseline-latent",
		job_queue=latent_job_queue,
		job_definition=latent_job_definition,
		array_size=array_size,
		layout=layout,
		split=split,
		batch_size=latent_batch_size,
		download_jobs=latent_download_jobs,
		device=latent_device,
		hop_length_sec=float(lineage["hop_length_sec"] or DEFAULT_HOP_LENGTH_SEC),
		emit_json=paths["latent_batch_payload"],
	)
	roi_smoke_cmd = build_roi_submit_command(
		repo_root=repo_root,
		job_name="ava-developmental-baseline-roi-smoke",
		job_queue=roi_job_queue,
		job_definition=roi_job_definition,
		array_size=1,
		layout=layout,
		split=split,
		download_jobs=roi_download_jobs,
		jobs=roi_jobs,
		emit_json=paths["roi_smoke_batch_payload"],
		max_dirs=smoke_max_dirs,
	)
	latent_smoke_cmd = build_latent_submit_command(
		repo_root=repo_root,
		job_name="ava-developmental-baseline-latent-smoke",
		job_queue=latent_job_queue,
		job_definition=latent_job_definition,
		array_size=1,
		layout=layout,
		split=split,
		batch_size=latent_batch_size,
		download_jobs=latent_download_jobs,
		device=latent_device,
		hop_length_sec=float(lineage["hop_length_sec"] or DEFAULT_HOP_LENGTH_SEC),
		emit_json=paths["latent_smoke_batch_payload"],
		max_dirs=smoke_max_dirs,
		max_files_per_dir=2,
	)

	upload_stdout = _run_capture(upload_cmd, cwd=repo_root)
	paths["upload_audio_dry_run_stdout"].write_text(upload_stdout, encoding="utf-8")
	_run_capture(roi_cmd, cwd=repo_root)
	_run_capture(latent_cmd, cwd=repo_root)
	_run_capture(roi_smoke_cmd, cwd=repo_root)
	_run_capture(latent_smoke_cmd, cwd=repo_root)

	aws_preflight = (
		run_readonly_aws_preflight(
			s3_root=s3_root,
			roi_job_queue=roi_job_queue,
			roi_job_definition=roi_job_definition,
			latent_job_queue=latent_job_queue,
			latent_job_definition=latent_job_definition,
			aws_profile=aws_profile,
		)
		if run_aws_preflight
		else {"status": "not_run", "reason": "Pass --run-aws-preflight to run read-only AWS checks."}
	)
	write_json(paths["aws_preflight"], aws_preflight)

	plan = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"repo_root": repo_root.as_posix(),
		"cohort_manifest_path": cohort_manifest_path.as_posix(),
		"segment_config_path": segment_config_path.as_posix(),
		"audio_root": Path(audio_root).as_posix(),
		"s3_layout": layout,
		"cohort_summary": summary,
		"array_size": int(array_size),
		"runtime_defaults": {
			"upload_jobs": int(upload_jobs),
			"roi_download_jobs": int(roi_download_jobs),
			"roi_jobs": int(roi_jobs),
			"latent_download_jobs": int(latent_download_jobs),
			"latent_batch_size": int(latent_batch_size),
			"latent_device": str(latent_device),
			"smoke_max_dirs": int(smoke_max_dirs),
		},
		"lineage": lineage,
		"safety": {
			"submits_aws_jobs": False,
			"uploads_audio": False,
			"requires_explicit_submit_step": True,
			"large_artifacts_out_of_git": True,
		},
		"commands": {
			"upload_audio_dry_run": upload_cmd,
			"roi_batch_payload": roi_cmd,
			"latent_batch_payload": latent_cmd,
			"roi_smoke_batch_payload": roi_smoke_cmd,
			"latent_smoke_batch_payload": latent_smoke_cmd,
		},
		"artifacts": {name: path.as_posix() for name, path in paths.items()},
		"aws_preflight": aws_preflight,
	}
	write_json(paths["aws_staging_plan"], plan)
	write_report(report_path, plan)
	return plan


def run_readonly_aws_preflight(
	s3_root: str,
	roi_job_queue: str,
	roi_job_definition: str,
	latent_job_queue: str,
	latent_job_definition: str,
	aws_profile: Optional[str] = None,
) -> dict:
	aws = shutil.which("aws")
	if not aws:
		return {"status": "aws_cli_missing"}
	base = [aws]
	if aws_profile:
		base.extend(["--profile", str(aws_profile)])
	checks = []
	for name, cmd in [
		("region", base + ["configure", "get", "region"]),
		("identity", base + ["sts", "get-caller-identity"]),
		("s3_root", base + ["s3", "ls", str(s3_root).rstrip("/") + "/"]),
		("roi_job_queue", base + ["batch", "describe-job-queues", "--job-queues", roi_job_queue]),
		("latent_job_queue", base + ["batch", "describe-job-queues", "--job-queues", latent_job_queue]),
		("roi_job_definition", base + ["batch", "describe-job-definitions", "--job-definitions", roi_job_definition]),
		("latent_job_definition", base + ["batch", "describe-job-definitions", "--job-definitions", latent_job_definition]),
	]:
		checks.append(_readonly_check(name, cmd))
	status = "ok" if all(check["returncode"] == 0 for check in checks) else "failed"
	return {"status": status, "checks": checks}


def write_report(report_path: Path, plan: dict) -> None:
	summary = plan["cohort_summary"]
	lineage = plan["lineage"]
	lines = [
		"# AWS Developmental Baseline Staging Plan",
		"",
		"## Summary",
		"",
		f"- Cohort rows: {summary['row_count']} directories; wav files: {summary['total_files']}.",
		f"- Birds: {summary['bird_count']} ({', '.join(summary['bird_ids'])}).",
		f"- dph range: {summary['dph_min']}..{summary['dph_max']}.",
		f"- S3 root: `{plan['s3_layout']['root']}`.",
		f"- AWS preflight: {plan['aws_preflight']['status']}.",
		"",
		"## AVA Lineage",
		"",
		f"- Source metadata: `{lineage['source_json']}`.",
		f"- Checkpoint path in export metadata: `{lineage.get('checkpoint_path')}`.",
		f"- Config path in export metadata: `{lineage.get('config_path')}`.",
		f"- Window/hop: {lineage['window_length_sec']} sec / {lineage['hop_length_sec']} sec.",
		f"- Export energy: {lineage['export_energy']}.",
		"",
		"## Dry-Run Artifacts",
		"",
	]
	for name, path in plan["artifacts"].items():
		lines.append(f"- `{name}`: `{os.path.relpath(path, report_path.parent.as_posix())}`")
	lines.extend(
		[
			"",
			"## Next Submit Gate",
			"",
			"No AWS jobs were submitted and no audio was uploaded by this preparation command. "
			"Before submission, stage the cohort manifest, segment config, recovered AVA config, "
			"and recovered checkpoint under the S3 `inputs/` layout, then run the one-shard smoke "
			"payloads before the full ROI and latent array payloads.",
		]
	)
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_capture(cmd: Sequence[str], cwd: Path) -> str:
	proc = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True)
	if proc.returncode != 0:
		output = (proc.stdout or "") + (proc.stderr or "")
		raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output}")
	return proc.stdout


def _readonly_check(name: str, cmd: Sequence[str]) -> dict:
	proc = subprocess.run(list(cmd), capture_output=True, text=True)
	stdout = (proc.stdout or "").strip()
	stderr = (proc.stderr or "").strip()
	if name == "identity" and proc.returncode == 0:
		stdout = _redact_identity_stdout(stdout)
	return {
		"name": name,
		"returncode": int(proc.returncode),
		"stdout": stdout[-2000:] if stdout else "",
		"stderr": stderr[-2000:] if stderr else "",
	}


def _redact_identity_stdout(stdout: str) -> str:
	try:
		payload = json.loads(stdout)
	except json.JSONDecodeError:
		return "[redacted identity output]"
	return json.dumps(
		{
			"UserId": "[redacted]",
			"Account": "[redacted]",
			"Arn": "[redacted]",
			"identity_available": bool(payload),
		},
		indent=2,
	)


def _optional_scalar(payload: Any, key: str) -> Optional[float]:
	if key not in payload:
		return None
	arr = payload[key]
	try:
		return float(arr.item())
	except Exception:
		return None
