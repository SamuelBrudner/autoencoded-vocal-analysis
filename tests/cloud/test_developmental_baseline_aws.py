from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import ava.cloud.developmental_baseline_aws as aws_plan
from ava.cloud.developmental_baseline_aws import (
	build_s3_layout,
	recover_pk249_latent_lineage,
	summarize_cohort_manifest,
)


def _manifest() -> dict:
	return {
		"train": [
			{
				"audio_dir_rel": "day43 Bells/pk249/33",
				"bird_id_norm": "PK249",
				"regime": "bells",
				"dph": 33,
				"num_files": 2,
			},
			{
				"audio_dir_rel": "day 60 bells/R426/80",
				"bird_id_norm": "R426",
				"regime": "bells",
				"dph": 80,
				"num_files": 3,
			},
		],
		"test": [],
	}


def _write_lineage_fixture(root: Path) -> None:
	latent_dir = root / "day43 Bells" / "pk249" / "33"
	latent_dir.mkdir(parents=True)
	(latent_dir / "clip.json").write_text(
		json.dumps(
			{
				"schema_version": "ava_latent_sequence_v1",
				"checkpoint_path": "/mnt/ava_cache/run/inputs/checkpoint_050.tar",
				"config_path": "/mnt/ava_cache/run/inputs/config.yaml",
			}
		),
		encoding="utf-8",
	)
	np.savez(
		latent_dir / "clip.npz",
		start_times_sec=np.array([0.0, 0.1]),
		window_length_sec=np.array(0.03),
		hop_length_sec=np.array(0.005804988662131519),
		mu=np.zeros((2, 32), dtype=np.float32),
		logvar=np.zeros((2, 32), dtype=np.float32),
		energy=np.ones(2, dtype=np.float32),
	)


def test_summarize_cohort_manifest_preserves_order() -> None:
	summary = summarize_cohort_manifest(_manifest())

	assert summary["row_count"] == 2
	assert summary["total_files"] == 5
	assert summary["bird_ids"] == ["PK249", "R426"]
	assert summary["dph_min"] == 33
	assert summary["dph_max"] == 80


def test_build_s3_layout_uses_expected_developmental_prefixes() -> None:
	layout = build_s3_layout("s3://bucket/ava/developmental-baseline-ava-v1/")

	assert layout["cohort_manifest"] == (
		"s3://bucket/ava/developmental-baseline-ava-v1/inputs/"
		"developmental_cohort_manifest.json"
	)
	assert layout["roi_root"].endswith("/roi")
	assert layout["latent_root"].endswith("/latents/ava_latent")


def test_recover_pk249_latent_lineage_from_metadata_and_npz(tmp_path: Path) -> None:
	_write_lineage_fixture(tmp_path)

	lineage = recover_pk249_latent_lineage(tmp_path)

	assert lineage["schema_version"] == "ava_latent_sequence_v1"
	assert lineage["checkpoint_path"].endswith("checkpoint_050.tar")
	assert lineage["config_path"].endswith("config.yaml")
	assert lineage["window_length_sec"] == 0.03
	assert lineage["hop_length_sec"] == 0.005804988662131519
	assert lineage["export_energy"] is True
	assert lineage["latent_dim"] == 32


def test_prepare_developmental_baseline_aws_cli_writes_payloads(tmp_path: Path) -> None:
	repo_root = Path(__file__).resolve().parents[2]
	manifest_path = tmp_path / "cohort.json"
	segment_config = tmp_path / "segment.yaml"
	latent_root = tmp_path / "latent"
	out_dir = tmp_path / "out"
	report_path = tmp_path / "report.md"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	segment_config.write_text("segment:\n  fs: 44100\n", encoding="utf-8")
	_write_lineage_fixture(latent_root)

	cmd = [
		sys.executable,
		(repo_root / "scripts" / "prepare_developmental_baseline_aws.py").as_posix(),
		"--cohort-manifest",
		manifest_path.as_posix(),
		"--segment-config",
		segment_config.as_posix(),
		"--pk249-latent-root",
		latent_root.as_posix(),
		"--audio-root",
		tmp_path.as_posix(),
		"--s3-root",
		"s3://bucket/ava/developmental-baseline-ava-v1",
		"--roi-job-queue",
		"roi-queue",
		"--roi-job-definition",
		"roi-jobdef",
		"--latent-job-queue",
		"latent-queue",
		"--latent-job-definition",
		"latent-jobdef",
		"--out-dir",
		out_dir.as_posix(),
		"--report-out",
		report_path.as_posix(),
	]
	result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
	if result.returncode != 0:
		raise AssertionError(f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

	plan = json.loads((out_dir / "aws_staging_plan.json").read_text(encoding="utf-8"))
	roi_payload = json.loads((out_dir / "roi_batch_payload.json").read_text(encoding="utf-8"))
	latent_payload = json.loads((out_dir / "latent_batch_payload.json").read_text(encoding="utf-8"))
	roi_smoke = json.loads((out_dir / "roi_smoke_batch_payload.json").read_text(encoding="utf-8"))

	roi_env = {item["name"]: item["value"] for item in roi_payload["containerOverrides"]["environment"]}
	latent_env = {
		item["name"]: item["value"] for item in latent_payload["containerOverrides"]["environment"]
	}
	roi_smoke_env = {
		item["name"]: item["value"] for item in roi_smoke["containerOverrides"]["environment"]
	}

	assert plan["cohort_summary"]["row_count"] == 2
	assert plan["array_size"] == 2
	assert plan["safety"]["submits_aws_jobs"] is False
	assert roi_env["AVA_NUM_SHARDS"] == "2"
	assert roi_env["AVA_S3_ROI_ROOT"].endswith("/roi")
	assert latent_env["AVA_EXPORT_ENERGY"] == "1"
	assert latent_env["AVA_HOP_LENGTH_SEC"] == "0.005804988662131519"
	assert roi_smoke_env["AVA_MAX_DIRS"] == "2"
	assert (out_dir / "upload_audio_dry_run_stdout.txt").exists()
	assert report_path.exists()


def test_readonly_aws_preflight_uses_job_definition_name_and_optional_prefix(monkeypatch) -> None:
	calls: list[list[str]] = []

	def fake_which(name: str) -> str:
		assert name == "aws"
		return "/usr/bin/aws"

	def fake_run(cmd, capture_output: bool, text: bool):
		calls.append(list(cmd))
		assert capture_output is True
		assert text is True
		if cmd[1:4] == ["configure", "get", "region"]:
			return subprocess.CompletedProcess(cmd, 0, stdout="us-east-1\n", stderr="")
		if cmd[1:3] == ["sts", "get-caller-identity"]:
			return subprocess.CompletedProcess(
				cmd,
				0,
				stdout=json.dumps({"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}),
				stderr="",
			)
		if cmd[1:3] == ["s3api", "head-bucket"]:
			return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
		if cmd[1:3] == ["s3", "ls"]:
			return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
		if cmd[1:3] == ["batch", "describe-job-queues"]:
			return subprocess.CompletedProcess(
				cmd,
				0,
				stdout=json.dumps(
					{"jobQueues": [{"jobQueueName": cmd[-1], "state": "ENABLED", "status": "VALID"}]}
				),
				stderr="",
			)
		if cmd[1:3] == ["batch", "describe-job-definitions"]:
			assert "--job-definition-name" in cmd
			assert "--job-definitions" not in cmd
			name = cmd[cmd.index("--job-definition-name") + 1]
			return subprocess.CompletedProcess(
				cmd,
				0,
				stdout=json.dumps(
					{
						"jobDefinitions": [
							{
								"jobDefinitionName": name,
								"revision": 1,
								"status": "ACTIVE",
								"type": "container",
								"platformCapabilities": ["FARGATE"],
							}
						]
					}
				),
				stderr="",
			)
		raise AssertionError(f"unexpected command: {cmd}")

	monkeypatch.setattr(aws_plan.shutil, "which", fake_which)
	monkeypatch.setattr(aws_plan.subprocess, "run", fake_run)

	result = aws_plan.run_readonly_aws_preflight(
		"s3://bucket/ava/developmental-baseline-ava-v1",
		"roi-queue",
		"roi-jobdef",
		"latent-queue",
		"latent-jobdef",
	)

	assert result["status"] == "ok"
	prefix_check = next(check for check in result["checks"] if check["name"] == "s3_prefix")
	assert prefix_check["required"] is False
	assert prefix_check["returncode"] == 1
	assert any("--job-definition-name" in call for call in calls)
