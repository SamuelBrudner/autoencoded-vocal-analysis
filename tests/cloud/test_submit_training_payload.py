from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _submit_script() -> str:
	root = Path(__file__).resolve().parents[2]
	return (root / "scripts" / "cloud" / "aws" / "submit_birdsong_training_job.py").as_posix()


def _runner_module():
	root = Path(__file__).resolve().parents[2]
	path = root / "scripts" / "cloud" / "aws" / "run_birdsong_training_job.py"
	spec = importlib.util.spec_from_file_location("run_birdsong_training_job", path)
	module = importlib.util.module_from_spec(spec)
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


def _env_map(payload: dict) -> dict[str, str]:
	return {
		item["name"]: item["value"]
		for item in payload["containerOverrides"]["environment"]
	}


def test_submit_training_payload_omits_command_override_by_default():
	cmd = [
		sys.executable,
		_submit_script(),
		"--job-name",
		"shotgun-train",
		"--job-queue",
		"queue",
		"--job-definition",
		"jobdef",
		"--manifest-s3-uri",
		"s3://bucket/manifest.json",
		"--config-s3-uri",
		"s3://bucket/config.yaml",
		"--s3-audio-root",
		"s3://bucket/audio",
		"--s3-roi-root",
		"s3://bucket/roi",
		"--s3-output-root",
		"s3://bucket/out",
		"--run-name",
		"run1",
		"--epochs",
		"10",
		"--dataset-length",
		"1234",
	]
	out = subprocess.check_output(cmd, text=True)
	payload = json.loads(out)
	env = _env_map(payload)

	assert "command" not in payload["containerOverrides"]
	assert env["AVA_TRAIN_RUN_NAME"] == "run1"
	assert env["AVA_TRAIN_EPOCHS"] == "10"
	assert env["AVA_TRAIN_DATASET_LENGTH"] == "1234"
	assert env["AVA_TRAIN_ROI_FORMAT"] == "parquet"


def test_submit_training_payload_includes_command_override_when_requested():
	cmd = [
		sys.executable,
		_submit_script(),
		"--job-name",
		"shotgun-train",
		"--job-queue",
		"queue",
		"--job-definition",
		"jobdef",
		"--manifest-s3-uri",
		"s3://bucket/manifest.json",
		"--config-s3-uri",
		"s3://bucket/config.yaml",
		"--s3-audio-root",
		"s3://bucket/audio",
		"--s3-roi-root",
		"s3://bucket/roi",
		"--s3-output-root",
		"s3://bucket/out",
		"--override-command",
	]
	out = subprocess.check_output(cmd, text=True)
	payload = json.loads(out)

	assert payload["containerOverrides"]["command"] == [
		"python",
		"scripts/cloud/aws/run_birdsong_training_job.py",
	]


def test_training_runner_command_construction(tmp_path: Path):
	module = _runner_module()
	cmd = module.build_training_command(
		manifest_path=tmp_path / "manifest.json",
		config_path=tmp_path / "config.yaml",
		save_dir=tmp_path / "save",
		audio_root=tmp_path / "audio",
		roi_root=tmp_path / "roi",
		epochs=10,
		batch_size=16,
		num_workers=2,
		dataset_length=1024,
		roi_cache_size=8,
		trainer_kwargs_json='{"accelerator":"gpu","devices":1}',
	)

	assert "scripts/launch_birdsong_training.py" in cmd[1]
	assert "--streaming" in cmd
	assert cmd[cmd.index("--roi-format") + 1] == "parquet"
	assert cmd[cmd.index("--epochs") + 1] == "10"
	assert cmd[cmd.index("--dataset-length") + 1] == "1024"
	assert cmd[cmd.index("--trainer-kwargs-json") + 1] == '{"accelerator":"gpu","devices":1}'
