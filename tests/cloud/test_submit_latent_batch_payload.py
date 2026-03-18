from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "scripts"
        / "cloud"
        / "aws"
        / "submit_birdsong_latent_export_batch_array_job.py"
    ).as_posix()


def _to_env_map(payload: dict) -> dict[str, str]:
    env_items = payload["containerOverrides"]["environment"]
    return {item["name"]: item["value"] for item in env_items}


def test_submit_latent_payload_omits_command_override_by_default() -> None:
    cmd = [
        sys.executable,
        _script_path(),
        "--job-name",
        "latent-test",
        "--job-queue",
        "queue",
        "--job-definition",
        "jobdef",
        "--array-size",
        "3",
        "--manifest-s3-uri",
        "s3://bucket/prefix/manifest.json",
        "--config-s3-uri",
        "s3://bucket/prefix/config.yaml",
        "--checkpoint-s3-uri",
        "s3://bucket/prefix/checkpoint.tar",
        "--s3-audio-root",
        "s3://bucket/prefix/audio",
        "--s3-roi-root",
        "s3://bucket/prefix/roi",
        "--s3-latent-root",
        "s3://bucket/prefix/latent",
        "--split",
        "train",
        "--download-jobs",
        "2",
        "--batch-size",
        "8",
        "--skip-existing",
        "--export-energy",
        "--s3-summary-root",
        "s3://bucket/prefix/summaries",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)

    overrides = payload["containerOverrides"]
    assert "environment" in overrides
    assert "command" not in overrides

    env_map = _to_env_map(payload)
    assert env_map["AVA_NUM_SHARDS"] == "3"
    assert env_map["AVA_SPLIT"] == "train"
    assert env_map["AVA_EXPORT_ENERGY"] == "1"
    assert env_map["AVA_SKIP_EXISTING"] == "1"
    assert env_map["AVA_S3_ROI_ROOT"] == "s3://bucket/prefix/roi"


def test_submit_latent_payload_includes_command_override_when_requested() -> None:
    cmd = [
        sys.executable,
        _script_path(),
        "--job-name",
        "latent-test",
        "--job-queue",
        "queue",
        "--job-definition",
        "jobdef",
        "--array-size",
        "1",
        "--manifest-s3-uri",
        "s3://bucket/prefix/manifest.json",
        "--config-s3-uri",
        "s3://bucket/prefix/config.yaml",
        "--checkpoint-s3-uri",
        "s3://bucket/prefix/checkpoint.tar",
        "--s3-audio-root",
        "s3://bucket/prefix/audio",
        "--s3-roi-root",
        "s3://bucket/prefix/roi",
        "--s3-latent-root",
        "s3://bucket/prefix/latent",
        "--override-command",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    overrides = payload["containerOverrides"]
    assert overrides["command"] == [
        "python",
        "scripts/cloud/aws/run_birdsong_latent_export_batch_shard.py",
    ]


def test_submit_latent_payload_supports_no_rois_without_roi_root() -> None:
    cmd = [
        sys.executable,
        _script_path(),
        "--job-name",
        "latent-test",
        "--job-queue",
        "queue",
        "--job-definition",
        "jobdef",
        "--array-size",
        "1",
        "--manifest-s3-uri",
        "s3://bucket/prefix/manifest.json",
        "--config-s3-uri",
        "s3://bucket/prefix/config.yaml",
        "--checkpoint-s3-uri",
        "s3://bucket/prefix/checkpoint.tar",
        "--s3-audio-root",
        "s3://bucket/prefix/audio",
        "--s3-latent-root",
        "s3://bucket/prefix/latent",
        "--no-rois",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    env_map = _to_env_map(payload)

    assert env_map["AVA_NO_ROIS"] == "1"
    assert "AVA_S3_ROI_ROOT" not in env_map
