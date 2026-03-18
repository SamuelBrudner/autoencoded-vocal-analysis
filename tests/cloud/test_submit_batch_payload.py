from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "scripts" / "cloud" / "aws" / "submit_birdsong_roi_batch_array_job.py").as_posix()


def test_submit_payload_omits_command_override_by_default() -> None:
    cmd = [
        sys.executable,
        _script_path(),
        "--job-name",
        "test",
        "--job-queue",
        "queue",
        "--job-definition",
        "jobdef",
        "--array-size",
        "3",
        "--manifest-s3-uri",
        "s3://bucket/prefix/manifest.json",
        "--segment-config-s3-uri",
        "s3://bucket/prefix/segment.yaml",
        "--s3-audio-root",
        "s3://bucket/prefix/audio",
        "--s3-roi-root",
        "s3://bucket/prefix/roi",
        "--split",
        "train",
        "--skip-existing",
        "--download-jobs",
        "1",
        "--jobs",
        "1",
        "--s3-summary-root",
        "s3://bucket/prefix/summaries",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    overrides = payload["containerOverrides"]
    assert "environment" in overrides
    assert "command" not in overrides


def test_submit_payload_includes_command_override_when_requested() -> None:
    cmd = [
        sys.executable,
        _script_path(),
        "--job-name",
        "test",
        "--job-queue",
        "queue",
        "--job-definition",
        "jobdef",
        "--array-size",
        "1",
        "--manifest-s3-uri",
        "s3://bucket/prefix/manifest.json",
        "--segment-config-s3-uri",
        "s3://bucket/prefix/segment.yaml",
        "--s3-audio-root",
        "s3://bucket/prefix/audio",
        "--s3-roi-root",
        "s3://bucket/prefix/roi",
        "--override-command",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out)
    overrides = payload["containerOverrides"]
    assert overrides["command"] == ["python", "scripts/cloud/aws/run_birdsong_roi_batch_shard.py"]

