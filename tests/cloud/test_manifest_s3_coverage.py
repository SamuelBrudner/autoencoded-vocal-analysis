from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ava.cloud.manifest_s3_coverage import (
	build_manifest_audio_prefixes,
	parse_aws_s3_ls_key,
	parse_s3_uri,
	summarize_manifest_audio_coverage,
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
				"audio_dir_rel": "day 60 samba/Day_60_SAMBA/R150/36",
				"bird_id_norm": "R150",
				"regime": "samba",
				"dph": 36,
				"num_files": 1,
			},
		],
		"test": [],
	}


def test_parse_s3_uri_requires_bucket_and_key() -> None:
	parsed = parse_s3_uri("s3://bucket/root/audio")

	assert parsed.bucket == "bucket"
	assert parsed.key == "root/audio"


def test_parse_aws_s3_ls_key_preserves_spaces() -> None:
	key = parse_aws_s3_ls_key(
		"2026-05-15 12:00:00       1234 root/audio/day43 Bells/pk249/33/a.wav\n"
	)

	assert key == "root/audio/day43 Bells/pk249/33/a.wav"


def test_summarize_manifest_audio_coverage_counts_expected_prefixes() -> None:
	prefixes = build_manifest_audio_prefixes(
		_manifest(),
		split="all",
		s3_audio_root="s3://bucket/root/audio",
	)
	keys = [
		"root/audio/day43 Bells/pk249/33/a.wav",
		"root/audio/day43 Bells/pk249/33/b.WAV",
		"root/audio/day43 Bells/pk249/33/._b.wav",
		"root/audio/day 60 samba/Day_60_SAMBA/R150/36/c.wav",
		"root/audio/untracked/file.wav",
		"root/audio/day43 Bells/pk249/33/not_audio.txt",
	]

	summary = summarize_manifest_audio_coverage(prefixes, keys)
	by_rel = {row["audio_dir_rel"]: row for row in summary["rows"]}

	assert summary["status"] == "ok"
	assert summary["observed_dirs"] == 2
	assert summary["missing_dirs"] == 0
	assert summary["count_mismatch_dirs"] == 0
	assert summary["unmatched_wav_keys"] == 1
	assert by_rel["day43 Bells/pk249/33"]["observed_wav_files"] == 2
	assert by_rel["day 60 samba/Day_60_SAMBA/R150/36"]["observed_wav_files"] == 1


def test_check_manifest_audio_s3_coverage_cli_uses_listing_file(tmp_path: Path) -> None:
	repo_root = Path(__file__).resolve().parents[2]
	manifest_path = tmp_path / "manifest.json"
	listing_path = tmp_path / "listing.txt"
	out_path = tmp_path / "coverage.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	listing_path.write_text(
		"\n".join(
			[
				"2026-05-15 12:00:00       1234 root/audio/day43 Bells/pk249/33/a.wav",
				"2026-05-15 12:00:01       5678 root/audio/day43 Bells/pk249/33/b.wav",
			]
		),
		encoding="utf-8",
	)

	cmd = [
		sys.executable,
		(repo_root / "scripts" / "cloud" / "aws" / "check_manifest_audio_s3_coverage.py").as_posix(),
		"--manifest",
		manifest_path.as_posix(),
		"--s3-audio-root",
		"s3://bucket/root/audio",
		"--listing-file",
		listing_path.as_posix(),
		"--out",
		out_path.as_posix(),
	]
	result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
	if result.returncode != 0:
		raise AssertionError(f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

	summary = json.loads(out_path.read_text(encoding="utf-8"))
	assert summary["missing_dirs"] == 1
	assert summary["count_mismatch_dirs"] == 1
	assert summary["observed_wav_files"] == 2


def test_check_manifest_audio_s3_coverage_cli_can_fail_on_missing(tmp_path: Path) -> None:
	repo_root = Path(__file__).resolve().parents[2]
	manifest_path = tmp_path / "manifest.json"
	listing_path = tmp_path / "listing.txt"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	listing_path.write_text(
		"2026-05-15 12:00:00       1234 root/audio/day43 Bells/pk249/33/a.wav\n",
		encoding="utf-8",
	)

	cmd = [
		sys.executable,
		(repo_root / "scripts" / "cloud" / "aws" / "check_manifest_audio_s3_coverage.py").as_posix(),
		"--manifest",
		manifest_path.as_posix(),
		"--s3-audio-root",
		"s3://bucket/root/audio",
		"--listing-file",
		listing_path.as_posix(),
		"--fail-on-missing",
	]
	result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

	assert result.returncode == 1
