import json
import subprocess
import sys
from pathlib import Path


def _run_validator(manifest_path: Path, *args: str) -> subprocess.CompletedProcess:
	cmd = [
		sys.executable,
		"scripts/validate_birdsong_rois.py",
		"--manifest",
		str(manifest_path),
		*args,
	]
	return subprocess.run(cmd, capture_output=True, text=True)


def test_validate_birdsong_rois_missing_roi_file_exits_2(tmp_path):
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	(audio_dir / "sample.wav").write_bytes(b"")

	manifest = {
		"train": [
			{
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"audio_dir_rel": ".",
			}
		],
		"test": [],
	}
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

	result = _run_validator(manifest_path)
	assert result.returncode == 2
	report = json.loads(result.stdout)
	assert report["missing_roi_files"] == 1
	assert report["missing_roi_dirs"] == 0


def test_validate_birdsong_rois_passes_when_roi_present(tmp_path):
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	(audio_dir / "sample.wav").write_bytes(b"")
	(roi_dir / "sample.txt").write_text("0.0 0.1\n", encoding="utf-8")

	manifest = {
		"train": [
			{
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"audio_dir_rel": ".",
			}
		],
		"test": [],
	}
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

	result = _run_validator(manifest_path)
	assert result.returncode == 0
	report = json.loads(result.stdout)
	assert report["missing_roi_files"] == 0
	assert report["missing_roi_dirs"] == 0
	assert report["empty_roi_files"] == 0


def test_validate_birdsong_rois_fail_on_empty_exits_3(tmp_path):
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	(audio_dir / "sample.wav").write_bytes(b"")
	(roi_dir / "sample.txt").write_text("# header only\n", encoding="utf-8")

	manifest = {
		"train": [
			{
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"audio_dir_rel": ".",
			}
		],
		"test": [],
	}
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

	result = _run_validator(manifest_path, "--fail-on-empty")
	assert result.returncode == 3
	report = json.loads(result.stdout)
	assert report["missing_roi_files"] == 0
	assert report["missing_roi_dirs"] == 0
	assert report["empty_roi_files"] == 1

