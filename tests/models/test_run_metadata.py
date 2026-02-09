import hashlib
import json
from pathlib import Path

from ava.models.run_metadata import build_run_metadata, write_run_metadata


def test_build_run_metadata_includes_hash_and_manifest_root(tmp_path):
	config_path = tmp_path / "config.yaml"
	config_text = "preprocess:\n  fs: 32000\n"
	config_path.write_text(config_text, encoding="utf-8")

	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps({"root": "/data/birdsong"}), encoding="utf-8")

	expected_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
	metadata = build_run_metadata(
		config_path=config_path.as_posix(),
		manifest_path=manifest_path.as_posix(),
	)

	assert metadata["config_path"] == config_path.as_posix()
	assert metadata["config_sha256"] == expected_hash
	assert metadata["manifest_path"] == manifest_path.as_posix()
	assert metadata["dataset_root"] == "/data/birdsong"
	assert "git_commit" in metadata
	if metadata["git_commit"] is not None:
		assert isinstance(metadata["git_commit"], str)
		assert metadata["git_commit"]


def test_write_run_metadata_writes_file(tmp_path):
	config_path = tmp_path / "config.yaml"
	config_text = "config: 1\n"
	config_path.write_text(config_text, encoding="utf-8")

	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(
		json.dumps({"dataset_root": "/mnt/birds"}), encoding="utf-8"
	)

	save_dir = tmp_path / "run"
	out_path = write_run_metadata(
		save_dir.as_posix(),
		config_path=config_path.as_posix(),
		manifest_path=manifest_path.as_posix(),
	)

	assert out_path == (save_dir / "run_metadata.json").as_posix()
	loaded = json.loads(Path(out_path).read_text(encoding="utf-8"))
	assert loaded["config_sha256"] == hashlib.sha256(
		config_text.encode("utf-8")
	).hexdigest()
	assert loaded["dataset_root"] == "/mnt/birds"
