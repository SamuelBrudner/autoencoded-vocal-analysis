import importlib.util
import json
import shutil
from pathlib import Path

import pytest


MODULE_PATH = (
	Path(__file__).resolve().parents[2]
	/ "scripts"
	/ "validate_latent_export_collection.py"
)
SPEC = importlib.util.spec_from_file_location(
	"validate_latent_export_collection", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _copy_valid_pair(root: Path) -> tuple[Path, Path]:
	fixture = (
		Path(__file__).resolve().parents[2]
		/ "contracts"
		/ "ava_latent_sequence_v1"
		/ "fixtures"
		/ "valid"
	)
	root.mkdir(parents=True)
	npz_path = root / "fixture_clip.npz"
	json_path = root / "fixture_clip.json"
	shutil.copy(fixture / "fixture_clip.npz", npz_path)
	shutil.copy(fixture / "fixture_clip.json", json_path)
	return npz_path, json_path


def test_validates_and_hashes_complete_collection(tmp_path: Path) -> None:
	root = tmp_path / "latents"
	_copy_valid_pair(root)
	members, acceptance, summary = MODULE.validate_collection(
		root,
		dataset_sha256="a" * 64,
		configuration_sha256="b" * 64,
		checkpoint_sha256="c" * 64,
		code_sha256="d" * 64,
		expected_clips=1,
		require_energy=True,
	)
	assert len(members) == 1
	assert acceptance["schema_conformance"] == "pass"
	assert acceptance["member_manifest_sha256"] == summary["member_manifest_sha256"]
	assert summary["same_stem_pairs"] == 1


def test_rejects_absolute_metadata_paths(tmp_path: Path) -> None:
	root = tmp_path / "latents"
	_, json_path = _copy_valid_pair(root)
	metadata = json.loads(json_path.read_text(encoding="utf-8"))
	metadata["audio_path"] = "/tmp/private.wav"
	json_path.write_text(json.dumps(metadata), encoding="utf-8")
	with pytest.raises(ValueError, match="portable"):
		MODULE.validate_collection(
			root,
			dataset_sha256="a" * 64,
			configuration_sha256="b" * 64,
			checkpoint_sha256="c" * 64,
			code_sha256="d" * 64,
		)
