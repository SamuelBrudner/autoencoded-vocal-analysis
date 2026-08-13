import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
	Path(__file__).resolve().parents[2]
	/ "scripts"
	/ "build_vdp_adult_recurrence_panel.py"
)
SPEC = importlib.util.spec_from_file_location(
	"build_vdp_adult_recurrence_panel", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_source(tmp_path: Path, birds: tuple[str, ...] = ("A", "B")) -> tuple[Path, Path]:
	audio_root = tmp_path / "audio"
	rows = []
	for bird_index, bird in enumerate(birds):
		bird_dir = audio_root / bird / bird
		bird_dir.mkdir(parents=True)
		paths = []
		for clip_index in range(4):
			path = bird_dir / f"{bird}_{clip_index}.wav"
			path.write_bytes(b"RIFF")
			paths.append(path)
		for pair_index in range(3):
			rows.append(
				{
					"bird_id": bird,
					"age_dph": str(90 + bird_index),
					"experimental_condition": "Isolate",
					"clip_a": paths[pair_index].relative_to(tmp_path).as_posix(),
					"clip_b": paths[pair_index + 1].relative_to(tmp_path).as_posix(),
					"clip_a_id": paths[pair_index].stem,
					"clip_b_id": paths[pair_index + 1].stem,
				}
			)
	manifest = tmp_path / "source.csv"
	with manifest.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	return manifest, audio_root


def test_build_panel_is_exact_portable_and_deterministic(tmp_path: Path) -> None:
	source, audio_root = _write_source(tmp_path)
	manifest, members = MODULE.build_panel(
		source_manifest_path=source,
		audio_root=audio_root,
		clips_per_bird=3,
	)

	assert manifest["summary"] == {
		"birds": 2,
		"members": 6,
		"panel": "adult_positive_control",
		"birds_with_null_dph": [],
		"birds_with_null_regime": [],
	}
	assert [row["filename"] for row in members if row["bird_id"] == "A"] == [
		"A_0.wav",
		"A_1.wav",
		"A_2.wav",
	]
	assert all(not Path(row["audio_dir_rel"]).is_absolute() for row in members)
	assert all(row["recording_id"] == Path(row["filename"]).stem for row in members)
	assert all(row["tutor_start_dph"] is None for row in members)
	assert all(entry["recording_id"] is None for entry in manifest["test"])

	manifest_again, members_again = MODULE.build_panel(
		source_manifest_path=source,
		audio_root=audio_root,
		clips_per_bird=3,
	)
	assert manifest_again["test"] == manifest["test"]
	assert members_again == members


def test_build_panel_rejects_clip_outside_audio_root(tmp_path: Path) -> None:
	source, audio_root = _write_source(tmp_path, birds=("A",))
	rows = list(csv.DictReader(source.open(encoding="utf-8")))
	outside = tmp_path / "outside.wav"
	outside.write_bytes(b"RIFF")
	rows[0]["clip_a"] = outside.name
	with source.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)

	with pytest.raises(ValueError, match="outside the declared audio root"):
		MODULE.build_panel(
			source_manifest_path=source,
			audio_root=audio_root,
			clips_per_bird=3,
		)


def test_build_panel_preserves_missing_dph_as_null(tmp_path: Path) -> None:
	source, audio_root = _write_source(tmp_path, birds=("A",))
	rows = list(csv.DictReader(source.open(encoding="utf-8")))
	for row in rows:
		row["age_dph"] = ""
	with source.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)

	manifest, members = MODULE.build_panel(
		source_manifest_path=source,
		audio_root=audio_root,
		clips_per_bird=3,
	)
	assert manifest["summary"]["birds_with_null_dph"] == ["A"]
	assert manifest["test"][0]["dph"] is None
	assert all(row["dph"] is None for row in members)


def test_build_panel_preserves_missing_regime_as_null(tmp_path: Path) -> None:
	source, audio_root = _write_source(tmp_path, birds=("A",))
	rows = list(csv.DictReader(source.open(encoding="utf-8")))
	for row in rows:
		row["experimental_condition"] = ""
	with source.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)

	manifest, members = MODULE.build_panel(
		source_manifest_path=source,
		audio_root=audio_root,
		clips_per_bird=3,
	)
	assert manifest["summary"]["birds_with_null_regime"] == ["A"]
	assert manifest["test"][0]["regime"] is None
	assert all(row["regime"] is None for row in members)
