import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
	Path(__file__).resolve().parents[2] / "scripts" / "build_vdp_handoff_panel.py"
)
SPEC = importlib.util.spec_from_file_location("build_vdp_handoff_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _entry(split: str, bird: str, dph: int, rel: str, regime: str = "samba") -> dict:
	return {
		"audio_dir_rel": rel,
		"bird_id_norm": bird,
		"bird_id_raw": bird.lower(),
		"regime": regime,
		"dph": dph,
		"num_files": 2,
		"split": split,
	}


def test_build_panel_is_exact_balanced_and_metadata_backed(tmp_path: Path) -> None:
	pa = pytest.importorskip("pyarrow")
	pq = pytest.importorskip("pyarrow.parquet")
	train = []
	test = []
	members = []
	metadata_rows = []

	for dph in (33, 34, 35, 60, 61, 62):
		rel = f"samba/R658/{dph}"
		train.append(_entry("train", "R658", dph, rel))
		for index in range(2):
			members.append(
				{
					"split": "train",
					"bird_id": "R658",
					"audio_dir_rel": rel,
					"filename": f"r658_{dph}_{index}.wav",
					"size_bytes": 10,
				}
			)
		metadata_rows.append({"audio_dir_rel": rel, "tutor_start_day": 43})

	for bird, ages, tutor in (
		("A1", (88, 90, 91), 43),
		("A2", (95, 96), None),
	):
		for dph in ages:
			rel = f"adult/{bird}/{dph}"
			test.append(_entry("test", bird, dph, rel, regime="bells"))
			for index in range(2):
				members.append(
					{
						"split": "test",
						"bird_id": bird,
						"audio_dir_rel": rel,
						"filename": f"{bird}_{dph}_{index}.wav",
						"size_bytes": 10,
					}
				)
			metadata_rows.append(
				{"audio_dir_rel": rel, "tutor_start_day": tutor}
			)

	manifest_path = tmp_path / "source.json"
	manifest_path.write_text(
		json.dumps({"train": train, "test": test}), encoding="utf-8"
	)
	members_path = tmp_path / "members.jsonl"
	members_path.write_text(
		"".join(json.dumps(row) + "\n" for row in members), encoding="utf-8"
	)
	metadata_path = tmp_path / "metadata.parquet"
	pq.write_table(pa.Table.from_pylist(metadata_rows), metadata_path)

	panel, selected = MODULE.build_panel(
		source_manifest_path=manifest_path,
		source_members_path=members_path,
		metadata_path=metadata_path,
	)

	assert panel["summary"] == {
		"members": 14,
		"panels": {"early_r658": 6, "heldout_adult": 2, "later_r658": 6},
		"adult_birds": 2,
	}
	adult = [row for row in selected if row["panel"] == "heldout_adult"]
	assert {(row["bird_id"], row["dph"]) for row in adult} == {
		("A1", 90),
		("A2", 95),
	}
	assert {row["tutor_start_dph"] for row in selected if row["bird_id"] == "R658"} == {43.0}
	assert all(row["recording_id"] is None for row in selected)
	assert all("audio_dir" not in entry for entry in panel["train"] + panel["test"])
	assert all("roi_dir" not in entry for entry in panel["train"] + panel["test"])

	panel_again, selected_again = MODULE.build_panel(
		source_manifest_path=manifest_path,
		source_members_path=members_path,
		metadata_path=metadata_path,
	)
	assert panel_again["train"] == panel["train"]
	assert panel_again["test"] == panel["test"]
	assert selected_again == selected
