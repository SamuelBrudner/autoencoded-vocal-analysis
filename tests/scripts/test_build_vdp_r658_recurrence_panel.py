import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
	Path(__file__).resolve().parents[2]
	/ "scripts"
	/ "build_vdp_r658_recurrence_panel.py"
)
SPEC = importlib.util.spec_from_file_location("build_vdp_r658_recurrence_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path, *, include_baseline: bool = True):
	pa = pytest.importorskip("pyarrow")
	pq = pytest.importorskip("pyarrow.parquet")
	entries = []
	members = []
	metadata = []
	days = (33, 34, 40) if include_baseline else (40, 41)
	for dph in days:
		rel = f"samba/R658/{dph}"
		entries.append(
			{
				"audio_dir_rel": rel,
				"audio_dir": f"/local/{rel}",
				"roi_dir": f"/roi/{rel}",
				"bird_id_norm": "R658",
				"regime": "samba",
				"dph": dph,
				"split": "train",
			}
		)
		for index in range(5):
			members.append(
				{
					"split": "train",
					"bird_id": "R658",
					"audio_dir_rel": rel,
					"filename": f"clip_{dph}_{index}.wav",
				}
			)
		metadata.append({"audio_dir_rel": rel, "tutor_start_day": 43})
	manifest_path = tmp_path / "source.json"
	manifest_path.write_text(json.dumps({"train": entries, "test": []}), encoding="utf-8")
	members_path = tmp_path / "members.jsonl"
	members_path.write_text(
		"".join(json.dumps(row) + "\n" for row in members), encoding="utf-8"
	)
	metadata_path = tmp_path / "metadata.parquet"
	pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
	return manifest_path, members_path, metadata_path


def test_panel_caps_each_day_and_preserves_explicit_metadata(tmp_path: Path) -> None:
	manifest_path, members_path, metadata_path = _fixture(tmp_path)
	panel, members = MODULE.build_panel(
		source_manifest_path=manifest_path,
		source_members_path=members_path,
		metadata_path=metadata_path,
		max_clips_per_day=3,
		seed=7,
	)
	assert panel["split_semantics"] == "descriptive_no_holdout"
	assert panel["summary"]["members"] == 9
	assert panel["summary"]["days"] == 3
	assert panel["summary"]["baseline_members"] == 6
	assert panel["summary"]["members_per_dph"] == {"33": 3, "34": 3, "40": 3}
	assert all(row["recording_id"] is None for row in members)
	assert {row["tutor_start_dph"] for row in members} == {43.0}
	assert all("audio_dir" not in row for row in panel["train"])
	assert all("roi_dir" not in row for row in panel["train"])

	panel_again, members_again = MODULE.build_panel(
		source_manifest_path=manifest_path,
		source_members_path=members_path,
		metadata_path=metadata_path,
		max_clips_per_day=3,
		seed=7,
	)
	assert panel_again["train"] == panel["train"]
	assert members_again == members


def test_panel_requires_early_explicit_baseline(tmp_path: Path) -> None:
	manifest_path, members_path, metadata_path = _fixture(
		tmp_path, include_baseline=False
	)
	with pytest.raises(ValueError, match="No explicit baseline members"):
		MODULE.build_panel(
			source_manifest_path=manifest_path,
			source_members_path=members_path,
			metadata_path=metadata_path,
		)
