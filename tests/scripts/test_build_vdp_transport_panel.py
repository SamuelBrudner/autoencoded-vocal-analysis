from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "build_vdp_transport_panel.py"
)
SPEC = importlib.util.spec_from_file_location("build_vdp_transport_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _entry(bird: str, dph: int, split: str, regime: str = "bells") -> dict:
    rel = f"{regime}/{bird}/{dph}"
    return {
        "audio_dir_rel": rel,
        "audio_dir": f"/local/audio/{rel}",
        "roi_dir": f"/local/roi/{rel}",
        "bird_id_norm": bird,
        "regime": regime,
        "dph": float(dph),
        "num_files": 2,
        "split": split,
    }


def test_transport_panel_is_bird_disjoint_stable_and_filters_sparse_measures(
    tmp_path: Path, monkeypatch
) -> None:
    train = [_entry("A", day, "train") for day in (30, 31, 32)]
    train += [_entry("C", day, "train", "isolates") for day in (30, 31)]
    test = [_entry("B", day, "test", "samba") for day in (40, 41, 42)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"train": train, "test": test}), encoding="utf-8"
    )
    members_path = tmp_path / "members.jsonl"
    rows = []
    for entry in [*train, *test]:
        for index in range(2):
            rows.append(
                {
                    "split": entry["split"],
                    "bird_id": entry["bird_id_norm"],
                    "audio_dir_rel": entry["audio_dir_rel"],
                    "filename": f"clip-{index}.wav",
                    "size_bytes": 100,
                }
            )
    members_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.write_bytes(b"metadata")
    monkeypatch.setattr(
        MODULE,
        "_tutor_start_by_dir",
        lambda _path, directories: {directory: 35.0 for directory in directories},
    )

    first_manifest, first_members = MODULE.build_panel(
        source_manifest_path=manifest_path,
        source_members_path=members_path,
        metadata_path=metadata_path,
        min_clips_per_measure=2,
        max_clips_per_measure=2,
        min_measures_per_bird=3,
        seed=7,
    )
    second_manifest, second_members = MODULE.build_panel(
        source_manifest_path=manifest_path,
        source_members_path=members_path,
        metadata_path=metadata_path,
        min_clips_per_measure=2,
        max_clips_per_measure=2,
        min_measures_per_bird=3,
        seed=7,
    )

    assert first_members == second_members
    assert first_manifest["split_semantics"] == "bird_disjoint"
    assert first_manifest["summary"]["included_birds"] == 2
    assert first_manifest["summary"]["excluded_birds"] == ["C"]
    assert first_manifest["summary"]["included_measures"] == 6
    assert first_manifest["summary"]["selected_members"] == 12
    assert first_manifest["summary"]["bird_split_overlap"] == []
    assert {row["bird_id"] for row in first_members} == {"A", "B"}
    assert all("audio_dir" not in row for row in first_manifest["train"])
    assert all("roi_dir" not in row for row in first_manifest["test"])
    assert all(row["recording_id"] is None for row in first_members)
    assert all(row["tutor_start_dph"] == 35.0 for row in first_members)


def test_transport_panel_rejects_cross_split_bird(tmp_path: Path, monkeypatch) -> None:
    train = [_entry("A", 30, "train")]
    test = [_entry("A", 31, "test")]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"train": train, "test": test}), encoding="utf-8"
    )
    members_path = tmp_path / "members.jsonl"
    members_path.write_text(
        "".join(
            json.dumps(
                {
                    "split": entry["split"],
                    "bird_id": "A",
                    "audio_dir_rel": entry["audio_dir_rel"],
                    "filename": "clip.wav",
                }
            )
            + "\n"
            for entry in [*train, *test]
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.write_bytes(b"metadata")
    monkeypatch.setattr(MODULE, "_tutor_start_by_dir", lambda _path, dirs: {})

    try:
        MODULE.build_panel(
            source_manifest_path=manifest_path,
            source_members_path=members_path,
            metadata_path=metadata_path,
            min_clips_per_measure=1,
            max_clips_per_measure=1,
            min_measures_per_bird=2,
        )
    except ValueError as exc:
        assert "train/test" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("cross-split bird must be rejected")


def test_transport_panel_filters_members_without_exportable_rois(
    tmp_path: Path, monkeypatch
) -> None:
    train = [_entry("A", day, "train") for day in (30, 31)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"train": train, "test": []}), encoding="utf-8"
    )
    members_path = tmp_path / "members.jsonl"
    rows = []
    roi_root = tmp_path / "rois"
    for entry in train:
        rows.extend(
            {
                "split": "train",
                "bird_id": "A",
                "audio_dir_rel": entry["audio_dir_rel"],
                "filename": f"clip-{index}.wav",
            }
            for index in range(3)
        )
        roi_dir = roi_root / entry["audio_dir_rel"]
        roi_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "clip_stem": ["clip-0", "clip-1", "clip-2"],
                    "onsets_sec": [[0.1], [], [0.4]],
                    "offsets_sec": [[0.2], [], [0.3]],
                }
            ),
            roi_dir / "roi.parquet",
        )
    members_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata_path = tmp_path / "metadata.parquet"
    metadata_path.write_bytes(b"metadata")
    monkeypatch.setattr(
        MODULE,
        "_tutor_start_by_dir",
        lambda _path, directories: {directory: 35.0 for directory in directories},
    )

    manifest, members = MODULE.build_panel(
        source_manifest_path=manifest_path,
        source_members_path=members_path,
        metadata_path=metadata_path,
        roi_root_path=roi_root,
        min_clips_per_measure=1,
        max_clips_per_measure=1,
        min_measures_per_bird=2,
        seed=7,
    )

    assert {row["filename"] for row in members} == {"clip-0.wav"}
    assert manifest["summary"]["source_members_without_exportable_roi"] == 4
    assert manifest["summary"]["source_groups_without_exportable_roi"] == 0
    assert manifest["selection"]["roi_availability"] == (
        "at_least_one_finite_positive_duration_roi"
    )
