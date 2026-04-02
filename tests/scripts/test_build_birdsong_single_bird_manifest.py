import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_birdsong_single_bird_manifest.py"


def _entry(
    *,
    bird_id_norm: str,
    bird_id_raw: str,
    regime: str,
    dph: int | None,
    num_files: int,
    split: str,
) -> dict:
    suffix = "na" if dph is None else str(dph)
    rel = f"day43 Bells/{bird_id_raw.lower()}/{suffix}"
    return {
        "audio_dir_rel": rel,
        "audio_dir": f"/dataset/audio/{rel}",
        "roi_dir": f"/dataset/roi/{rel}",
        "bird_id_norm": bird_id_norm,
        "bird_id_raw": bird_id_raw,
        "regime": regime,
        "top_dir": "day43 Bells",
        "pre_bird_path": "day43 Bells",
        "dph": dph,
        "session_label": None,
        "num_files": num_files,
        "split": split,
    }


def test_build_single_bird_manifest_filters_range_and_rewrites_split(tmp_path: Path) -> None:
    entries = []
    for dph in range(30, 93):
        entries.append(
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=dph,
                num_files=dph,
                split="train" if dph % 2 else "test",
            )
        )
    entries.extend(
        [
            _entry(
                bird_id_norm="PK245",
                bird_id_raw="pk245",
                regime="bells",
                dph=50,
                num_files=999,
                split="train",
            ),
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=None,
                num_files=111,
                split="train",
            ),
        ]
    )

    source_manifest = {
        "created_utc": "2026-04-02T00:00:00Z",
        "metadata_path": "data/metadata/birdsong/birdsong_files.parquet",
        "root": "/dataset/audio",
        "roi_root": "/dataset/roi",
        "seed": 0,
        "train_fraction": 0.8,
        "train": [entry for entry in entries if entry["split"] == "train"],
        "test": [entry for entry in entries if entry["split"] == "test"],
    }
    source_path = tmp_path / "source_manifest.json"
    out_path = tmp_path / "single_bird.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            SCRIPT.as_posix(),
            "--source-manifest",
            source_path.as_posix(),
            "--bird-id",
            "pk249",
            "--min-dph",
            "33",
            "--max-dph",
            "90",
            "--test-every-n",
            "5",
            "--test-offset",
            "4",
            "--out",
            out_path.as_posix(),
        ],
        check=True,
        cwd=ROOT,
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    train_entries = manifest["train"]
    test_entries = manifest["test"]
    all_entries = train_entries + test_entries

    assert manifest["filters"]["bird_id_norm"] == "PK249"
    assert manifest["root"] == "/dataset/audio"
    assert manifest["roi_root"] == "/dataset/roi"
    assert len(train_entries) == 47
    assert len(test_entries) == 11
    assert all(entry["bird_id_norm"] == "PK249" for entry in all_entries)
    assert min(int(entry["dph"]) for entry in all_entries) == 33
    assert max(int(entry["dph"]) for entry in all_entries) == 90
    assert [int(entry["dph"]) for entry in test_entries] == [
        37,
        42,
        47,
        52,
        57,
        62,
        67,
        72,
        77,
        82,
        87,
    ]
    assert next(entry for entry in train_entries if int(entry["dph"]) == 34)["split"] == "train"
    assert next(entry for entry in test_entries if int(entry["dph"]) == 37)["split"] == "test"

    selected_dphs = list(range(33, 91))
    test_dphs = {37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87}
    expected_train_files = sum(dph for dph in selected_dphs if dph not in test_dphs)
    expected_test_files = sum(dph for dph in selected_dphs if dph in test_dphs)

    assert manifest["summary"]["train"] == {
        "birds": 1,
        "directories": 47,
        "files": expected_train_files,
        "by_regime": {
            "bells": {"birds": 1, "directories": 47, "files": expected_train_files}
        },
    }
    assert manifest["summary"]["test"] == {
        "birds": 1,
        "directories": 11,
        "files": expected_test_files,
        "by_regime": {
            "bells": {"birds": 1, "directories": 11, "files": expected_test_files}
        },
    }


def test_build_single_bird_manifest_applies_root_overrides(tmp_path: Path) -> None:
    source_manifest = {
        "created_utc": "2026-04-02T00:00:00Z",
        "metadata_path": "data/metadata/birdsong/birdsong_files.parquet",
        "root": "/dataset/audio",
        "roi_root": "/dataset/roi",
        "seed": 0,
        "train_fraction": 0.8,
        "train": [
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=33,
                num_files=10,
                split="train",
            ),
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=34,
                num_files=11,
                split="train",
            ),
        ],
        "test": [
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=35,
                num_files=12,
                split="test",
            ),
            _entry(
                bird_id_norm="PK249",
                bird_id_raw="pk249",
                regime="bells",
                dph=36,
                num_files=13,
                split="test",
            ),
        ],
    }
    source_path = tmp_path / "source_manifest.json"
    out_path = tmp_path / "single_bird.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            SCRIPT.as_posix(),
            "--source-manifest",
            source_path.as_posix(),
            "--bird-id",
            "PK249",
            "--min-dph",
            "33",
            "--max-dph",
            "36",
            "--test-every-n",
            "2",
            "--test-offset",
            "1",
            "--audio-root",
            "/override/audio",
            "--roi-root",
            "/override/roi",
            "--out",
            out_path.as_posix(),
        ],
        check=True,
        cwd=ROOT,
    )

    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["root"] == "/override/audio"
    assert manifest["roi_root"] == "/override/roi"

    first_train = manifest["train"][0]
    first_test = manifest["test"][0]
    assert first_train["audio_dir"] == "/override/audio/day43 Bells/pk249/33"
    assert first_train["roi_dir"] == "/override/roi/day43 Bells/pk249/33"
    assert first_test["audio_dir"] == "/override/audio/day43 Bells/pk249/34"
    assert first_test["roi_dir"] == "/override/roi/day43 Bells/pk249/34"
