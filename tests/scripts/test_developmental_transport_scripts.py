import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_developmental_transport_readiness.py"
INDEX_SCRIPT = ROOT / "scripts" / "build_latent_measure_index.py"


def _manifest_entry(
    *,
    bird: str,
    regime: str,
    top_dir: str,
    dph: int | None,
    split: str,
    num_files: int = 10,
) -> dict:
    suffix = "missing" if dph is None else str(dph)
    rel = f"{top_dir}/{bird}/{suffix}"
    return {
        "audio_dir_rel": rel,
        "audio_dir": f"/audio/{rel}",
        "roi_dir": f"/roi/{rel}",
        "bird_id_norm": bird,
        "bird_id_raw": bird.lower(),
        "regime": regime,
        "top_dir": top_dir,
        "pre_bird_path": top_dir,
        "dph": dph,
        "session_label": None,
        "num_files": num_files,
        "split": split,
    }


def test_audit_developmental_transport_readiness_reports_gates(tmp_path: Path) -> None:
    train = []
    for dph in range(35, 57):
        train.append(
            _manifest_entry(
                bird="BIRD1",
                regime="bells",
                top_dir="day35 Bells",
                dph=dph,
                split="train",
            )
        )
    for dph in range(40, 63):
        train.append(
            _manifest_entry(
                bird="BIRD2",
                regime="samba",
                top_dir="day43 Samba",
                dph=dph,
                split="train",
            )
        )
    test = [
        _manifest_entry(
            bird="ISO1",
            regime="isolates",
            top_dir="isolates",
            dph=50,
            split="test",
        ),
        _manifest_entry(
            bird="ISO1",
            regime="isolates",
            top_dir="isolates",
            dph=None,
            split="test",
            num_files=3,
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    out_json = tmp_path / "readiness.json"
    out_md = tmp_path / "readiness.md"
    manifest_path.write_text(json.dumps({"train": train, "test": test}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            AUDIT_SCRIPT.as_posix(),
            "--manifest",
            manifest_path.as_posix(),
            "--out-json",
            out_json.as_posix(),
            "--out-md",
            out_md.as_posix(),
            "--min-longitudinal-days",
            "20",
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summary = payload["summary"]

    assert summary["total_birds"] == 3
    assert summary["longitudinal_birds"] == 2
    assert summary["dph_missing_dirs"] == 1
    assert summary["tutor_start_day_values_by_regime"]["bells"] == [35.0]
    assert summary["tutor_start_day_values_by_regime"]["samba"] == [43.0]
    assert summary["readiness"]["bird_level_split_ok"] is True
    assert summary["readiness"]["has_longitudinal_multi_bird"] is True
    assert summary["readiness"]["has_tutor_onset_variation"] is True
    assert summary["readiness"]["has_isolate_contrast"] is True
    assert "Readiness Gates" in out_md.read_text(encoding="utf-8")


def _write_latent_clip(
    latent_dir: Path,
    rel_stem: str,
    *,
    bird: str,
    regime: str,
    top_dir: str,
    dph: int,
    split: str,
    n_windows: int,
) -> None:
    prefix = latent_dir / rel_stem
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mu = np.ones((n_windows, 3), dtype=np.float32)
    logvar = np.zeros((n_windows, 3), dtype=np.float32)
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        start_times_sec=np.arange(n_windows, dtype=np.float64),
        window_length_sec=np.asarray(0.01, dtype=np.float64),
        hop_length_sec=np.asarray(0.01, dtype=np.float64),
        mu=mu,
        logvar=logvar,
    )
    meta = {
        "schema_version": "ava_latent_sequence_v1",
        "clip_id": rel_stem,
        "entry": {
            "audio_dir_rel": f"{top_dir}/{bird}/{dph}",
            "bird_id_norm": bird,
            "regime": regime,
            "top_dir": top_dir,
            "dph": dph,
            "split": split,
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def test_build_latent_measure_index_groups_frozen_exports(tmp_path: Path) -> None:
    latent_dir = tmp_path / "latent"
    _write_latent_clip(
        latent_dir,
        "day35/BIRD1/clip_a",
        bird="BIRD1",
        regime="bells",
        top_dir="day35 Bells",
        dph=40,
        split="train",
        n_windows=4,
    )
    _write_latent_clip(
        latent_dir,
        "day35/BIRD1/clip_b",
        bird="BIRD1",
        regime="bells",
        top_dir="day35 Bells",
        dph=40,
        split="train",
        n_windows=6,
    )
    _write_latent_clip(
        latent_dir,
        "day43/BIRD2/clip_c",
        bird="BIRD2",
        regime="samba",
        top_dir="day43 Samba",
        dph=50,
        split="test",
        n_windows=5,
    )

    out_json = tmp_path / "measure_index.json"
    clips_csv = tmp_path / "clips.csv"
    measures_csv = tmp_path / "measures.csv"

    subprocess.run(
        [
            sys.executable,
            INDEX_SCRIPT.as_posix(),
            "--latent-dir",
            latent_dir.as_posix(),
            "--out-json",
            out_json.as_posix(),
            "--clips-csv",
            clips_csv.as_posix(),
            "--measures-csv",
            measures_csv.as_posix(),
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert payload["n_clips"] == 3
    assert payload["n_measures"] == 2
    bird1 = next(
        measure for measure in payload["measures"] if measure["bird_id_norm"] == "BIRD1"
    )
    assert bird1["dph"] == 40.0
    assert bird1["tutor_start_day"] == 35.0
    assert bird1["days_since_tutor"] == 5.0
    assert bird1["n_clips"] == 2
    assert bird1["total_windows"] == 10
    assert bird1["all_have_logvar"] is True
    assert clips_csv.exists()
    assert measures_csv.exists()
