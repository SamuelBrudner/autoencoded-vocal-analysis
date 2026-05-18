import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow is required for parquet ROI bundle tests.")


def test_run_birdsong_roi_parquet_bundle_and_coverage(tmp_path: Path) -> None:
    _require_pyarrow()

    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "rois"
    audio_dir.mkdir()
    roi_dir.mkdir()

    fs = 44100
    duration_sec = 1.0
    n = int(fs * duration_sec)
    t = np.arange(n, dtype=np.float32) / float(fs)
    audio = np.zeros(n, dtype=np.float32)
    burst = (t >= 0.2) & (t <= 0.25)
    audio[burst] = 0.9 * np.sin(2 * np.pi * 1000.0 * t[burst])
    wav = (audio * 32767.0).astype(np.int16)
    wav_path = audio_dir / "sample.wav"
    wavfile.write(wav_path.as_posix(), fs, wav)

    segment_config = tmp_path / "segment.yaml"
    segment_config.write_text(
        "\n".join(
            [
                "segment:",
                "  fs: 44100",
                "  min_freq: 300.0",
                "  max_freq: 10000.0",
                "  nperseg: 512",
                "  noverlap: 256",
                "  spec_min_val: 0.0",
                "  spec_max_val: 10.0",
                "  th_1: 0.01",
                "  th_2: 0.02",
                "  th_3: 0.03",
                "  min_dur: 0.005",
                "  max_dur: 0.5",
                "  smoothing_timescale: 0.004",
                "  softmax: false",
                "  temperature: 1.0",
                "  algorithm: amplitude",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "train": [
            {
                "audio_dir": audio_dir.as_posix(),
                "roi_dir": roi_dir.as_posix(),
                "audio_dir_rel": ".",
                "split": "train",
            }
        ],
        "test": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_birdsong_roi.py",
            "--segment-config",
            str(segment_config),
            "--manifest",
            str(manifest_path),
            "--split",
            "train",
            "--jobs",
            "1",
            "--roi-output-format",
            "parquet",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    parquet_path = roi_dir / "roi.parquet"
    assert parquet_path.exists()

    out_dir = tmp_path / "coverage"
    cov = subprocess.run(
        [
            sys.executable,
            "scripts/report_birdsong_roi_coverage.py",
            "--manifest",
            str(manifest_path),
            "--split",
            "train",
            "--roi-format",
            "parquet",
            "--out-dir",
            str(out_dir),
            "--max-dirs",
            "1",
            "--max-files-per-dir",
            "10",
            "--jobs",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert cov.returncode == 0, cov.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing_roi_dirs"] == 0
    assert summary["missing_roi_files"] == 0
    assert summary["roi_parse_errors"] == 0


def test_run_birdsong_roi_parquet_skips_unreadable_wavs(tmp_path: Path) -> None:
    _require_pyarrow()
    import pyarrow.parquet as pq

    audio_dir = tmp_path / "audio"
    roi_dir = tmp_path / "rois"
    audio_dir.mkdir()
    roi_dir.mkdir()

    fs = 44100
    n = int(fs * 0.5)
    t = np.arange(n, dtype=np.float32) / float(fs)
    audio = (0.8 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    wavfile.write(
        (audio_dir / "good.wav").as_posix(),
        fs,
        (audio * 32767.0).astype(np.int16),
    )
    (audio_dir / "bad.wav").write_bytes(b"\x00\x00\x00\x00" + b"\x01" * 1024)

    segment_config = tmp_path / "segment.yaml"
    segment_config.write_text(
        "\n".join(
            [
                "segment:",
                "  fs: 44100",
                "  min_freq: 300.0",
                "  max_freq: 10000.0",
                "  nperseg: 512",
                "  noverlap: 256",
                "  spec_min_val: 0.0",
                "  spec_max_val: 10.0",
                "  th_1: 0.01",
                "  th_2: 0.02",
                "  th_3: 0.03",
                "  min_dur: 0.005",
                "  max_dur: 0.5",
                "  smoothing_timescale: 0.004",
                "  softmax: false",
                "  temperature: 1.0",
                "  algorithm: amplitude",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "train": [
            {
                "audio_dir": audio_dir.as_posix(),
                "roi_dir": roi_dir.as_posix(),
                "audio_dir_rel": ".",
                "split": "train",
            }
        ],
        "test": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = tmp_path / "roi_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_birdsong_roi.py",
            "--segment-config",
            str(segment_config),
            "--manifest",
            str(manifest_path),
            "--split",
            "train",
            "--jobs",
            "1",
            "--roi-output-format",
            "parquet",
            "--summary-out",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    table = pq.read_table(roi_dir / "roi.parquet")
    assert table.column("clip_stem").to_pylist() == ["good"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["ok"] == 1
    assert summary["failed"] == 0
    assert summary["clips_total"] == 1
    assert summary["clips_failed_read"] == 1
    assert summary["errors"] == []
    assert summary["results"][0]["clips_total"] == 1
    assert summary["results"][0]["clips_failed_read"] == 1
    assert summary["results"][0]["clip_read_error_examples"][0]["clip_stem"] == "bad"
