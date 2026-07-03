import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_latent_hop_sensitivity.py"


def _write_clip(latent_dir: Path, rel_stem: str, hop: float, mu: np.ndarray) -> None:
    prefix = latent_dir / rel_stem
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(mu, dtype=np.float32)
    n = mu.shape[0]
    np.savez_compressed(
        prefix.with_suffix(".npz"),
        start_times_sec=np.arange(n, dtype=np.float64) * float(hop),
        window_length_sec=np.asarray(0.03, dtype=np.float64),
        hop_length_sec=np.asarray(hop, dtype=np.float64),
        mu=mu,
        logvar=np.zeros_like(mu, dtype=np.float32),
    )
    prefix.with_suffix(".json").write_text(
        json.dumps({"schema_version": "ava_latent_sequence_v1", "clip_id": rel_stem}),
        encoding="utf-8",
    )


def test_analyze_latent_hop_sensitivity_reports_overlap_metrics(tmp_path: Path) -> None:
    coarse = tmp_path / "hop030"
    dense = tmp_path / "hop010"

    coarse_mu = np.column_stack(
        [
            np.arange(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ]
    )
    dense_mu = np.column_stack(
        [
            np.linspace(0.0, 3.0, 10, dtype=np.float32),
            np.sin(np.linspace(0.0, 1.0, 10, dtype=np.float32)),
        ]
    )
    _write_clip(coarse, "bird1/clip_a", 0.03, coarse_mu)
    _write_clip(dense, "bird1/clip_a", 0.01, dense_mu)
    _write_clip(dense, "bird1/clip_b", 0.01, dense_mu)

    out_json = tmp_path / "hop_sensitivity.json"
    summary_csv = tmp_path / "summary.csv"
    clips_csv = tmp_path / "clips.csv"

    subprocess.run(
        [
            sys.executable,
            SCRIPT.as_posix(),
            "--latent-dir",
            f"hop030={coarse.as_posix()}",
            "--latent-dir",
            f"hop010={dense.as_posix()}",
            "--out-json",
            out_json.as_posix(),
            "--summary-csv",
            summary_csv.as_posix(),
            "--clips-csv",
            clips_csv.as_posix(),
        ],
        check=True,
        cwd=ROOT,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summaries = {row["label"]: row for row in payload["summaries"]}

    assert payload["n_labels"] == 2
    assert payload["common_clip_count"] == 1
    assert summaries["hop030"]["n_clips"] == 1
    assert summaries["hop010"]["n_clips"] == 2
    assert summaries["hop030"]["median_hop_length_sec"] == 0.03
    assert summaries["hop010"]["median_hop_length_sec"] == 0.01
    assert summaries["hop010"]["median_lag1_autocorr"] is not None
    assert 0 < summaries["hop010"]["median_ar1_effective_fraction"] <= 1
    assert summary_csv.exists()
    assert clips_csv.exists()
