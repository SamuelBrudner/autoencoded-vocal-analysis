import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ava.analysis.shotgun_development import (
	compare_replication_metrics,
	filter_manifest_by_birds_and_dph,
	write_shotgun_config,
	write_shotgun_manifests,
)


def _manifest() -> dict:
	return {
		"train": [
			{"bird_id_norm": "PK249", "regime": "bells", "dph": 33, "num_files": 2, "audio_dir_rel": "bells/PK249/33"},
			{"bird_id_norm": "PK249", "regime": "bells", "dph": 91, "num_files": 2, "audio_dir_rel": "bells/PK249/91"},
			{"bird_id_norm": "R426", "regime": "bells", "dph": 40, "num_files": 5, "audio_dir_rel": "bells/R426/40"},
			{"bird_id_norm": "R426", "regime": "bells", "dph": 88, "num_files": 5, "audio_dir_rel": "bells/R426/88"},
			{"bird_id_norm": "R999", "regime": "bells", "dph": 40, "num_files": 99, "audio_dir_rel": "bells/R999/40"},
		],
		"test": [
			{"bird_id_norm": "R426", "regime": "bells", "dph": 80, "num_files": 3, "audio_dir_rel": "bells/R426/80"},
		],
	}


def test_filter_manifest_by_birds_and_dph_is_deterministic():
	filtered = filter_manifest_by_birds_and_dph(
		_manifest(),
		["PK249", "R426"],
		dph_min=33,
		dph_max=90,
	)

	assert [row["audio_dir_rel"] for row in filtered["train"]] == [
		"bells/PK249/33",
		"bells/R426/40",
		"bells/R426/88",
	]
	assert [row["audio_dir_rel"] for row in filtered["test"]] == ["bells/R426/80"]


def test_write_shotgun_manifests_and_config(tmp_path: Path):
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	repo_root = Path(__file__).resolve().parents[2]

	summary = write_shotgun_manifests(
		manifest_path=manifest_path,
		out_dir=tmp_path / "out",
		cohort_birds=["PK249", "R426"],
	)
	config = write_shotgun_config(
		base_config=repo_root / "examples" / "configs" / "fixed_window_finch_30ms_44k.yaml",
		out_path=tmp_path / "out" / "shotgun_config.yaml",
		epochs=10,
	)

	pilot = json.loads(Path(summary["artifacts"]["pilot_manifest"]).read_text(encoding="utf-8"))
	cohort = json.loads(Path(summary["artifacts"]["cohort_manifest"]).read_text(encoding="utf-8"))
	assert len(pilot["train"]) == 1
	assert len(cohort["train"]) == 3
	assert config["overrides"]["preprocess.min_freq"] == 300.0
	assert config["overrides"]["training.epochs"] == 10
	assert (tmp_path / "out" / "shotgun_config.yaml").exists()


def _per_bird(conf: float, entropy: float, distance: float, margin: float, radius: float) -> dict:
	return {
		"B1": {
			"bird_id": "B1",
			"status": "analyzed",
			"events_analyzed": 100,
			"primary_metrics": {
				"branch_confidence": {"spearman_rho": conf},
				"branch_entropy": {"spearman_rho": entropy},
				"branch_nearest_distance": {"spearman_rho": distance},
				"branch_distance_margin": {"spearman_rho": margin},
				"branch_probability_margin": {"spearman_rho": margin},
				"branch_within_standardized_distance": {"spearman_rho": distance},
			},
			"negative_controls": {
				"poincare_radius": {"radius_age": {"spearman_rho": radius}}
			},
			"coverage_bias": {"counts_available": True},
		}
	}


def test_compare_replication_metrics_classifies_strengthening_and_weakening():
	baseline = _per_bird(0.5, -0.5, -0.5, 0.5, -0.2)
	stronger = _per_bird(0.8, -0.8, -0.8, 0.8, -0.4)
	weaker = _per_bird(0.2, 0.1, -0.1, 0.1, 0.2)

	strong = compare_replication_metrics(baseline, stronger)
	weak = compare_replication_metrics(baseline, weaker)

	assert "strengthens" in strong["conclusion"]
	assert "weakens" in weak["conclusion"]
	assert strong["mean_expected_direction_delta"] > 0


def test_compare_shotgun_development_cli_smoke(tmp_path: Path):
	repo_root = Path(__file__).resolve().parents[2]
	baseline = tmp_path / "baseline.json"
	shotgun = tmp_path / "shotgun.json"
	baseline.write_text(json.dumps(_per_bird(0.5, -0.5, -0.5, 0.5, -0.2)), encoding="utf-8")
	shotgun.write_text(json.dumps(_per_bird(0.8, -0.8, -0.8, 0.8, -0.4)), encoding="utf-8")
	out_dir = tmp_path / "out"
	report = tmp_path / "report.md"

	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "compare_shotgun_development.py"),
			"--baseline-per-bird",
			str(baseline),
			"--shotgun-per-bird",
			str(shotgun),
			"--out-dir",
			str(out_dir),
			"--report-out",
			str(report),
		],
		cwd=repo_root,
		capture_output=True,
		text=True,
	)
	if result.returncode != 0:
		raise AssertionError(f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

	assert (out_dir / "shotgun_comparison_metrics.json").exists()
	assert (out_dir / "figures" / "branch_metric_deltas.png").exists()
	assert (out_dir / "figures" / "branch_metric_deltas.pdf").exists()
	assert report.exists()


def test_shotgun_replication_cli_full_rebuild_smoke(tmp_path: Path):
	pytest.importorskip("pyarrow")
	import pyarrow as pa
	import pyarrow.parquet as pq

	repo_root = Path(__file__).resolve().parents[2]
	manifest = {"train": [], "test": []}
	latent_root = tmp_path / "latent"
	audio_root = tmp_path / "audio"
	roi_root = tmp_path / "roi"

	for dph, bases in {
		33: ([2.5, 2.5, 0.0, 0.0], [2.7, 2.3, 0.0, 0.0]),
		40: ([2.0, 2.7, 0.0, 0.0], [2.8, 2.0, 0.0, 0.0]),
		80: ([5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]),
		85: ([5.4, 0.0, 0.0, 0.0], [0.0, 5.4, 0.0, 0.0]),
	}.items():
		audio_dir_rel = f"bells/PKS/{dph}"
		audio_dir = audio_root / audio_dir_rel
		roi_dir = roi_root / audio_dir_rel
		audio_dir.mkdir(parents=True)
		roi_dir.mkdir(parents=True)
		(latent_root / audio_dir_rel).mkdir(parents=True)
		stems = []
		onsets = []
		offsets = []
		for idx, base in enumerate(bases):
			stem = f"clip_{dph}_{idx}"
			(audio_dir / f"{stem}.wav").write_bytes(b"")
			stems.append(stem)
			onsets.append([0.0])
			offsets.append([0.11])
			clip_id = f"{audio_dir_rel}/{stem}"
			mu = np.stack(
				[
					np.asarray(base, dtype=np.float32),
					np.asarray(base, dtype=np.float32) + 0.01,
				],
				axis=0,
			)
			np.savez_compressed(
				latent_root / f"{clip_id}.npz",
				start_times_sec=np.asarray([0.0, 0.05], dtype=np.float64),
				window_length_sec=np.asarray(0.05, dtype=np.float64),
				hop_length_sec=np.asarray(0.05, dtype=np.float64),
				mu=mu.astype(np.float32),
				logvar=np.zeros((2, 4), dtype=np.float32),
				energy=np.ones(2, dtype=np.float32),
			)
			(latent_root / f"{clip_id}.json").write_text(
				json.dumps({"schema_version": "ava_latent_sequence_v1"}),
				encoding="utf-8",
			)
		pq.write_table(
			pa.table({"clip_stem": stems, "onsets_sec": onsets, "offsets_sec": offsets}),
			roi_dir / "roi.parquet",
		)
		manifest["train"].append(
			{
				"audio_dir_rel": audio_dir_rel,
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"bird_id_norm": "PKS",
				"regime": "bells",
				"dph": dph,
				"num_files": 2,
				"split": "train",
			}
		)

	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
	out_dir = tmp_path / "artifacts"
	report = tmp_path / "report.md"
	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "analyze_developmental_replication.py"),
			"--manifest",
			str(manifest_path),
			"--cohort",
			"explicit",
			"--bird-ids",
			"PKS",
			"--latent-root",
			str(latent_root),
			"--audio-root",
			str(audio_root),
			"--roi-root",
			str(roi_root),
			"--latent-model-id",
			"shotgun_vae",
			"--out-dir",
			str(out_dir),
			"--report-out",
			str(report),
			"--cluster-min-k",
			"2",
			"--cluster-max-k",
			"2",
			"--bootstrap-iterations",
			"5",
			"--skip-radius",
			"--skip-bias-sensitivity",
		],
		cwd=repo_root,
		capture_output=True,
		text=True,
	)
	if result.returncode != 0:
		raise AssertionError(f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

	per_bird = json.loads((out_dir / "per_bird_metrics.json").read_text(encoding="utf-8"))
	assert per_bird["PKS"]["latent_model_id"] == "shotgun_vae"
	assert per_bird["PKS"]["status"] == "analyzed"
