import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ava.analysis.branch_commitment import (
	analyze_branch_records,
	fit_branch_clusters,
	load_event_records_parquet,
)
from ava.analysis.hyperbolic_development import latent_matrix, standardize_latents


def _synthetic_branch_records() -> list[dict]:
	records = []
	templates = {
		33: ([2.5, 2.5, 0.0, 0.0], [2.7, 2.3, 0.0, 0.0]),
		40: ([2.0, 2.7, 0.0, 0.0], [2.8, 2.0, 0.0, 0.0]),
		80: ([5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]),
		85: ([5.4, 0.0, 0.0, 0.0], [0.0, 5.4, 0.0, 0.0]),
	}
	for dph, bases in templates.items():
		for branch, base in enumerate(bases):
			for idx in range(3):
				offset = 0.02 * idx
				mu = np.asarray(base, dtype=np.float32).copy()
				mu[branch] += offset
				records.append(
					{
						"event_id": f"{dph}-{branch}-{idx}",
						"clip_id": f"clip-{dph}-{branch}-{idx}",
						"clip_stem": f"clip-{dph}-{branch}-{idx}",
						"audio_dir_rel": f"day43 Bells/pk249/{dph}",
						"bird_id_norm": "PK249",
						"regime": "bells",
						"split": "train",
						"dph": float(dph),
						"roi_index": idx,
						"onset_sec": 0.0,
						"offset_sec": 0.1,
						"duration_sec": 0.1 + 0.01 * idx,
						"n_windows": 2 + idx,
						"mean_energy": 1.0 + 0.1 * idx,
						"variance_mean": 0.2,
						"latent_dim": 4,
						"mu": mu.tolist(),
						"variance": [1.0, 1.0, 1.0, 1.0],
						"mu_energy_weighted": None,
					}
				)
	return records


def _write_event_table(path: Path, records: list[dict]) -> None:
	pytest.importorskip("pyarrow")
	import pyarrow as pa
	import pyarrow.parquet as pq

	fields = {
		"event_id": [record["event_id"] for record in records],
		"clip_id": [record["clip_id"] for record in records],
		"clip_stem": [record["clip_stem"] for record in records],
		"audio_dir_rel": [record["audio_dir_rel"] for record in records],
		"bird_id_norm": [record["bird_id_norm"] for record in records],
		"regime": [record["regime"] for record in records],
		"split": [record["split"] for record in records],
		"dph": [record["dph"] for record in records],
		"roi_index": [record["roi_index"] for record in records],
		"duration_sec": [record["duration_sec"] for record in records],
		"n_windows": [record["n_windows"] for record in records],
		"mean_energy": [record["mean_energy"] for record in records],
		"variance_mean": [record["variance_mean"] for record in records],
		"latent_dim": [record["latent_dim"] for record in records],
		"mu": pa.array([record["mu"] for record in records], type=pa.list_(pa.float32())),
		"variance": pa.array(
			[record["variance"] for record in records],
			type=pa.list_(pa.float32()),
		),
		"mu_energy_weighted": pa.array(
			[record["mu_energy_weighted"] for record in records],
			type=pa.list_(pa.float32()),
		),
	}
	pq.write_table(pa.table(fields), path)


def test_event_table_loading_and_latent_matrix(tmp_path: Path):
	event_table = tmp_path / "events.parquet"
	records = _synthetic_branch_records()
	_write_event_table(event_table, records)

	loaded = load_event_records_parquet(
		event_table,
		bird_id="PK249",
		dph_min=35,
		dph_max=85,
	)
	x = latent_matrix(loaded)

	assert {record["dph"] for record in loaded} == {40.0, 80.0, 85.0}
	assert x.shape == (18, 4)
	assert loaded[0]["bird_id_norm"] == "PK249"


def test_branch_metrics_capture_known_commitment_signal():
	records = _synthetic_branch_records()

	result = analyze_branch_records(
		records,
		early_dph_max=45,
		late_dph_min=80,
		cluster_min_k=2,
		cluster_max_k=2,
		bootstrap_iterations=25,
		seed=7,
		run_sensitivity=True,
	)
	primary = result.metrics["primary_metrics"]

	assert result.clusters.best_k == 2
	assert primary["branch_confidence"]["spearman_rho"] > 0
	assert primary["branch_entropy"]["spearman_rho"] < 0
	assert primary["branch_nearest_distance"]["spearman_rho"] < 0
	assert primary["branch_distance_margin"]["spearman_rho"] > 0
	assert result.metrics["sensitivity"]["replicates_expected_signs"]["by_metric"][
		"branch_confidence"
	]["replicates"]


def test_branch_commitment_is_deterministic_with_fixed_seed():
	records = _synthetic_branch_records()

	first = analyze_branch_records(
		records,
		late_dph_min=80,
		cluster_min_k=2,
		cluster_max_k=2,
		bootstrap_iterations=10,
		seed=11,
		run_sensitivity=False,
	)
	second = analyze_branch_records(
		records,
		late_dph_min=80,
		cluster_min_k=2,
		cluster_max_k=2,
		bootstrap_iterations=10,
		seed=11,
		run_sensitivity=False,
	)

	assert first.clusters.best_k == second.clusters.best_k
	assert np.array_equal(first.clusters.labels, second.clusters.labels)
	assert (
		first.metrics["primary_metrics"]["branch_confidence"]["spearman_rho"]
		== second.metrics["primary_metrics"]["branch_confidence"]["spearman_rho"]
	)


def test_branch_commitment_cli_event_table_smoke(tmp_path: Path):
	pytest.importorskip("pyarrow")
	repo_root = Path(__file__).resolve().parents[2]
	event_table = tmp_path / "events.parquet"
	_write_event_table(event_table, _synthetic_branch_records())
	out_dir = tmp_path / "artifacts"
	report_path = tmp_path / "report.md"

	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "analyze_branch_commitment.py"),
			"--event-table",
			str(event_table),
			"--out-dir",
			str(out_dir),
			"--report-out",
			str(report_path),
			"--cluster-min-k",
			"2",
			"--cluster-max-k",
			"2",
			"--bootstrap-iterations",
			"10",
			"--no-sensitivity",
		],
		cwd=repo_root,
		capture_output=True,
		text=True,
	)
	if result.returncode != 0:
		raise AssertionError(
			"CLI failed.\n"
			f"stdout:\n{result.stdout}\n"
			f"stderr:\n{result.stderr}"
		)

	assert (out_dir / "branch_commitment.parquet").exists()
	assert (out_dir / "branch_clusters.json").exists()
	assert (out_dir / "metrics.json").exists()
	assert (out_dir / "figures" / "branch_confidence_entropy.png").exists()
	assert report_path.exists()
	metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
	assert metrics["branch_clusters"]["best_k"] == 2
	assert metrics["inputs"]["source_mode"] == "event_table"


def test_branch_commitment_cli_full_rebuild_smoke(tmp_path: Path):
	pytest.importorskip("pyarrow")
	import pyarrow as pa
	import pyarrow.parquet as pq

	repo_root = Path(__file__).resolve().parents[2]
	manifest = {"train": [], "test": []}
	latent_root = tmp_path / "latent"
	audio_root = tmp_path / "audio"
	roi_root = tmp_path / "roi"
	latent_root.mkdir()

	for dph, bases in {
		33: ([2.5, 2.5, 0.0, 0.0], [2.7, 2.3, 0.0, 0.0]),
		40: ([2.0, 2.7, 0.0, 0.0], [2.8, 2.0, 0.0, 0.0]),
		80: ([5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]),
		85: ([5.4, 0.0, 0.0, 0.0], [0.0, 5.4, 0.0, 0.0]),
	}.items():
		audio_dir_rel = f"day43 Bells/pk249/{dph}"
		audio_dir = audio_root / audio_dir_rel
		roi_dir = roi_root / audio_dir_rel
		audio_dir.mkdir(parents=True)
		roi_dir.mkdir(parents=True)
		latent_dir = latent_root / audio_dir_rel
		latent_dir.mkdir(parents=True)
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
			pa.table(
				{
					"clip_stem": stems,
					"onsets_sec": onsets,
					"offsets_sec": offsets,
				}
			),
			roi_dir / "roi.parquet",
		)
		manifest["train"].append(
			{
				"audio_dir_rel": audio_dir_rel,
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"bird_id_norm": "PK249",
				"bird_id_raw": "pk249",
				"regime": "bells",
				"dph": dph,
				"session_label": None,
				"num_files": 2,
				"split": "train",
			}
		)

	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
	out_dir = tmp_path / "artifacts"
	report_path = tmp_path / "report.md"

	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "analyze_branch_commitment.py"),
			"--manifest",
			str(manifest_path),
			"--latent-root",
			str(latent_root),
			"--out-dir",
			str(out_dir),
			"--report-out",
			str(report_path),
			"--cluster-min-k",
			"2",
			"--cluster-max-k",
			"2",
			"--bootstrap-iterations",
			"10",
			"--no-sensitivity",
		],
		cwd=repo_root,
		capture_output=True,
		text=True,
	)
	if result.returncode != 0:
		raise AssertionError(
			"CLI failed.\n"
			f"stdout:\n{result.stdout}\n"
			f"stderr:\n{result.stderr}"
		)

	assert (out_dir / "branch_commitment.parquet").exists()
	assert (out_dir / "branch_clusters.json").exists()
	assert (out_dir / "metrics.json").exists()
	assert report_path.exists()
	metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
	assert metrics["inputs"]["source_mode"] == "full_rebuild"
	assert metrics["event_summary"]["clips_seen"] == 8
