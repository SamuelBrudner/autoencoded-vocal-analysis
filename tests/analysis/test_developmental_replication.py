import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ava.analysis.developmental_replication import (
	equalize_events_per_dph,
	ingest_branch_report_metrics,
	run_developmental_replication_analysis,
	select_top_longitudinal_birds,
	summarize_cross_bird_metrics,
)


def _synthetic_branch_records(bird_id: str = "PKA", regime: str = "bells") -> list[dict]:
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
						"event_id": f"{bird_id}-{dph}-{branch}-{idx}",
						"clip_id": f"{bird_id}/clip-{dph}-{branch}-{idx}",
						"clip_stem": f"clip-{dph}-{branch}-{idx}",
						"audio_dir_rel": f"{regime}/{bird_id}/{dph}",
						"bird_id_norm": bird_id,
						"regime": regime,
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


def _manifest() -> dict:
	return {
		"train": [
			{
				"bird_id_norm": "PK249",
				"regime": "bells",
				"dph": 33,
				"num_files": 10,
			},
			{
				"bird_id_norm": "PK249",
				"regime": "bells",
				"dph": 85,
				"num_files": 10,
			},
			{
				"bird_id_norm": "R1",
				"regime": "bells",
				"dph": 35,
				"num_files": 50,
			},
			{
				"bird_id_norm": "R1",
				"regime": "bells",
				"dph": 90,
				"num_files": 60,
			},
			{
				"bird_id_norm": "R2",
				"regime": "samba",
				"dph": 40,
				"num_files": 40,
			},
			{
				"bird_id_norm": "R2",
				"regime": "samba",
				"dph": 82,
				"num_files": 40,
			},
			{
				"bird_id_norm": "R3",
				"regime": "simple",
				"dph": 70,
				"num_files": 500,
			},
		],
		"test": [],
	}


def test_top_longitudinal_selection_is_deterministic():
	selected = select_top_longitudinal_birds(
		_manifest(),
		include_bird="PK249",
		n_non_include=2,
		early_dph_max=45,
		late_dph_min=80,
	)

	assert selected == ["PK249", "R1", "R2"]


def test_cross_bird_summary_and_bootstrap_signs():
	metric_payload = {
		"branch_confidence": {"spearman_rho": 0.8, "bootstrap_rho_ci95": [0.5, 1.0]},
		"branch_entropy": {"spearman_rho": -0.7, "bootstrap_rho_ci95": [-1.0, -0.4]},
		"branch_nearest_distance": {"spearman_rho": -0.6, "bootstrap_rho_ci95": [-0.9, -0.2]},
		"branch_distance_margin": {"spearman_rho": 0.5, "bootstrap_rho_ci95": [0.1, 0.9]},
		"branch_probability_margin": {"spearman_rho": 0.4, "bootstrap_rho_ci95": [0.1, 0.8]},
		"branch_within_standardized_distance": {"spearman_rho": -0.5, "bootstrap_rho_ci95": [-0.9, -0.1]},
	}
	per_bird = {
		"B1": {
			"bird_id": "B1",
			"status": "analyzed",
			"manifest": {"regimes": ["bells"]},
			"primary_metrics": metric_payload,
			"negative_controls": {
				"poincare_radius": {
					"radius_age": {"spearman_rho": -0.2, "bootstrap_rho_ci95": [-0.6, 0.2]}
				}
			},
		},
		"B2": {
			"bird_id": "B2",
			"status": "analyzed",
			"manifest": {"regimes": ["samba"]},
			"primary_metrics": metric_payload,
			"negative_controls": {
				"poincare_radius": {
					"radius_age": {"spearman_rho": 0.1, "bootstrap_rho_ci95": [-0.3, 0.5]}
				}
			},
		},
	}

	cross = summarize_cross_bird_metrics(
		per_bird,
		bootstrap_iterations=20,
		seed=0,
		total_expected_birds=2,
	)

	assert cross["branch_metrics"]["branch_confidence"]["expected_sign_count"] == 2
	assert cross["branch_metrics"]["branch_entropy"]["expected_sign_count"] == 2
	assert cross["negative_controls"]["poincare_radius"]["n_birds"] == 2
	assert set(cross["regime_stratified"]) == {"bells", "samba"}


def test_equalize_events_per_dph_is_deterministic():
	records = _synthetic_branch_records()
	records.extend(_synthetic_branch_records())

	first = equalize_events_per_dph(records, seed=4)
	second = equalize_events_per_dph(records, seed=4)

	assert [row["event_id"] for row in first] == [row["event_id"] for row in second]
	counts = {}
	for row in first:
		counts[row["dph"]] = counts.get(row["dph"], 0) + 1
	assert len(set(counts.values())) == 1


def test_replication_reports_missing_inputs_without_crashing(tmp_path: Path):
	event_table = tmp_path / "pka_events.parquet"
	_write_event_table(event_table, _synthetic_branch_records("PKA"))
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

	result = run_developmental_replication_analysis(
		artifact_dir=tmp_path / "artifacts",
		report_path=tmp_path / "report.md",
		manifest_path=manifest_path,
		bird_ids=["PKA", "PKB"],
		cohort="explicit",
		event_tables={"PKA": event_table},
		event_table_roots=[],
		cluster_min_k=2,
		cluster_max_k=2,
		bootstrap_iterations=10,
		seed=2,
		skip_radius=True,
		skip_bias_sensitivity=True,
	)

	assert result["input_inventory"]["birds"]["PKA"]["status"] == "analyzed"
	assert result["input_inventory"]["birds"]["PKB"]["status"] == "missing_inputs"
	assert Path(result["artifacts"]["input_inventory"]).exists()
	assert Path(result["artifacts"]["per_bird_metrics"]).exists()
	assert (tmp_path / "report.md").exists()


def test_developmental_replication_cli_event_table_smoke(tmp_path: Path):
	repo_root = Path(__file__).resolve().parents[2]
	pka = tmp_path / "pka.parquet"
	pkb = tmp_path / "pkb.parquet"
	_write_event_table(pka, _synthetic_branch_records("PKA", "bells"))
	_write_event_table(pkb, _synthetic_branch_records("PKB", "samba"))
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	out_dir = tmp_path / "artifacts"
	report_path = tmp_path / "report.md"

	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "analyze_developmental_replication.py"),
			"--manifest",
			str(manifest_path),
			"--cohort",
			"explicit",
			"--bird-ids",
			"PKA,PKB",
			"--event-table",
			f"PKA={pka}",
			"--event-table",
			f"PKB={pkb}",
			"--event-table-root",
			str(tmp_path / "empty"),
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
			"--skip-radius",
			"--skip-bias-sensitivity",
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

	assert (out_dir / "cohort.json").exists()
	assert (out_dir / "input_inventory.json").exists()
	assert (out_dir / "per_bird_metrics.json").exists()
	assert (out_dir / "cross_bird_metrics.json").exists()
	assert (out_dir / "figures" / "per_bird_rho_forest.png").exists()
	assert (out_dir / "figures" / "per_bird_rho_forest.pdf").exists()
	assert report_path.exists()
	cross = json.loads((out_dir / "cross_bird_metrics.json").read_text(encoding="utf-8"))
	assert cross["n_analyzed_birds"] == 2


def test_existing_pk249_branch_metrics_schema_can_be_ingested():
	repo_root = Path(__file__).resolve().parents[2]
	metrics_path = (
		repo_root
		/ "docs"
		/ "runs"
		/ "artifacts"
		/ "autoencoded-vocal-analysis-obi.4"
		/ "20260508-155549-branch-commitment-autoencoded-vocal-analysis-obi.4"
		/ "metrics.json"
	)
	if not metrics_path.exists():
		pytest.skip("PK249 branch report metrics are not present in this checkout.")

	metrics = ingest_branch_report_metrics(metrics_path)

	assert metrics["status"] == "ingested"
	assert metrics["branch_clusters"]["best_k"] >= 1
	assert "branch_confidence" in metrics["primary_metrics"]
