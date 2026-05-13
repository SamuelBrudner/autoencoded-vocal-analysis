"""Multi-bird developmental branch-commitment replication analyses."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from ava.analysis.branch_commitment import (
	PRIMARY_METRICS,
	analyze_branch_records,
	grouped_metric_summary,
	load_event_records_parquet,
	load_event_summary_for_event_table,
)
from ava.analysis.hyperbolic_development import (
	attach_embedding_coordinates,
	build_event_records_from_manifest,
	fit_poincare_embedding,
	latent_matrix,
	load_manifest,
	radius_age_metrics,
	select_balanced_indices,
	standardize_latents,
	write_json,
)


DEFAULT_REPLICATION_BEAD = "autoencoded-vocal-analysis-obi.4"
DEFAULT_BIRD_IDS = (
	"PK249",
	"R426",
	"R467",
	"R404",
	"R150",
	"R493",
	"R470",
	"R203",
	"R425",
	"R229",
	"R122",
)
CRITERION_METRICS = (
	"branch_confidence",
	"branch_entropy",
	"branch_nearest_distance",
	"branch_distance_margin",
)
NEGATIVE_CONTROL_METRIC = "poincare_radius"
CONTROL_FIELDS = ("duration_sec", "n_windows", "mean_energy", "variance_mean")


@dataclass
class ReplicationRunResult:
	"""Paths and summary metrics for a replication run."""

	cohort: dict
	input_inventory: dict
	per_bird_metrics: dict
	cross_bird_metrics: dict
	figures: dict[str, str]
	artifacts: dict[str, str]
	report_path: str


def run_developmental_replication_analysis(
	artifact_dir: Path,
	report_path: Path,
	manifest_path: Path,
	bird_ids: Sequence[str] = DEFAULT_BIRD_IDS,
	cohort: str = "top-longitudinal",
	latent_model_id: str = "ava_latent",
	event_tables: Optional[dict[str, Path]] = None,
	event_table_roots: Optional[Sequence[Path]] = None,
	latent_root: Optional[Path] = None,
	audio_root: Optional[Path] = None,
	roi_root: Optional[Path] = None,
	split: str = "all",
	dph_min: float = 33,
	dph_max: float = 90,
	early_dph_max: float = 45,
	late_dph_min: float = 80,
	roi_format: str = "parquet",
	roi_parquet_name: str = "roi.parquet",
	max_events_per_dph: Optional[int] = 2000,
	cluster_min_k: int = 4,
	cluster_max_k: int = 12,
	bootstrap_iterations: int = 1000,
	seed: int = 0,
	use_energy_weighted: bool = False,
	embedding_max_events: int = 6000,
	embedding_knn: int = 10,
	embedding_epochs: int = 300,
	embedding_lr: float = 0.01,
	embedding_max_edges: int = 200_000,
	skip_radius: bool = False,
	skip_bias_sensitivity: bool = False,
) -> dict:
	"""Run the fixed-cohort multi-bird branch-commitment replication."""
	artifact_dir.mkdir(parents=True, exist_ok=True)
	fig_dir = artifact_dir / "figures"
	fig_dir.mkdir(parents=True, exist_ok=True)

	manifest = load_manifest(manifest_path)
	requested_birds = tuple(_normalize_bird_id(bird) for bird in bird_ids)
	cohort_payload = build_cohort_payload(
		manifest=manifest,
		bird_ids=requested_birds,
		cohort=cohort,
		latent_model_id=latent_model_id,
		dph_min=dph_min,
		dph_max=dph_max,
		early_dph_max=early_dph_max,
		late_dph_min=late_dph_min,
	)
	explicit_tables = {
		_normalize_bird_id(bird): Path(path)
		for bird, path in (event_tables or {}).items()
	}
	discovered_tables = discover_event_tables(
		bird_ids=requested_birds,
		event_table_roots=event_table_roots or (),
		explicit_event_tables=explicit_tables,
		dph_min=dph_min,
		dph_max=dph_max,
	)

	inventory: dict[str, Any] = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"manifest_path": manifest_path.as_posix(),
		"latent_model_id": str(latent_model_id),
		"event_table_roots": [
			Path(root).as_posix() for root in (event_table_roots or ())
		],
		"latent_root": None if latent_root is None else latent_root.as_posix(),
		"roi_root": None if roi_root is None else roi_root.as_posix(),
		"birds": {},
	}
	per_bird: dict[str, Any] = {}

	for bird_id in requested_birds:
		source = discovered_tables.get(bird_id)
		inventory["birds"][bird_id] = {
			"bird_id": bird_id,
			"manifest": cohort_payload["birds"].get(bird_id, {}),
			"status": "pending",
			"source_mode": None,
			"event_table": None,
			"coverage_metrics": None,
			"reason": None,
		}
		try:
			records, event_summary, inputs = _load_or_build_bird_records(
				bird_id=bird_id,
				source=source,
				manifest=manifest,
				manifest_path=manifest_path,
				latent_root=latent_root,
				audio_root=audio_root,
				roi_root=roi_root,
				split=split,
				dph_min=dph_min,
				dph_max=dph_max,
				roi_format=roi_format,
				roi_parquet_name=roi_parquet_name,
				max_events_per_dph=max_events_per_dph,
				seed=seed,
			)
		except FileNotFoundError as exc:
			inventory["birds"][bird_id].update(
				{
					"status": "missing_inputs",
					"reason": str(exc),
				}
			)
			per_bird[bird_id] = _missing_bird_metrics(
				bird_id,
				cohort_payload["birds"].get(bird_id, {}),
				str(exc),
			)
			continue
		except Exception as exc:
			inventory["birds"][bird_id].update(
				{
					"status": "failed",
					"reason": str(exc),
				}
			)
			per_bird[bird_id] = _missing_bird_metrics(
				bird_id,
				cohort_payload["birds"].get(bird_id, {}),
				str(exc),
				status="failed",
			)
			continue

		inventory["birds"][bird_id].update(
			{
				"status": "loaded",
				"source_mode": inputs["source_mode"],
				"event_table": inputs.get("event_table"),
				"coverage_metrics": inputs.get("coverage_metrics"),
				"events_loaded": len(records),
			}
		)
		per_bird[bird_id] = analyze_replication_bird(
			bird_id=bird_id,
			records=records,
			event_summary=event_summary,
			inputs=inputs,
			manifest_summary=cohort_payload["birds"].get(bird_id, {}),
			latent_model_id=latent_model_id,
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
			use_energy_weighted=use_energy_weighted,
			embedding_max_events=embedding_max_events,
			embedding_knn=embedding_knn,
			embedding_epochs=embedding_epochs,
			embedding_lr=embedding_lr,
			embedding_max_edges=embedding_max_edges,
			skip_radius=skip_radius,
			skip_bias_sensitivity=skip_bias_sensitivity,
		)
		inventory["birds"][bird_id]["status"] = per_bird[bird_id]["status"]

	cross_bird = summarize_cross_bird_metrics(
		per_bird,
		bootstrap_iterations=bootstrap_iterations,
		seed=seed,
		total_expected_birds=len(requested_birds),
	)
	artifacts = {
		"cohort": (artifact_dir / "cohort.json").as_posix(),
		"input_inventory": (artifact_dir / "input_inventory.json").as_posix(),
		"per_bird_metrics": (artifact_dir / "per_bird_metrics.json").as_posix(),
		"cross_bird_metrics": (artifact_dir / "cross_bird_metrics.json").as_posix(),
	}
	write_json(Path(artifacts["cohort"]), cohort_payload)
	write_json(Path(artifacts["input_inventory"]), inventory)
	write_json(Path(artifacts["per_bird_metrics"]), per_bird)
	write_json(Path(artifacts["cross_bird_metrics"]), cross_bird)
	figures = make_replication_figures(per_bird, cross_bird, inventory, fig_dir)
	write_replication_report(
		report_path=report_path,
		cohort=cohort_payload,
		inventory=inventory,
		per_bird=per_bird,
		cross_bird=cross_bird,
		figures=figures,
		artifacts=artifacts,
	)
	return {
		"cohort": cohort_payload,
		"input_inventory": inventory,
		"per_bird_metrics": per_bird,
		"cross_bird_metrics": cross_bird,
		"figures": figures,
		"artifacts": artifacts,
		"report_path": report_path.as_posix(),
	}


def build_cohort_payload(
	manifest: dict,
	bird_ids: Sequence[str],
	cohort: str,
	latent_model_id: str,
	dph_min: float,
	dph_max: float,
	early_dph_max: float,
	late_dph_min: float,
) -> dict:
	"""Return deterministic manifest coverage summaries for a requested cohort."""
	birds = {
		_normalize_bird_id(bird): summarize_manifest_bird(
			manifest,
			_normalize_bird_id(bird),
			dph_min=dph_min,
			dph_max=dph_max,
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
		)
		for bird in bird_ids
	}
	return {
		"cohort": cohort,
		"latent_model_id": str(latent_model_id),
		"bird_ids": list(birds),
		"dph_min": float(dph_min),
		"dph_max": float(dph_max),
		"early_dph_max": float(early_dph_max),
		"late_dph_min": float(late_dph_min),
		"birds": birds,
	}


def select_top_longitudinal_birds(
	manifest: dict,
	include_bird: str = "PK249",
	n_non_include: int = 10,
	early_dph_max: float = 45,
	late_dph_min: float = 80,
) -> list[str]:
	"""Select PK249 plus the top non-PK249 birds with finite early/late coverage."""
	include = _normalize_bird_id(include_bird)
	summaries = {}
	for row in _iter_manifest_rows(manifest):
		bird = _normalize_bird_id(row.get("bird_id_norm") or row.get("bird_id_raw"))
		if not bird:
			continue
		summaries.setdefault(
			bird,
			{
				"num_files": 0,
				"dph_values": set(),
				"row_count": 0,
			},
		)
		dph = _coerce_float(row.get("dph"))
		if dph is not None and math.isfinite(dph):
			summaries[bird]["dph_values"].add(float(dph))
		summaries[bird]["num_files"] += int(row.get("num_files") or 0)
		summaries[bird]["row_count"] += 1

	ranked = []
	for bird, summary in summaries.items():
		dph_values = summary["dph_values"]
		has_early = any(dph <= float(early_dph_max) for dph in dph_values)
		has_late = any(dph >= float(late_dph_min) for dph in dph_values)
		if bird == include or not (has_early and has_late):
			continue
		ranked.append(
			(
				-int(summary["num_files"]),
				-int(len(dph_values)),
				bird,
			)
		)
	ranked.sort()
	selected = [include]
	selected.extend(bird for _, _, bird in ranked[: int(n_non_include)])
	return selected


def summarize_manifest_bird(
	manifest: dict,
	bird_id: str,
	dph_min: float,
	dph_max: float,
	early_dph_max: float,
	late_dph_min: float,
) -> dict:
	"""Summarize manifest coverage for one bird."""
	bird_norm = _normalize_bird_id(bird_id)
	rows = []
	for row in _iter_manifest_rows(manifest):
		bird = _normalize_bird_id(row.get("bird_id_norm") or row.get("bird_id_raw"))
		if bird == bird_norm:
			rows.append(row)
	dph_values = []
	dph_in_range = []
	num_files = 0
	regimes = set()
	splits = set()
	for row in rows:
		num_files += int(row.get("num_files") or 0)
		if row.get("regime"):
			regimes.add(str(row["regime"]))
		if row.get("split"):
			splits.add(str(row["split"]))
		dph = _coerce_float(row.get("dph"))
		if dph is None or not math.isfinite(dph):
			continue
		dph_values.append(float(dph))
		if float(dph_min) <= dph <= float(dph_max):
			dph_in_range.append(float(dph))
	return {
		"bird_id": bird_norm,
		"manifest_rows": int(len(rows)),
		"num_files": int(num_files),
		"dph_min_all": float(min(dph_values)) if dph_values else None,
		"dph_max_all": float(max(dph_values)) if dph_values else None,
		"unique_dph_all": int(len(set(dph_values))),
		"unique_dph_in_range": int(len(set(dph_in_range))),
		"has_early_reference": bool(any(dph <= float(early_dph_max) for dph in dph_values)),
		"has_late_reference": bool(any(dph >= float(late_dph_min) for dph in dph_values)),
		"regimes": sorted(regimes),
		"splits": sorted(splits),
	}


def discover_event_tables(
	bird_ids: Sequence[str],
	event_table_roots: Sequence[Path],
	explicit_event_tables: Optional[dict[str, Path]] = None,
	dph_min: Optional[float] = None,
	dph_max: Optional[float] = None,
) -> dict[str, dict]:
	"""Discover per-bird ``event_latents.parquet`` files under local roots."""
	birds = {_normalize_bird_id(bird) for bird in bird_ids}
	out: dict[str, dict] = {}
	for bird, path in (explicit_event_tables or {}).items():
		bird_norm = _normalize_bird_id(bird)
		if bird_norm in birds:
			out[bird_norm] = {
				"source_mode": "event_table",
				"event_table": Path(path),
				"discovery": "explicit",
			}

	for root in event_table_roots:
		root = Path(root)
		if not root.exists():
			continue
		for path in sorted(root.glob("**/event_latents.parquet")):
			for bird in _event_table_birds(path, dph_min=dph_min, dph_max=dph_max):
				if bird not in birds or bird in out:
					continue
				out[bird] = {
					"source_mode": "event_table",
					"event_table": path,
					"discovery": "root_scan",
				}
	return out


def analyze_replication_bird(
	bird_id: str,
	records: Sequence[dict],
	event_summary: dict,
	inputs: dict,
	manifest_summary: dict,
	latent_model_id: str,
	early_dph_max: float,
	late_dph_min: float,
	cluster_min_k: int,
	cluster_max_k: int,
	bootstrap_iterations: int,
	seed: int,
	use_energy_weighted: bool,
	embedding_max_events: int,
	embedding_knn: int,
	embedding_epochs: int,
	embedding_lr: float,
	embedding_max_edges: int,
	skip_radius: bool = False,
	skip_bias_sensitivity: bool = False,
) -> dict:
	"""Run branch and negative-control metrics for one bird."""
	analysis = analyze_branch_records(
		records,
		event_summary=event_summary,
		inputs=inputs,
		early_dph_max=early_dph_max,
		late_dph_min=late_dph_min,
		cluster_min_k=cluster_min_k,
		cluster_max_k=cluster_max_k,
		bootstrap_iterations=bootstrap_iterations,
		seed=seed,
		use_energy_weighted=use_energy_weighted,
		run_sensitivity=True,
	)
	radius = (
		{"status": "skipped", "reason": "--skip-radius was set"}
		if skip_radius else
		compute_poincare_radius_negative_control(
			records=records,
			early_dph_max=early_dph_max,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
			use_energy_weighted=use_energy_weighted,
			embedding_max_events=embedding_max_events,
			embedding_knn=embedding_knn,
			embedding_epochs=embedding_epochs,
			embedding_lr=embedding_lr,
			embedding_max_edges=embedding_max_edges,
		)
	)
	bias = (
		{"status": "skipped", "reason": "--skip-bias-sensitivity was set"}
		if skip_bias_sensitivity else
		compute_bias_sensitivities(
			records=analysis.records,
			event_summary=event_summary,
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
			use_energy_weighted=use_energy_weighted,
		)
	)
	primary = analysis.metrics["primary_metrics"]
	return {
		"bird_id": _normalize_bird_id(bird_id),
		"latent_model_id": str(latent_model_id),
		"status": "analyzed",
		"manifest": manifest_summary,
		"inputs": inputs,
		"events_analyzed": int(len(analysis.records)),
		"dph_values_analyzed": sorted(
			float(dph) for dph in {float(record["dph"]) for record in analysis.records}
		),
		"branch_clusters": analysis.metrics["branch_clusters"],
		"primary_metrics": primary,
		"early_late_contrasts": analysis.metrics["early_late_contrasts"],
		"sensitivity": analysis.metrics.get("sensitivity", {}),
		"coverage_bias": analysis.metrics["coverage_bias"],
		"controls_by_dph": analysis.metrics["controls_by_dph"],
		"bias_sensitivities": bias,
		"negative_controls": {
			NEGATIVE_CONTROL_METRIC: radius,
		},
		"signs": sign_summary_for_metrics(primary),
	}


def compute_poincare_radius_negative_control(
	records: Sequence[dict],
	early_dph_max: float,
	bootstrap_iterations: int,
	seed: int,
	use_energy_weighted: bool,
	embedding_max_events: int,
	embedding_knn: int,
	embedding_epochs: int,
	embedding_lr: float,
	embedding_max_edges: int,
) -> dict:
	"""Fit a small optimized Poincare embedding and summarize radius-age trend."""
	try:
		if len(records) < 2:
			raise ValueError("At least two events are required for Poincare radius.")
		selected = select_balanced_indices(records, embedding_max_events, seed=seed)
		embedding_records = [dict(records[int(idx)]) for idx in selected]
		x_raw = latent_matrix(embedding_records, use_energy_weighted=use_energy_weighted)
		x_std, _ = standardize_latents(x_raw)
		embedding = fit_poincare_embedding(
			x_std,
			seed=seed,
			knn=embedding_knn,
			epochs=embedding_epochs,
			lr=embedding_lr,
			max_edges=embedding_max_edges,
		)
		attach_embedding_coordinates(
			embedding_records,
			embedding.points,
			early_dph_max=early_dph_max,
		)
		metrics = radius_age_metrics(
			embedding_records,
			radius_key=NEGATIVE_CONTROL_METRIC,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
		)
		return {
			"status": "computed",
			"events_embedded": int(len(embedding_records)),
			"edge_count": int(embedding.edge_count),
			"target_scale": float(embedding.target_scale),
			"loss_initial": float(embedding.loss_history[0]),
			"loss_final": float(embedding.loss_history[-1]),
			"radius_age": metrics,
		}
	except Exception as exc:
		return {
			"status": "failed",
			"reason": str(exc),
		}


def compute_bias_sensitivities(
	records: Sequence[dict],
	event_summary: dict,
	early_dph_max: float,
	late_dph_min: float,
	cluster_min_k: int,
	cluster_max_k: int,
	bootstrap_iterations: int,
	seed: int,
	use_energy_weighted: bool,
) -> dict:
	"""Run skip-rate, equalized-count, and covariate-residualized sensitivity checks."""
	out: dict[str, Any] = {
		"skip_rate_threshold": 0.30,
		"skip_filtered": {},
		"equalized_events_per_dph": {},
		"residualized_controls": {},
	}
	filtered, excluded = filter_high_skip_dph(records, event_summary, threshold=0.30)
	out["skip_filtered"]["excluded_dph"] = excluded
	out["skip_filtered"].update(
		_analyze_sensitivity_records(
			filtered,
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed + 101,
			use_energy_weighted=use_energy_weighted,
		)
	)
	equalized = equalize_events_per_dph(records, seed=seed + 202)
	out["equalized_events_per_dph"].update(
		{
			"events": int(len(equalized)),
			"events_per_dph": _events_per_dph(equalized),
		}
	)
	out["equalized_events_per_dph"].update(
		_analyze_sensitivity_records(
			equalized,
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed + 203,
			use_energy_weighted=use_energy_weighted,
		)
	)
	out["residualized_controls"] = residualized_metric_tests(
		records,
		bootstrap_iterations=bootstrap_iterations,
		seed=seed + 303,
	)
	return out


def filter_high_skip_dph(
	records: Sequence[dict],
	event_summary: dict,
	threshold: float = 0.30,
) -> tuple[list[dict], list[dict]]:
	"""Drop dph bins whose ROI-without-window rate exceeds threshold."""
	rates = _roi_without_window_rates_by_dph(event_summary)
	if not rates:
		return list(records), []
	excluded = [
		{"dph": float(dph), "roi_without_window_rate": float(rate)}
		for dph, rate in sorted(rates.items())
		if rate > float(threshold)
	]
	excluded_values = {float(row["dph"]) for row in excluded}
	filtered = [
		dict(record)
		for record in records
		if float(record.get("dph", float("nan"))) not in excluded_values
	]
	return filtered, excluded


def equalize_events_per_dph(records: Sequence[dict], seed: int = 0) -> list[dict]:
	"""Return a deterministic subset with equal event count per dph."""
	by_dph: dict[float, list[int]] = defaultdict(list)
	for idx, record in enumerate(records):
		dph = _coerce_float(record.get("dph"))
		if dph is not None and math.isfinite(dph):
			by_dph[float(dph)].append(idx)
	if not by_dph:
		return []
	keep_per_dph = min(len(indices) for indices in by_dph.values())
	rng = np.random.default_rng(int(seed))
	selected = []
	for dph in sorted(by_dph):
		indices = np.asarray(by_dph[dph], dtype=np.int64)
		if len(indices) <= keep_per_dph:
			chosen = indices
		else:
			chosen = rng.choice(indices, size=keep_per_dph, replace=False)
		selected.extend(int(i) for i in chosen)
	return [dict(records[idx]) for idx in sorted(selected)]


def residualized_metric_tests(
	records: Sequence[dict],
	bootstrap_iterations: int,
	seed: int,
) -> dict:
	"""Residualize branch metrics against basic event controls and retest dph trends."""
	out = {
		"controls": list(CONTROL_FIELDS),
		"primary_metrics": {},
	}
	for metric in PRIMARY_METRICS:
		residual_records = _attach_residuals(records, metric, CONTROL_FIELDS)
		residual_key = f"{metric}_residual"
		if not residual_records:
			out["primary_metrics"][metric] = {
				"status": "failed",
				"reason": "No finite rows after control filtering.",
			}
			continue
		try:
			payload = radius_age_metrics(
				residual_records,
				radius_key=residual_key,
				bootstrap_iterations=bootstrap_iterations,
				seed=seed,
			)
			payload["status"] = "computed"
			out["primary_metrics"][metric] = payload
		except Exception as exc:
			out["primary_metrics"][metric] = {
				"status": "failed",
				"reason": str(exc),
			}
	return out


def summarize_cross_bird_metrics(
	per_bird_metrics: dict[str, dict],
	bootstrap_iterations: int = 1000,
	seed: int = 0,
	total_expected_birds: int = len(DEFAULT_BIRD_IDS),
) -> dict:
	"""Summarize per-bird trends as cross-bird replication evidence."""
	analyzed = {
		bird: payload
		for bird, payload in per_bird_metrics.items()
		if payload.get("status") == "analyzed"
	}
	branch_summary = {}
	for metric, expected in PRIMARY_METRICS.items():
		rows = _per_bird_metric_rows(analyzed, metric)
		branch_summary[metric] = _cross_metric_summary(
			rows,
			expected_sign=expected,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
		)
	radius_rows = _per_bird_radius_rows(analyzed)
	radius_summary = _cross_metric_summary(
		radius_rows,
		expected_sign=0,
		bootstrap_iterations=bootstrap_iterations,
		seed=seed + 17,
	)
	regime = regime_stratified_summary(analyzed)
	success = replication_success_summary(
		branch_summary,
		n_expected_total=total_expected_birds,
		min_success_birds=9,
	)
	return {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"n_expected_birds": int(total_expected_birds),
		"n_analyzed_birds": int(len(analyzed)),
		"analyzed_birds": sorted(analyzed),
		"missing_or_failed_birds": sorted(
			bird for bird, payload in per_bird_metrics.items()
			if payload.get("status") != "analyzed"
		),
		"branch_metrics": branch_summary,
		"negative_controls": {
			NEGATIVE_CONTROL_METRIC: radius_summary,
		},
		"regime_stratified": regime,
		"replication_success": success,
		"sign_consistency_table": sign_consistency_table(analyzed),
	}


def sign_summary_for_metrics(primary_metrics: dict) -> dict:
	"""Return expected-sign pass/fail flags for one bird."""
	return {
		metric: {
			"expected_sign": int(expected),
			"spearman_rho": _optional_float(
				primary_metrics.get(metric, {}).get("spearman_rho")
			),
			"matches_expected_sign": _metric_matches_expected(
				primary_metrics.get(metric, {}),
				expected,
			),
			"ci_supports_expected_sign": _metric_ci_supports_expected(
				primary_metrics.get(metric, {}),
				expected,
			),
		}
		for metric, expected in PRIMARY_METRICS.items()
	}


def sign_consistency_table(analyzed: dict[str, dict]) -> list[dict]:
	"""Create one compact per-bird row of primary-sign results."""
	rows = []
	for bird, payload in sorted(analyzed.items()):
		row = {
			"bird_id": bird,
			"regime": _manifest_regime_label(payload.get("manifest", {})),
		}
		for metric, expected in PRIMARY_METRICS.items():
			metric_payload = payload.get("primary_metrics", {}).get(metric, {})
			rho = _optional_float(metric_payload.get("spearman_rho"))
			row[f"{metric}_rho"] = rho
			row[f"{metric}_expected"] = (
				None if rho is None else bool(rho * float(expected) > 0)
			)
		radius = (
			payload.get("negative_controls", {})
			.get(NEGATIVE_CONTROL_METRIC, {})
			.get("radius_age", {})
		)
		row["poincare_radius_rho"] = _optional_float(radius.get("spearman_rho"))
		rows.append(row)
	return rows


def ingest_branch_report_metrics(metrics_path: Path) -> dict:
	"""Load the existing single-bird branch report metrics schema."""
	payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
	required = ("primary_metrics", "branch_clusters", "coverage_bias")
	missing = [key for key in required if key not in payload]
	if missing:
		raise ValueError(
			"Branch report metrics are missing required keys: " + ", ".join(missing)
		)
	inputs = payload.get("inputs", {})
	return {
		"status": "ingested",
		"bird_id": _normalize_bird_id(inputs.get("bird_id")),
		"source_mode": inputs.get("source_mode"),
		"branch_clusters": payload["branch_clusters"],
		"primary_metrics": payload["primary_metrics"],
		"coverage_bias": payload["coverage_bias"],
		"sensitivity": payload.get("sensitivity", {}),
	}


def regime_stratified_summary(analyzed: dict[str, dict]) -> dict:
	"""Summarize metric rhos by tutor/acoustic regime label."""
	grouped: dict[str, list[dict]] = defaultdict(list)
	for bird, payload in analyzed.items():
		grouped[_manifest_regime_label(payload.get("manifest", {}))].append(payload)
	out = {}
	for regime, payloads in sorted(grouped.items()):
		metrics = {}
		for metric, expected in PRIMARY_METRICS.items():
			values = [
				_optional_float(payload.get("primary_metrics", {}).get(metric, {}).get("spearman_rho"))
				for payload in payloads
			]
			values = [value for value in values if value is not None]
			metrics[metric] = {
				"n_birds": int(len(values)),
				"mean_rho": float(np.mean(values)) if values else None,
				"median_rho": float(np.median(values)) if values else None,
				"expected_sign_count": int(
					sum(value * float(expected) > 0 for value in values)
				),
			}
		out[regime] = {
			"birds": sorted(payload["bird_id"] for payload in payloads),
			"metrics": metrics,
		}
	return out


def replication_success_summary(
	branch_summary: dict[str, dict],
	n_expected_total: int,
	min_success_birds: int = 9,
) -> dict:
	"""Evaluate the fixed success criterion for the primary replication claim."""
	criteria = {}
	for metric in CRITERION_METRICS:
		payload = branch_summary.get(metric, {})
		expected = int(payload.get("expected_sign", PRIMARY_METRICS[metric]))
		ci = payload.get("bootstrap_mean_rho_ci95")
		sign_count = int(payload.get("expected_sign_count", 0))
		if ci is None:
			ci_excludes_zero = False
		elif expected > 0:
			ci_excludes_zero = float(ci[0]) > 0
		else:
			ci_excludes_zero = float(ci[1]) < 0
		criteria[metric] = {
			"expected_sign_count": sign_count,
			"required_sign_count": int(min_success_birds),
			"sign_count_pass": bool(sign_count >= int(min_success_birds)),
			"bootstrap_ci_excludes_zero": bool(ci_excludes_zero),
			"passes": bool(sign_count >= int(min_success_birds) and ci_excludes_zero),
		}
	return {
		"n_expected_total": int(n_expected_total),
		"min_success_birds": int(min_success_birds),
		"criteria": criteria,
		"passes": bool(all(row["passes"] for row in criteria.values())),
	}


def make_replication_figures(
	per_bird: dict[str, dict],
	cross_bird: dict,
	inventory: dict,
	out_dir: Path,
) -> dict[str, str]:
	"""Create replication summary figures and return PNG paths."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_dir.mkdir(parents=True, exist_ok=True)
	figures = {}

	fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
	plot_metrics = list(CRITERION_METRICS) + [
		"branch_probability_margin",
		NEGATIVE_CONTROL_METRIC,
	]
	for ax, metric in zip(axes.ravel(), plot_metrics):
		_plot_forest_metric(ax, per_bird, metric)
	fig.suptitle("Per-bird developmental trend rhos")
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	_save_figure(fig, out_dir / "per_bird_rho_forest")
	figures["per_bird_rho_forest"] = (
		out_dir / "per_bird_rho_forest.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(10, 4.5))
	_plot_sign_consistency(ax, cross_bird.get("sign_consistency_table", []))
	_save_figure(fig, out_dir / "sign_consistency")
	figures["sign_consistency"] = (out_dir / "sign_consistency.png").as_posix()

	fig, ax = plt.subplots(figsize=(10, 4.5))
	_plot_regime_summary(ax, cross_bird.get("regime_stratified", {}))
	_save_figure(fig, out_dir / "regime_stratified_rhos")
	figures["regime_stratified_rhos"] = (
		out_dir / "regime_stratified_rhos.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	_plot_input_inventory(ax, inventory)
	_save_figure(fig, out_dir / "input_inventory")
	figures["input_inventory"] = (out_dir / "input_inventory.png").as_posix()
	return figures


def write_replication_report(
	report_path: Path,
	cohort: dict,
	inventory: dict,
	per_bird: dict[str, dict],
	cross_bird: dict,
	figures: dict[str, str],
	artifacts: dict[str, str],
) -> None:
	"""Write the markdown multi-bird replication report."""
	report_path.parent.mkdir(parents=True, exist_ok=True)
	n_analyzed = cross_bird.get("n_analyzed_birds", 0)
	n_expected = cross_bird.get("n_expected_birds", len(cohort.get("bird_ids", [])))
	success = cross_bird.get("replication_success", {})
	missing = cross_bird.get("missing_or_failed_birds", [])
	lines = [
		"# Multi-Bird Developmental Branch Commitment Replication",
		"",
		"## Summary",
		"",
		f"- Cohort requested: {n_expected} birds ({', '.join(cohort.get('bird_ids', []))}).",
		f"- Birds analyzed with local inputs: {n_analyzed}.",
		f"- Missing or failed birds: {', '.join(missing) if missing else 'none'}.",
		f"- Replication success criterion passed: {'yes' if success.get('passes') else 'no'}.",
		_replication_conclusion(cross_bird),
		_radius_conclusion(cross_bird),
		"",
		"## Figures",
		"",
		f"![Per-bird rho forest]({_rel(figures['per_bird_rho_forest'], report_path)})",
		"",
		f"![Sign consistency]({_rel(figures['sign_consistency'], report_path)})",
		"",
		f"![Regime-stratified rhos]({_rel(figures['regime_stratified_rhos'], report_path)})",
		"",
		f"![Input inventory]({_rel(figures['input_inventory'], report_path)})",
		"",
		"## Cross-Bird Metrics",
		"",
	]
	for metric in CRITERION_METRICS:
		payload = cross_bird.get("branch_metrics", {}).get(metric, {})
		lines.append(_cross_metric_line(metric, payload))
	lines.extend(
		[
			"",
			"## Coverage And Bias",
			"",
			_coverage_bias_text(per_bird),
			"",
			_bias_sensitivity_text(per_bird),
			"",
			"## Missing Inputs",
			"",
		]
	)
	if missing:
		for bird in missing:
			row = inventory.get("birds", {}).get(bird, {})
			lines.append(f"- `{bird}`: {row.get('reason') or row.get('status')}")
	else:
		lines.append("- None.")
	lines.extend(
		[
			"",
			"## Artifacts",
			"",
		]
	)
	for name, path in artifacts.items():
		lines.append(f"- `{name}`: `{_rel(path, report_path)}`")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_or_build_bird_records(
	bird_id: str,
	source: Optional[dict],
	manifest: dict,
	manifest_path: Path,
	latent_root: Optional[Path],
	audio_root: Optional[Path],
	roi_root: Optional[Path],
	split: str,
	dph_min: float,
	dph_max: float,
	roi_format: str,
	roi_parquet_name: str,
	max_events_per_dph: Optional[int],
	seed: int,
) -> tuple[list[dict], dict, dict]:
	if source is not None:
		event_table = Path(source["event_table"])
		if not event_table.exists():
			raise FileNotFoundError(f"event table is missing: {event_table.as_posix()}")
		records = load_event_records_parquet(
			event_table,
			bird_id=bird_id,
			dph_min=dph_min,
			dph_max=dph_max,
		)
		event_summary = load_event_summary_for_event_table(event_table)
		return records, event_summary, {
			"source_mode": "event_table",
			"event_table": event_table.as_posix(),
			"coverage_metrics": _coverage_source(event_summary),
			"discovery": source.get("discovery"),
			"bird_id": bird_id,
			"dph_min": float(dph_min),
			"dph_max": float(dph_max),
		}
	if latent_root is None:
		raise FileNotFoundError(
			"No local event_latents.parquet was found and --latent-root was not provided."
		)
	build = build_event_records_from_manifest(
		manifest=manifest,
		latent_root=latent_root,
		bird_id=bird_id,
		dph_min=dph_min,
		dph_max=dph_max,
		split=split,
		audio_root=audio_root,
		roi_root=roi_root,
		roi_format=roi_format,
		roi_parquet_name=roi_parquet_name,
		max_events_per_dph=max_events_per_dph,
		seed=seed,
	)
	if not build.records:
		raise FileNotFoundError(
			"No event records were built from local latent/ROI artifacts."
		)
	return build.records, build.summary, {
		"source_mode": "full_rebuild",
		"manifest_path": manifest_path.as_posix(),
		"latent_root": latent_root.as_posix(),
		"roi_root": None if roi_root is None else roi_root.as_posix(),
		"split": split,
		"roi_format": roi_format,
		"max_events_per_dph": max_events_per_dph,
		"bird_id": bird_id,
		"dph_min": float(dph_min),
		"dph_max": float(dph_max),
	}


def _analyze_sensitivity_records(
	records: Sequence[dict],
	early_dph_max: float,
	late_dph_min: float,
	cluster_min_k: int,
	cluster_max_k: int,
	bootstrap_iterations: int,
	seed: int,
	use_energy_weighted: bool,
) -> dict:
	if len(records) == 0:
		return {"status": "failed", "reason": "No records after filtering."}
	dph_values = {float(record["dph"]) for record in records if "dph" in record}
	if len(dph_values) < 2:
		return {"status": "failed", "reason": "Fewer than two dph bins."}
	if not any(float(record["dph"]) >= float(late_dph_min) for record in records):
		return {"status": "failed", "reason": "No late-reference records."}
	try:
		result = analyze_branch_records(
			records,
			event_summary=None,
			inputs={"source_mode": "sensitivity"},
			early_dph_max=early_dph_max,
			late_dph_min=late_dph_min,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
			use_energy_weighted=use_energy_weighted,
			run_sensitivity=False,
		)
	except Exception as exc:
		return {"status": "failed", "reason": str(exc)}
	return {
		"status": "computed",
		"events": int(len(result.records)),
		"primary_metrics": result.metrics["primary_metrics"],
		"signs": sign_summary_for_metrics(result.metrics["primary_metrics"]),
	}


def _attach_residuals(
	records: Sequence[dict],
	metric: str,
	control_fields: Sequence[str],
) -> list[dict]:
	rows = []
	y = []
	x_rows = []
	for record in records:
		value = _coerce_float(record.get(metric))
		if value is None or not math.isfinite(value):
			continue
		controls = []
		valid = True
		for field in control_fields:
			control = _coerce_float(record.get(field))
			if control is None or not math.isfinite(control):
				valid = False
				break
			controls.append(float(control))
		if not valid:
			continue
		rows.append(dict(record))
		y.append(float(value))
		x_rows.append(controls)
	if len(rows) <= len(control_fields) + 1:
		return []
	x = np.asarray(x_rows, dtype=np.float64)
	y_arr = np.asarray(y, dtype=np.float64)
	x_mean = np.mean(x, axis=0)
	x_std = np.std(x, axis=0)
	x_std[x_std <= 1e-12] = 1.0
	x_design = np.column_stack([np.ones(x.shape[0]), (x - x_mean) / x_std])
	beta, *_ = np.linalg.lstsq(x_design, y_arr, rcond=None)
	residuals = y_arr - x_design @ beta
	for row, residual in zip(rows, residuals):
		row[f"{metric}_residual"] = float(residual)
	return rows


def _cross_metric_summary(
	rows: Sequence[dict],
	expected_sign: int,
	bootstrap_iterations: int,
	seed: int,
) -> dict:
	values = [
		float(row["spearman_rho"]) for row in rows
		if row.get("spearman_rho") is not None
		and math.isfinite(float(row["spearman_rho"]))
	]
	arr = np.asarray(values, dtype=np.float64)
	boot = _bootstrap_mean(arr, bootstrap_iterations, seed)
	ci = (
		[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
		if boot.size else None
	)
	if expected_sign == 0:
		expected_count = None
	else:
		expected_count = int(sum(value * float(expected_sign) > 0 for value in values))
	return {
		"expected_sign": int(expected_sign),
		"n_birds": int(arr.size),
		"expected_sign_count": expected_count,
		"mean_rho": float(np.mean(arr)) if arr.size else None,
		"median_rho": float(np.median(arr)) if arr.size else None,
		"bootstrap_iterations": int(bootstrap_iterations),
		"bootstrap_mean_rho_ci95": ci,
		"per_bird": list(rows),
	}


def _per_bird_metric_rows(analyzed: dict[str, dict], metric: str) -> list[dict]:
	rows = []
	for bird, payload in sorted(analyzed.items()):
		metric_payload = payload.get("primary_metrics", {}).get(metric, {})
		rows.append(
			{
				"bird_id": bird,
				"regime": _manifest_regime_label(payload.get("manifest", {})),
				"spearman_rho": _optional_float(metric_payload.get("spearman_rho")),
				"bootstrap_rho_ci95": metric_payload.get("bootstrap_rho_ci95"),
			}
		)
	return rows


def _per_bird_radius_rows(analyzed: dict[str, dict]) -> list[dict]:
	rows = []
	for bird, payload in sorted(analyzed.items()):
		radius_payload = (
			payload.get("negative_controls", {})
			.get(NEGATIVE_CONTROL_METRIC, {})
			.get("radius_age", {})
		)
		rows.append(
			{
				"bird_id": bird,
				"regime": _manifest_regime_label(payload.get("manifest", {})),
				"spearman_rho": _optional_float(radius_payload.get("spearman_rho")),
				"bootstrap_rho_ci95": radius_payload.get("bootstrap_rho_ci95"),
			}
		)
	return rows


def _bootstrap_mean(values: np.ndarray, iterations: int, seed: int) -> np.ndarray:
	values = np.asarray(values, dtype=np.float64)
	values = values[np.isfinite(values)]
	if values.size < 2 or iterations <= 0:
		return np.zeros((0,), dtype=np.float64)
	rng = np.random.default_rng(int(seed))
	out = []
	for _ in range(int(iterations)):
		idx = rng.integers(0, values.size, size=values.size)
		out.append(float(np.mean(values[idx])))
	return np.asarray(out, dtype=np.float64)


def _missing_bird_metrics(
	bird_id: str,
	manifest_summary: dict,
	reason: str,
	status: str = "missing_inputs",
) -> dict:
	return {
		"bird_id": _normalize_bird_id(bird_id),
		"status": status,
		"manifest": manifest_summary,
		"reason": reason,
	}


def _event_table_birds(
	path: Path,
	dph_min: Optional[float],
	dph_max: Optional[float],
) -> set[str]:
	try:
		import pyarrow.parquet as pq  # type: ignore

		table = pq.read_table(path.as_posix(), columns=["bird_id_norm", "dph"])
	except Exception:
		return set()
	birds = set()
	for row in table.to_pylist():
		dph = _coerce_float(row.get("dph"))
		if dph is None or not math.isfinite(dph):
			continue
		if dph_min is not None and dph < float(dph_min):
			continue
		if dph_max is not None and dph > float(dph_max):
			continue
		bird = _normalize_bird_id(row.get("bird_id_norm"))
		if bird:
			birds.add(bird)
	return birds


def _roi_without_window_rates_by_dph(event_summary: dict) -> dict[float, float]:
	rates = {}
	for dph_key, row in (event_summary or {}).get("by_dph", {}).items():
		total = _coerce_float(row.get("roi_events_total"))
		without = _coerce_float(row.get("roi_events_without_windows"))
		dph = _coerce_float(row.get("dph", dph_key))
		if dph is None or total is None or without is None or total <= 0:
			continue
		rates[float(dph)] = float(without / total)
	return rates


def _events_per_dph(records: Sequence[dict]) -> dict[str, int]:
	counts: dict[float, int] = defaultdict(int)
	for record in records:
		dph = _coerce_float(record.get("dph"))
		if dph is not None and math.isfinite(dph):
			counts[float(dph)] += 1
	return {str(_format_dph_key(dph)): int(count) for dph, count in sorted(counts.items())}


def _iter_manifest_rows(manifest: dict):
	for split, rows in manifest.items():
		if not isinstance(rows, list):
			continue
		for row in rows:
			if not isinstance(row, dict):
				continue
			yield {**row, "split": row.get("split", split)}


def _manifest_regime_label(summary: dict) -> str:
	regimes = summary.get("regimes") or []
	if not regimes:
		return "unknown"
	return "+".join(str(regime) for regime in regimes)


def _metric_matches_expected(metric: dict, expected_sign: int) -> bool:
	rho = _optional_float(metric.get("spearman_rho"))
	return bool(rho is not None and rho * float(expected_sign) > 0)


def _metric_ci_supports_expected(metric: dict, expected_sign: int) -> bool:
	ci = metric.get("bootstrap_rho_ci95")
	if not ci:
		return False
	if expected_sign > 0:
		return float(ci[0]) > 0
	return float(ci[1]) < 0


def _coverage_source(event_summary: dict) -> Optional[str]:
	if not event_summary:
		return None
	source = event_summary.get("coverage_source")
	return None if source is None else str(source)


def _optional_float(value: Any) -> Optional[float]:
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return out if math.isfinite(out) else None


def _coerce_float(value: Any) -> Optional[float]:
	return _optional_float(value)


def _normalize_bird_id(value: Any) -> str:
	return str(value or "").strip().upper()


def _format_dph_key(dph: float) -> str:
	if float(dph).is_integer():
		return str(int(dph))
	return f"{float(dph):g}"


def _rel(path: str, report_path: Path) -> str:
	return os.path.relpath(str(path), report_path.parent.as_posix())


def _replication_conclusion(cross_bird: dict) -> str:
	success = cross_bird.get("replication_success", {})
	if success.get("passes"):
		return "- Branch commitment replicated across the fixed cohort by the prespecified sign and bootstrap criteria."
	n_analyzed = int(cross_bird.get("n_analyzed_birds", 0))
	if n_analyzed < 9:
		return "- Branch commitment cannot yet be evaluated against the 9-of-11 replication criterion because local inputs are missing for most birds."
	return "- Branch commitment did not meet the prespecified multi-bird replication criterion."


def _radius_conclusion(cross_bird: dict) -> str:
	payload = (
		cross_bird.get("negative_controls", {})
		.get(NEGATIVE_CONTROL_METRIC, {})
	)
	mean_rho = payload.get("mean_rho")
	if mean_rho is None:
		return "- Optimized Poincare radius negative-control evidence is unavailable or insufficient."
	if float(mean_rho) > 0:
		return "- Optimized Poincare radius trends positive on average, so the hyperbolic VAE gate remains an open question rather than supported by branch metrics alone."
	return "- Optimized Poincare radius is not positive on average, consistent with leaving the hyperbolic VAE gate untriggered."


def _coverage_bias_text(per_bird: dict[str, dict]) -> str:
	analyzed = [
		payload for payload in per_bird.values()
		if payload.get("status") == "analyzed"
	]
	if not analyzed:
		return "No analyzed birds are available, so coverage-bias sensitivity cannot be interpreted."
	flagged = [
		payload["bird_id"] for payload in analyzed
		if payload.get("coverage_bias", {}).get("potential_bias_flag")
	]
	missing_counts = [
		payload["bird_id"] for payload in analyzed
		if not payload.get("coverage_bias", {}).get("counts_available")
	]
	parts = []
	parts.append(
		"Coverage/skips could plausibly bias interpretation for "
		+ (", ".join(flagged) if flagged else "no analyzed birds with available counts")
		+ "."
	)
	if missing_counts:
		parts.append(
			"Coverage counts are unavailable for "
			+ ", ".join(missing_counts)
			+ "."
		)
	return " ".join(parts)


def _bias_sensitivity_text(per_bird: dict[str, dict]) -> str:
	analyzed = [
		payload for payload in per_bird.values()
		if payload.get("status") == "analyzed"
	]
	if not analyzed:
		return "No analyzed birds are available for skip-filter, equalized-dph, or residualized-control sensitivity checks."
	rows = []
	for payload in analyzed:
		bias = payload.get("bias_sensitivities", {})
		if bias.get("status") == "skipped":
			rows.append(f"`{payload['bird_id']}`: sensitivity checks skipped.")
			continue
		skip = bias.get("skip_filtered", {})
		equalized = bias.get("equalized_events_per_dph", {})
		residual = bias.get("residualized_controls", {}).get("primary_metrics", {})
		residual_signs = []
		for metric in CRITERION_METRICS:
			metric_payload = residual.get(metric, {})
			rho = _optional_float(metric_payload.get("spearman_rho"))
			expected = PRIMARY_METRICS[metric]
			if rho is not None:
				residual_signs.append(rho * float(expected) > 0)
		residual_text = (
			"all criterion residualized signs match"
			if residual_signs and all(residual_signs)
			else "residualized signs are incomplete or mixed"
		)
		rows.append(
			f"`{payload['bird_id']}`: skip-filter status {skip.get('status', 'n/a')} "
			f"with {len(skip.get('excluded_dph') or [])} excluded dph bins; "
			f"equalized status {equalized.get('status', 'n/a')} "
			f"over {equalized.get('events', 'n/a')} events; {residual_text}."
		)
	return " ".join(rows)


def _cross_metric_line(metric: str, payload: dict) -> str:
	ci = payload.get("bootstrap_mean_rho_ci95")
	ci_text = "n/a" if ci is None else f"[{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
	mean = payload.get("mean_rho")
	mean_text = "n/a" if mean is None else f"{float(mean):.3f}"
	return (
		f"- `{metric}`: mean rho {mean_text}, CI {ci_text}, "
		f"expected-sign birds {payload.get('expected_sign_count')} / {payload.get('n_birds')}."
	)


def _plot_forest_metric(ax, per_bird: dict[str, dict], metric: str) -> None:
	rows = []
	for bird, payload in sorted(per_bird.items()):
		if payload.get("status") != "analyzed":
			continue
		if metric == NEGATIVE_CONTROL_METRIC:
			metric_payload = (
				payload.get("negative_controls", {})
				.get(NEGATIVE_CONTROL_METRIC, {})
				.get("radius_age", {})
			)
			expected = 0
		else:
			metric_payload = payload.get("primary_metrics", {}).get(metric, {})
			expected = PRIMARY_METRICS.get(metric, 0)
		rho = _optional_float(metric_payload.get("spearman_rho"))
		if rho is None:
			continue
		rows.append((bird, rho, expected))
	if not rows:
		ax.text(0.5, 0.5, "no analyzed birds", ha="center", va="center")
		ax.set_title(metric)
		ax.set_xlim(-1, 1)
		return
	birds = [row[0] for row in rows]
	rhos = np.asarray([row[1] for row in rows], dtype=np.float64)
	colors = []
	for _, rho, expected in rows:
		if expected == 0:
			colors.append("#4C78A8")
		elif rho * float(expected) > 0:
			colors.append("#54A24B")
		else:
			colors.append("#E45756")
	y = np.arange(len(rows), dtype=np.float64)
	ax.axvline(0, color="0.5", linewidth=1.0)
	ax.scatter(rhos, y, c=colors, s=35)
	ax.set_yticks(y)
	ax.set_yticklabels(birds, fontsize=8)
	ax.set_xlim(-1.05, 1.05)
	ax.set_title(metric.replace("branch_", "").replace("_", " "))
	ax.set_xlabel("Spearman rho")


def _plot_sign_consistency(ax, rows: Sequence[dict]) -> None:
	metrics = list(CRITERION_METRICS)
	if not rows:
		ax.text(0.5, 0.5, "no analyzed birds", ha="center", va="center")
		ax.axis("off")
		return
	matrix = []
	birds = []
	for row in rows:
		birds.append(row["bird_id"])
		matrix.append([
			1.0 if row.get(f"{metric}_expected") else -1.0
			for metric in metrics
		])
	arr = np.asarray(matrix, dtype=np.float64)
	ax.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
	ax.set_xticks(np.arange(len(metrics)))
	ax.set_xticklabels(
		[metric.replace("branch_", "").replace("_", "\n") for metric in metrics],
		fontsize=8,
	)
	ax.set_yticks(np.arange(len(birds)))
	ax.set_yticklabels(birds, fontsize=8)
	ax.set_title("Expected-sign consistency")


def _plot_regime_summary(ax, regime_summary: dict) -> None:
	if not regime_summary:
		ax.text(0.5, 0.5, "no analyzed birds", ha="center", va="center")
		ax.axis("off")
		return
	regimes = sorted(regime_summary)
	metrics = list(CRITERION_METRICS)
	width = 0.8 / max(1, len(metrics))
	x = np.arange(len(regimes), dtype=np.float64)
	for i, metric in enumerate(metrics):
		values = []
		for regime in regimes:
			value = (
				regime_summary[regime]
				.get("metrics", {})
				.get(metric, {})
				.get("mean_rho")
			)
			values.append(np.nan if value is None else float(value))
		ax.bar(x + (i - (len(metrics) - 1) / 2) * width, values, width=width, label=metric)
	ax.axhline(0, color="0.5", linewidth=1.0)
	ax.set_xticks(x)
	ax.set_xticklabels(regimes)
	ax.set_ylabel("mean rho")
	ax.set_title("Regime-stratified branch trends")
	ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _plot_input_inventory(ax, inventory: dict) -> None:
	status_counts: dict[str, int] = defaultdict(int)
	for row in inventory.get("birds", {}).values():
		status_counts[str(row.get("status", "unknown"))] += 1
	if not status_counts:
		ax.text(0.5, 0.5, "empty inventory", ha="center", va="center")
		ax.axis("off")
		return
	labels = sorted(status_counts)
	values = [status_counts[label] for label in labels]
	ax.bar(labels, values, color="#4C78A8")
	ax.set_ylabel("birds")
	ax.set_title("Input inventory status")
	ax.tick_params(axis="x", rotation=20)


def _save_figure(fig, prefix: Path) -> None:
	fig.tight_layout()
	fig.savefig(Path(f"{prefix.as_posix()}.png"), dpi=180)
	fig.savefig(Path(f"{prefix.as_posix()}.pdf"))
	import matplotlib.pyplot as plt

	plt.close(fig)
