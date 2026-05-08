"""Developmental branch-commitment analyses for AVA event latents."""

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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ava.analysis.hyperbolic_development import (
	DEFAULT_CLUSTER_K_RANGE,
	EPS,
	build_event_records_from_manifest,
	early_late_metric_contrast,
	grouped_median_metric,
	latent_matrix,
	load_manifest,
	radius_age_metrics,
	standardize_latents,
	write_json,
)


DEFAULT_BRANCH_BEAD = "autoencoded-vocal-analysis-obi.4"
PRIMARY_METRICS = {
	"branch_confidence": 1,
	"branch_entropy": -1,
	"branch_nearest_distance": -1,
	"branch_distance_margin": 1,
	"branch_probability_margin": 1,
	"branch_within_standardized_distance": -1,
}


@dataclass
class BranchClusterResult:
	"""Adult-like branch fit and per-event assignment arrays."""

	best_k: int
	silhouette_scores: dict[str, float]
	centers: np.ndarray
	labels: np.ndarray
	nearest_distance: np.ndarray
	second_distance: np.ndarray
	distance_margin: np.ndarray
	confidence: np.ndarray
	entropy: np.ndarray
	probability_margin: np.ndarray
	within_standardized_distance: np.ndarray
	late_event_count: int
	distance_scale: float
	late_branch_distance_medians: dict[str, float]


@dataclass
class BranchAnalysisResult:
	"""Branch-commitment records and metrics."""

	records: list[dict]
	clusters: BranchClusterResult
	metrics: dict


def load_event_records_parquet(
	path: Path,
	bird_id: Optional[str] = None,
	dph_min: Optional[float] = None,
	dph_max: Optional[float] = None,
) -> list[dict]:
	"""Load an event-level latent parquet table and apply bird/dph filters."""
	try:
		import pyarrow.parquet as pq  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError("Event-table branch analysis requires pyarrow.") from exc

	table = pq.read_table(path.as_posix())
	records = []
	bird_id_norm = None if bird_id is None else str(bird_id).strip().upper()
	for record in table.to_pylist():
		dph = _coerce_dph(record.get("dph"))
		if dph is None:
			continue
		if bird_id_norm is not None:
			record_bird = str(record.get("bird_id_norm", "")).strip().upper()
			if record_bird != bird_id_norm:
				continue
		if dph_min is not None and dph < float(dph_min):
			continue
		if dph_max is not None and dph > float(dph_max):
			continue
		if record.get("mu") is None:
			raise ValueError(f"Event table row is missing mu: {path.as_posix()}")
		record["dph"] = float(dph)
		records.append(record)
	if not records:
		raise ValueError("No event records remained after event-table filtering.")
	return records


def fit_branch_clusters(
	records: Sequence[dict],
	x_std: np.ndarray,
	late_dph_min: float = 80,
	min_k: int = DEFAULT_CLUSTER_K_RANGE[0],
	max_k: int = DEFAULT_CLUSTER_K_RANGE[1],
	seed: int = 0,
) -> BranchClusterResult:
	"""Fit adult-like branches on late events and score all events."""
	dph = np.asarray([float(record["dph"]) for record in records], dtype=np.float64)
	late_idx = np.flatnonzero(dph >= float(late_dph_min))
	if late_idx.size == 0:
		raise ValueError("No late-age events available for branch clustering.")

	x_late = np.asarray(x_std[late_idx], dtype=np.float32)
	scores: dict[str, float] = {}
	candidate_min = max(2, int(min_k))
	candidate_max = min(int(max_k), int(late_idx.size) - 1)
	best_k = 1
	best_score = -np.inf
	best_model = None
	if candidate_max >= candidate_min:
		for k in range(candidate_min, candidate_max + 1):
			model = KMeans(n_clusters=k, n_init=10, random_state=int(seed))
			labels = model.fit_predict(x_late)
			if len(set(labels.tolist())) < 2:
				continue
			score = float(silhouette_score(x_late, labels))
			scores[str(k)] = score
			if score > best_score:
				best_score = score
				best_k = int(k)
				best_model = model

	if best_model is None:
		best_k = 1
		centers = np.mean(x_late, axis=0, keepdims=True)
	else:
		centers = best_model.cluster_centers_.astype(np.float32)

	distances = np.linalg.norm(x_std[:, None, :] - centers[None, :, :], axis=2)
	labels = np.argmin(distances, axis=1).astype(np.int32)
	sorted_distances = np.sort(distances, axis=1)
	nearest = sorted_distances[:, 0].astype(np.float32)
	if centers.shape[0] == 1:
		second = np.full_like(nearest, np.nan, dtype=np.float32)
		distance_margin = np.full_like(nearest, np.nan, dtype=np.float32)
		confidence = np.ones(x_std.shape[0], dtype=np.float32)
		entropy = np.zeros(x_std.shape[0], dtype=np.float32)
		probability_margin = np.ones(x_std.shape[0], dtype=np.float32)
	else:
		second = sorted_distances[:, 1].astype(np.float32)
		distance_margin = (second - nearest).astype(np.float32)
		scale = float(np.median(nearest))
		if not math.isfinite(scale) or scale <= EPS:
			scale = 1.0
		logits = -distances / scale
		logits = logits - np.max(logits, axis=1, keepdims=True)
		probs = np.exp(logits)
		probs = probs / np.sum(probs, axis=1, keepdims=True)
		sorted_probs = np.sort(probs, axis=1)
		confidence = sorted_probs[:, -1].astype(np.float32)
		probability_margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).astype(np.float32)
		entropy_raw = -np.sum(probs * np.log(probs + EPS), axis=1)
		entropy = (entropy_raw / math.log(centers.shape[0])).astype(np.float32)

	late_branch_medians: dict[str, float] = {}
	global_late = nearest[late_idx]
	global_late_median = float(np.median(global_late)) if global_late.size else 1.0
	if not math.isfinite(global_late_median) or global_late_median <= EPS:
		global_late_median = 1.0
	within = np.zeros_like(nearest, dtype=np.float32)
	for cluster in range(int(centers.shape[0])):
		mask = (labels == cluster) & (dph >= float(late_dph_min))
		denom = float(np.median(nearest[mask])) if np.any(mask) else global_late_median
		if not math.isfinite(denom) or denom <= EPS:
			denom = global_late_median
		late_branch_medians[str(cluster)] = float(denom)
		within[labels == cluster] = nearest[labels == cluster] / float(denom)

	distance_scale = float(np.median(nearest))
	if not math.isfinite(distance_scale) or distance_scale <= EPS:
		distance_scale = 1.0
	return BranchClusterResult(
		best_k=int(best_k),
		silhouette_scores=scores,
		centers=np.asarray(centers, dtype=np.float32),
		labels=labels,
		nearest_distance=nearest,
		second_distance=second,
		distance_margin=distance_margin,
		confidence=confidence,
		entropy=entropy,
		probability_margin=probability_margin,
		within_standardized_distance=within.astype(np.float32),
		late_event_count=int(late_idx.size),
		distance_scale=float(distance_scale),
		late_branch_distance_medians=late_branch_medians,
	)


def apply_branch_assignments(records: Sequence[dict], result: BranchClusterResult) -> None:
	"""Attach branch commitment fields to event records."""
	for i, record in enumerate(records):
		record["branch_cluster"] = int(result.labels[i])
		record["branch_nearest_distance"] = float(result.nearest_distance[i])
		record["branch_second_distance"] = _finite_or_none(result.second_distance[i])
		record["branch_distance_margin"] = _finite_or_none(result.distance_margin[i])
		record["branch_confidence"] = float(result.confidence[i])
		record["branch_entropy"] = float(result.entropy[i])
		record["branch_probability_margin"] = float(result.probability_margin[i])
		record["branch_within_standardized_distance"] = float(
			result.within_standardized_distance[i]
		)


def analyze_branch_records(
	records: Sequence[dict],
	event_summary: Optional[dict] = None,
	inputs: Optional[dict] = None,
	early_dph_max: float = 45,
	late_dph_min: float = 80,
	cluster_min_k: int = DEFAULT_CLUSTER_K_RANGE[0],
	cluster_max_k: int = DEFAULT_CLUSTER_K_RANGE[1],
	bootstrap_iterations: int = 1000,
	seed: int = 0,
	use_energy_weighted: bool = False,
	run_sensitivity: bool = True,
) -> BranchAnalysisResult:
	"""Run branch-commitment metrics on event records."""
	analysis_records = [dict(record) for record in records]
	if not analysis_records:
		raise ValueError("No event records available for branch commitment analysis.")

	x_raw = latent_matrix(analysis_records, use_energy_weighted=use_energy_weighted)
	x_std, standardization = standardize_latents(x_raw)
	clusters = fit_branch_clusters(
		analysis_records,
		x_std,
		late_dph_min=late_dph_min,
		min_k=cluster_min_k,
		max_k=cluster_max_k,
		seed=seed,
	)
	apply_branch_assignments(analysis_records, clusters)
	event_summary = event_summary or summarize_records_only(analysis_records)
	primary = _primary_metric_payloads(
		analysis_records,
		bootstrap_iterations=bootstrap_iterations,
		seed=seed,
	)
	metrics = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"inputs": dict(inputs or {}),
		"event_summary": event_summary,
		"standardization": standardization,
		"branch_clusters": branch_cluster_payload(clusters, late_dph_min),
		"primary_metrics": primary,
		"early_late_contrasts": {
			key: early_late_metric_contrast(
				analysis_records,
				key,
				early_dph_max,
				late_dph_min,
			)
			for key in PRIMARY_METRICS
		},
		"controls_by_dph": {
			key: grouped_metric_summary(analysis_records, key)
			for key in (
				"duration_sec",
				"n_windows",
				"mean_energy",
				"variance_mean",
			)
		},
		"coverage_bias": coverage_bias_summary(event_summary),
	}
	if run_sensitivity:
		sensitivity_records = [dict(record) for record in records]
		sensitivity_clusters = fit_branch_clusters(
			sensitivity_records,
			x_std,
			late_dph_min=late_dph_min,
			min_k=cluster_min_k,
			max_k=cluster_max_k,
			seed=int(seed) + 1,
		)
		apply_branch_assignments(sensitivity_records, sensitivity_clusters)
		sensitivity_primary = _primary_metric_payloads(
			sensitivity_records,
			bootstrap_iterations=bootstrap_iterations,
			seed=int(seed) + 1,
		)
		metrics["sensitivity"] = {
			"seed": int(seed) + 1,
			"branch_clusters": branch_cluster_payload(sensitivity_clusters, late_dph_min),
			"primary_metrics": sensitivity_primary,
			"replicates_expected_signs": _replicates_expected_signs(primary, sensitivity_primary),
		}
	return BranchAnalysisResult(
		records=analysis_records,
		clusters=clusters,
		metrics=metrics,
	)


def run_branch_commitment_analysis(
	artifact_dir: Path,
	report_path: Path,
	event_table: Optional[Path] = None,
	coverage_metrics: Optional[Path] = None,
	manifest_path: Optional[Path] = None,
	latent_root: Optional[Path] = None,
	bird_id: str = "PK249",
	dph_min: float = 33,
	dph_max: float = 90,
	early_dph_max: float = 45,
	late_dph_min: float = 80,
	split: str = "all",
	audio_root: Optional[Path] = None,
	roi_root: Optional[Path] = None,
	roi_format: str = "parquet",
	roi_parquet_name: str = "roi.parquet",
	max_events_per_dph: Optional[int] = 2000,
	cluster_min_k: int = DEFAULT_CLUSTER_K_RANGE[0],
	cluster_max_k: int = DEFAULT_CLUSTER_K_RANGE[1],
	bootstrap_iterations: int = 1000,
	seed: int = 0,
	use_energy_weighted: bool = False,
	run_sensitivity: bool = True,
) -> dict:
	"""Run the branch-commitment analysis and write report artifacts."""
	if event_table is not None:
		records = load_event_records_parquet(
			event_table,
			bird_id=bird_id,
			dph_min=dph_min,
			dph_max=dph_max,
		)
		event_summary = load_event_summary_for_event_table(event_table, coverage_metrics)
		inputs = {
			"source_mode": "event_table",
			"event_table": event_table.as_posix(),
			"coverage_metrics": (
				None if coverage_metrics is None else coverage_metrics.as_posix()
			),
		}
	else:
		if manifest_path is None or latent_root is None:
			raise ValueError("Full rebuild mode requires --manifest and --latent-root.")
		manifest = load_manifest(manifest_path)
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
		records = build.records
		event_summary = build.summary
		inputs = {
			"source_mode": "full_rebuild",
			"manifest_path": manifest_path.as_posix(),
			"latent_root": latent_root.as_posix(),
			"split": split,
			"roi_format": roi_format,
			"max_events_per_dph": max_events_per_dph,
		}

	inputs.update(
		{
			"bird_id": bird_id,
			"dph_min": float(dph_min),
			"dph_max": float(dph_max),
			"early_dph_max": float(early_dph_max),
			"late_dph_min": float(late_dph_min),
			"use_energy_weighted": bool(use_energy_weighted),
		}
	)
	result = analyze_branch_records(
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
		run_sensitivity=run_sensitivity,
	)
	artifact_dir.mkdir(parents=True, exist_ok=True)
	branch_table = artifact_dir / "branch_commitment.parquet"
	cluster_path = artifact_dir / "branch_clusters.json"
	metrics_path = artifact_dir / "metrics.json"
	fig_dir = artifact_dir / "figures"
	write_branch_records_parquet(result.records, branch_table)
	write_json(cluster_path, result.metrics["branch_clusters"])
	figures = make_figures(result.records, result.metrics, fig_dir)
	artifacts = {
		"branch_commitment": branch_table.as_posix(),
		"branch_clusters": cluster_path.as_posix(),
		"metrics": metrics_path.as_posix(),
	}
	result.metrics["artifacts"] = artifacts
	result.metrics["figures"] = figures
	result.metrics["report_path"] = report_path.as_posix()
	write_json(metrics_path, result.metrics)
	write_report(report_path, result.metrics, figures, artifacts)
	return result.metrics


def summarize_records_only(records: Sequence[dict]) -> dict:
	"""Create a minimal event summary when only an event table is available."""
	by_dph: dict[float, dict] = defaultdict(lambda: {"events_sampled": 0})
	for record in records:
		dph = _coerce_dph(record.get("dph"))
		if dph is None:
			continue
		by_dph[dph]["events_sampled"] += 1
	return {
		"coverage_source": "event_table_only",
		"coverage_counts_available": False,
		"events_sampled": int(len(records)),
		"events_available": None,
		"clips_seen": None,
		"clips_missing_latent": None,
		"clips_missing_roi": None,
		"roi_events_total": None,
		"roi_events_without_windows": None,
		"by_dph": {
			str(_format_dph_key(dph)): dict(values)
			for dph, values in sorted(by_dph.items())
		},
	}


def load_event_summary_for_event_table(
	event_table: Path,
	coverage_metrics: Optional[Path] = None,
) -> dict:
	"""Load sibling metrics coverage if available for event-table mode."""
	candidates = []
	if coverage_metrics is not None:
		candidates.append(coverage_metrics)
	candidates.append(event_table.parent / "metrics.json")
	for path in candidates:
		if path is None or not path.exists():
			continue
		try:
			payload = json.loads(path.read_text(encoding="utf-8"))
		except Exception:
			continue
		event_summary = payload.get("event_summary")
		if isinstance(event_summary, dict):
			event_summary = dict(event_summary)
			event_summary["coverage_source"] = path.as_posix()
			event_summary["coverage_counts_available"] = True
			return event_summary
	return {}


def branch_cluster_payload(result: BranchClusterResult, late_dph_min: float) -> dict:
	"""Return serializable cluster fit details."""
	return {
		"late_dph_min": float(late_dph_min),
		"late_event_count": int(result.late_event_count),
		"best_k": int(result.best_k),
		"silhouette_scores": result.silhouette_scores,
		"centers": result.centers.astype(float).tolist(),
		"distance_scale": float(result.distance_scale),
		"late_branch_distance_medians": result.late_branch_distance_medians,
	}


def grouped_metric_summary(records: Sequence[dict], key: str) -> list[dict]:
	"""Summarize a scalar record field by dph with median and IQR."""
	grouped: dict[float, list[float]] = defaultdict(list)
	for record in records:
		dph = _coerce_dph(record.get("dph"))
		value = record.get(key)
		if dph is None or value is None:
			continue
		try:
			value_f = float(value)
		except (TypeError, ValueError):
			continue
		if math.isfinite(value_f):
			grouped[dph].append(value_f)
	return [
		{
			"dph": float(dph),
			"n": int(len(values)),
			"median": float(np.median(values)),
			"p25": float(np.percentile(values, 25)),
			"p75": float(np.percentile(values, 75)),
		}
		for dph, values in sorted(grouped.items())
	]


def coverage_bias_summary(event_summary: dict) -> dict:
	"""Summarize whether coverage/skips are visible and uneven by age."""
	if not event_summary:
		return {
			"counts_available": False,
			"note": "Coverage/skips are unavailable; event-table input has no source metrics.",
		}
	if not event_summary.get("coverage_counts_available", True):
		return {
			"counts_available": False,
			"note": "Coverage/skips are unavailable; event-table input has no source metrics.",
		}
	roi_total = _optional_float(event_summary.get("roi_events_total"))
	roi_without = _optional_float(event_summary.get("roi_events_without_windows"))
	without_rate = (
		float(roi_without / roi_total)
		if roi_total is not None and roi_total > 0 and roi_without is not None
		else None
	)
	by_dph = event_summary.get("by_dph") or {}
	rates = []
	for row in by_dph.values():
		total = _optional_float(row.get("roi_events_total"))
		without = _optional_float(row.get("roi_events_without_windows"))
		if total is not None and total > 0 and without is not None:
			rates.append(float(without / total))
	rate_range = (
		[float(np.min(rates)), float(np.max(rates))]
		if rates else None
	)
	flag = bool(rate_range and (rate_range[1] - rate_range[0]) >= 0.10)
	note = (
		"Coverage/skips vary across dph and could plausibly bias branch metrics."
		if flag else
		"Coverage/skips are summarized by dph; no large dph skip-rate range was flagged."
	)
	return {
		"counts_available": True,
		"roi_events_without_windows_rate": without_rate,
		"roi_without_window_rate_by_dph_range": rate_range,
		"potential_bias_flag": flag,
		"note": note,
	}


def write_branch_records_parquet(records: Sequence[dict], path: Path) -> None:
	"""Write branch-commitment event records to parquet."""
	try:
		import pyarrow as pa  # type: ignore
		import pyarrow.parquet as pq  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError("Writing branch commitment parquet requires pyarrow.") from exc
	if not records:
		raise ValueError("Cannot write an empty branch commitment table.")
	path.parent.mkdir(parents=True, exist_ok=True)
	scalar_fields = [
		"event_id",
		"clip_id",
		"clip_stem",
		"audio_dir_rel",
		"bird_id_norm",
		"regime",
		"split",
		"dph",
		"roi_index",
		"onset_sec",
		"offset_sec",
		"duration_sec",
		"n_windows",
		"start_time_first_sec",
		"start_time_last_sec",
		"latent_dim",
		"mean_energy",
		"energy_weight_sum",
		"latent_norm",
		"variance_mean",
		"latent_schema_version",
		"branch_cluster",
		"branch_nearest_distance",
		"branch_second_distance",
		"branch_distance_margin",
		"branch_confidence",
		"branch_entropy",
		"branch_probability_margin",
		"branch_within_standardized_distance",
	]
	list_fields = ["mu", "variance", "mu_energy_weighted"]
	arrays = []
	names = []
	for field in scalar_fields:
		arrays.append(pa.array([record.get(field) for record in records]))
		names.append(field)
	for field in list_fields:
		arrays.append(
			pa.array(
				[record.get(field) for record in records],
				type=pa.list_(pa.float32()),
			)
		)
		names.append(field)
	pq.write_table(pa.Table.from_arrays(arrays, names=names), path.as_posix())


def make_figures(records: Sequence[dict], metrics: dict, out_dir: Path) -> dict[str, str]:
	"""Create branch-commitment figures and return PNG paths."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_dir.mkdir(parents=True, exist_ok=True)
	figures: dict[str, str] = {}

	fig, ax = plt.subplots(figsize=(8, 4.5))
	_plot_metric_line(
		metrics["primary_metrics"]["branch_confidence"],
		ax,
		"confidence",
		"#54A24B",
	)
	_plot_metric_line(
		metrics["primary_metrics"]["branch_entropy"],
		ax,
		"entropy",
		"#E45756",
	)
	ax.set_xlabel("dph")
	ax.set_ylabel("median")
	ax.set_title("Branch confidence and entropy by age")
	ax.legend()
	_save_figure(fig, out_dir / "branch_confidence_entropy")
	figures["branch_confidence_entropy"] = (
		out_dir / "branch_confidence_entropy.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	_plot_metric_line(
		metrics["primary_metrics"]["branch_nearest_distance"],
		ax,
		"nearest distance",
		"#4C78A8",
	)
	ax.set_xlabel("dph")
	ax.set_ylabel("standardized latent distance")
	ax.set_title("Distance to nearest adult-like branch")
	_save_figure(fig, out_dir / "branch_nearest_distance")
	figures["branch_nearest_distance"] = (
		out_dir / "branch_nearest_distance.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	_plot_metric_line(
		metrics["primary_metrics"]["branch_distance_margin"],
		ax,
		"distance margin",
		"#72B7B2",
	)
	_plot_metric_line(
		metrics["primary_metrics"]["branch_probability_margin"],
		ax,
		"probability margin",
		"#F58518",
	)
	ax.set_xlabel("dph")
	ax.set_ylabel("median margin")
	ax.set_title("Adult-like branch margins by age")
	ax.legend()
	_save_figure(fig, out_dir / "branch_margins")
	figures["branch_margins"] = (out_dir / "branch_margins.png").as_posix()

	fig, ax = plt.subplots(figsize=(10, 4.5))
	_plot_branch_composition(records, ax)
	ax.set_title("Adult-like branch composition by age")
	_save_figure(fig, out_dir / "branch_composition")
	figures["branch_composition"] = (out_dir / "branch_composition.png").as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	_plot_metric_line(
		metrics["primary_metrics"]["branch_within_standardized_distance"],
		ax,
		"within-branch standardized distance",
		"#B279A2",
	)
	ax.set_xlabel("dph")
	ax.set_ylabel("median distance / late branch median")
	ax.set_title("Within-branch dispersion by age")
	_save_figure(fig, out_dir / "within_branch_dispersion")
	figures["within_branch_dispersion"] = (
		out_dir / "within_branch_dispersion.png"
	).as_posix()

	fig, axes = plt.subplots(2, 3, figsize=(12, 7))
	_plot_control_panel(records, metrics, axes.ravel())
	fig.suptitle("Branch commitment controls and coverage")
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	_save_figure(fig, out_dir / "branch_bias_controls")
	figures["branch_bias_controls"] = (
		out_dir / "branch_bias_controls.png"
	).as_posix()
	return figures


def write_report(
	report_path: Path,
	metrics: dict,
	figures: dict[str, str],
	artifacts: dict[str, str],
) -> None:
	"""Write a figure-ready markdown branch-commitment report."""
	report_path.parent.mkdir(parents=True, exist_ok=True)

	def rel(path: str) -> str:
		return os.path.relpath(path, report_path.parent.as_posix())

	cluster = metrics["branch_clusters"]
	primary = metrics["primary_metrics"]
	lines = [
		"# Branch Commitment Developmental AVA Report",
		"",
		"## Summary",
		"",
		f"- Events analyzed: {metrics['event_summary'].get('events_sampled', len(primary))}.",
		f"- Adult-like branches: K={cluster['best_k']} from "
		f"{cluster['late_event_count']} late-age events.",
		_metric_summary_line("Confidence", primary["branch_confidence"]),
		_metric_summary_line("Entropy", primary["branch_entropy"]),
		_metric_summary_line("Nearest branch distance", primary["branch_nearest_distance"]),
		_metric_summary_line("Distance margin", primary["branch_distance_margin"]),
		_metric_summary_line("Probability margin", primary["branch_probability_margin"]),
		f"- Sensitivity signs replicate: {_sensitivity_text(metrics)}.",
		f"- Coverage/skips: {metrics['coverage_bias']['note']}",
		"",
		"## Figures",
		"",
		f"![Branch confidence and entropy]({rel(figures['branch_confidence_entropy'])})",
		"",
		f"![Nearest branch distance]({rel(figures['branch_nearest_distance'])})",
		"",
		f"![Branch margins]({rel(figures['branch_margins'])})",
		"",
		f"![Branch composition]({rel(figures['branch_composition'])})",
		"",
		f"![Within-branch dispersion]({rel(figures['within_branch_dispersion'])})",
		"",
		f"![Bias controls]({rel(figures['branch_bias_controls'])})",
		"",
		"## Interpretation",
		"",
		interpret_branch_commitment(metrics),
		"",
		"## Coverage Summary",
		"",
		"```json",
		json.dumps(_jsonable(metrics["event_summary"]), indent=2, sort_keys=True),
		"```",
		"",
		"## Artifacts",
		"",
	]
	for name, path in artifacts.items():
		lines.append(f"- `{name}`: `{rel(path)}`")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret_branch_commitment(metrics: dict) -> str:
	"""Return the plain-language branch commitment interpretation."""
	primary = metrics["primary_metrics"]
	conf_up = _ci_supports(primary["branch_confidence"], 1)
	entropy_down = _ci_supports(primary["branch_entropy"], -1)
	distance_down = _ci_supports(primary["branch_nearest_distance"], -1)
	distance_margin_up = _ci_supports(primary["branch_distance_margin"], 1)
	prob_margin_up = _ci_supports(primary["branch_probability_margin"], 1)
	parts = []
	parts.append(
		"Confidence increases with age." if conf_up
		else "Confidence does not show a bootstrap-supported age increase."
	)
	parts.append(
		"Entropy decreases with age." if entropy_down
		else "Entropy does not show a bootstrap-supported age decrease."
	)
	parts.append(
		"Events move closer to adult branch centers." if distance_down
		else "Events do not move closer to adult branch centers by nearest-center distance."
	)
	parts.append(
		"Branch margins increase." if distance_margin_up and prob_margin_up
		else "Branch margins do not both show bootstrap-supported increases."
	)
	return " ".join(parts)


def _primary_metric_payloads(
	records: Sequence[dict],
	bootstrap_iterations: int,
	seed: int,
) -> dict:
	return {
		key: radius_age_metrics(
			records,
			radius_key=key,
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
		)
		for key in PRIMARY_METRICS
	}


def _replicates_expected_signs(primary: dict, sensitivity: dict) -> dict:
	by_metric = {}
	for key, expected in PRIMARY_METRICS.items():
		primary_ok = _rho_matches(primary.get(key, {}), expected)
		sensitivity_ok = _rho_matches(sensitivity.get(key, {}), expected)
		by_metric[key] = {
			"expected_sign": int(expected),
			"primary_matches": bool(primary_ok),
			"sensitivity_matches": bool(sensitivity_ok),
			"replicates": bool(primary_ok and sensitivity_ok),
		}
	return {
		"all_primary_metrics": bool(all(row["replicates"] for row in by_metric.values())),
		"by_metric": by_metric,
	}


def _rho_matches(metric: dict, expected_sign: int) -> bool:
	rho = metric.get("spearman_rho")
	if rho is None:
		return False
	try:
		rho_f = float(rho)
	except (TypeError, ValueError):
		return False
	return math.isfinite(rho_f) and rho_f * float(expected_sign) > 0


def _ci_supports(metric: dict, expected_sign: int) -> bool:
	ci = metric.get("bootstrap_rho_ci95")
	if not ci:
		return False
	if expected_sign > 0:
		return float(ci[0]) > 0
	return float(ci[1]) < 0


def _metric_summary_line(label: str, metric: dict) -> str:
	ci = metric.get("bootstrap_rho_ci95")
	ci_text = "n/a" if ci is None else f"[{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
	return (
		f"- {label} age Spearman rho: "
		f"{float(metric.get('spearman_rho', float('nan'))):.3f} "
		f"(bootstrap CI {ci_text})."
	)


def _sensitivity_text(metrics: dict) -> str:
	sensitivity = metrics.get("sensitivity")
	if not sensitivity:
		return "not run"
	rep = sensitivity.get("replicates_expected_signs", {})
	return "yes" if rep.get("all_primary_metrics") else "partial/no"


def _plot_metric_line(metric: dict, ax, label: str, color: str) -> None:
	rows = metric.get("per_dph") or []
	if not rows:
		return
	dph = np.asarray([row["dph"] for row in rows], dtype=np.float64)
	median = np.asarray([row["median"] for row in rows], dtype=np.float64)
	p25 = np.asarray([row["p25"] for row in rows], dtype=np.float64)
	p75 = np.asarray([row["p75"] for row in rows], dtype=np.float64)
	ax.fill_between(dph, p25, p75, color=color, alpha=0.15)
	ax.plot(dph, median, color=color, marker="o", markersize=3, linewidth=1.5, label=label)


def _plot_branch_composition(records: Sequence[dict], ax) -> None:
	grouped: dict[float, list[int]] = defaultdict(list)
	for record in records:
		cluster = record.get("branch_cluster")
		dph = _coerce_dph(record.get("dph"))
		if cluster is not None and dph is not None:
			grouped[dph].append(int(cluster))
	dph_values = sorted(grouped)
	clusters = sorted({cluster for values in grouped.values() for cluster in values})
	bottom = np.zeros(len(dph_values), dtype=np.float64)
	for cluster in clusters:
		values = []
		for dph in dph_values:
			arr = np.asarray(grouped[dph], dtype=np.int32)
			values.append(float(np.mean(arr == cluster)) if arr.size else 0.0)
		ax.bar(dph_values, values, bottom=bottom, width=0.8, label=f"B{cluster}")
		bottom += np.asarray(values)
	ax.set_xlabel("dph")
	ax.set_ylabel("fraction of events")
	ax.set_ylim(0.0, 1.0)
	ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def _plot_control_panel(records: Sequence[dict], metrics: dict, axes) -> None:
	for ax, key, title in zip(
		axes[:4],
		("duration_sec", "n_windows", "mean_energy", "variance_mean"),
		("duration", "n windows", "mean energy", "mean variance"),
	):
		rows = metrics["controls_by_dph"].get(key, [])
		_plot_rows(rows, ax, title, "#4C78A8")
		ax.set_xlabel("dph")
	ax = axes[4]
	coverage = metrics.get("event_summary", {}).get("by_dph") or {}
	_plot_coverage_counts(coverage, ax)
	ax.set_title("skip counts")
	ax.set_xlabel("dph")
	ax = axes[5]
	rows = grouped_metric_summary(records, "branch_within_standardized_distance")
	_plot_rows(rows, ax, "within branch distance", "#B279A2")
	ax.set_xlabel("dph")


def _plot_rows(rows: Sequence[dict], ax, title: str, color: str) -> None:
	if not rows:
		ax.set_title(f"{title} unavailable")
		return
	dph = np.asarray([row["dph"] for row in rows], dtype=np.float64)
	median = np.asarray([row["median"] for row in rows], dtype=np.float64)
	p25 = np.asarray([row["p25"] for row in rows], dtype=np.float64)
	p75 = np.asarray([row["p75"] for row in rows], dtype=np.float64)
	ax.fill_between(dph, p25, p75, color=color, alpha=0.15)
	ax.plot(dph, median, color=color, marker="o", markersize=2, linewidth=1.2)
	ax.set_title(title)


def _plot_coverage_counts(coverage: dict, ax) -> None:
	if not coverage:
		ax.text(0.5, 0.5, "skip counts unavailable", ha="center", va="center")
		return
	keys = ("clips_missing_latent", "clips_missing_roi", "roi_events_without_windows")
	colors = ("#E45756", "#F58518", "#4C78A8")
	for key, color in zip(keys, colors):
		x = []
		y = []
		for dph_key, row in sorted(coverage.items(), key=lambda item: float(item[0])):
			value = row.get(key)
			if value is None:
				continue
			x.append(float(dph_key))
			y.append(float(value))
		if x:
			ax.plot(x, y, marker="o", markersize=2, linewidth=1.2, label=key, color=color)
	if ax.lines:
		ax.legend(fontsize=7)
	else:
		ax.text(0.5, 0.5, "skip counts unavailable", ha="center", va="center")


def _save_figure(fig, base_path: Path) -> None:
	fig.savefig(f"{base_path.as_posix()}.png", dpi=180, bbox_inches="tight")
	fig.savefig(f"{base_path.as_posix()}.pdf", bbox_inches="tight")
	import matplotlib.pyplot as plt

	plt.close(fig)


def _coerce_dph(value: Any) -> Optional[float]:
	try:
		dph = float(value)
	except (TypeError, ValueError):
		return None
	return dph if math.isfinite(dph) else None


def _format_dph_key(dph: float) -> Any:
	return int(dph) if float(dph).is_integer() else float(dph)


def _optional_float(value: Any) -> Optional[float]:
	try:
		value_f = float(value)
	except (TypeError, ValueError):
		return None
	return value_f if math.isfinite(value_f) else None


def _finite_or_none(value: Any) -> Optional[float]:
	try:
		value_f = float(value)
	except (TypeError, ValueError):
		return None
	return value_f if math.isfinite(value_f) else None


def _jsonable(value: Any) -> Any:
	if isinstance(value, dict):
		return {str(k): _jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	if isinstance(value, np.ndarray):
		return _jsonable(value.tolist())
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		value = float(value)
		return value if math.isfinite(value) else None
	if isinstance(value, float):
		return value if math.isfinite(value) else None
	return value
