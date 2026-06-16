"""Shotgun VAE developmental branch-commitment helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ava.analysis.branch_commitment import PRIMARY_METRICS
from ava.analysis.developmental_replication import (
	CRITERION_METRICS,
	DEFAULT_BIRD_IDS,
	NEGATIVE_CONTROL_METRIC,
)
from ava.models.fixed_window_config import FixedWindowExperimentConfig


DEFAULT_SHOTGUN_BEAD = "autoencoded-vocal-analysis-obi.5"
SHOTGUN_MODEL_ID = "shotgun_vae"
DEFAULT_CONFIG_OVERRIDES = {
	"preprocess": {
		"min_freq": 300.0,
		"spec_min_val": 1.0,
	},
	"training": {
		"kl_beta": 1.0,
		"kl_warmup_epochs": 20,
	},
}


def load_manifest(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def filter_manifest_by_birds_and_dph(
	manifest: dict,
	bird_ids: Sequence[str],
	dph_min: float = 33,
	dph_max: float = 90,
) -> dict:
	"""Filter a birdsong manifest by bird id and finite dph range."""
	bird_order = {str(bird).strip().upper(): idx for idx, bird in enumerate(bird_ids)}
	out = {"train": [], "test": []}
	for split in ("train", "test"):
		rows = []
		for row in manifest.get(split, []):
			bird = str(row.get("bird_id_norm") or row.get("bird_id_raw") or "").strip().upper()
			if bird not in bird_order:
				continue
			dph = _optional_float(row.get("dph"))
			if dph is None or dph < float(dph_min) or dph > float(dph_max):
				continue
			payload = dict(row)
			payload["split"] = row.get("split", split)
			rows.append(payload)
		rows.sort(
			key=lambda row: (
				bird_order[str(row.get("bird_id_norm") or row.get("bird_id_raw")).strip().upper()],
				float(row.get("dph")),
				str(row.get("audio_dir_rel") or row.get("audio_dir") or ""),
			)
		)
		out[split] = rows
	return out


def split_manifest_for_validation(
	manifest: dict,
	validation_fraction: float = 0.1,
	seed: int = 0,
	min_test_per_bird: int = 1,
) -> dict:
	"""Create a deterministic per-bird validation split from manifest rows."""
	validation_fraction = float(validation_fraction)
	if validation_fraction <= 0:
		return {
			"train": [dict(row, split="train") for row in manifest.get("train", [])],
			"test": [dict(row, split="test") for row in manifest.get("test", [])],
		}
	if validation_fraction >= 1:
		raise ValueError("validation_fraction must be in [0, 1).")
	if min_test_per_bird < 0:
		raise ValueError("min_test_per_bird must be non-negative.")

	all_rows = [
		dict(row)
		for split in ("train", "test")
		for row in manifest.get(split, [])
	]
	by_bird: dict[str, list[dict]] = {}
	bird_order: list[str] = []
	for row in all_rows:
		bird = str(row.get("bird_id_norm") or row.get("bird_id_raw") or "").strip().upper()
		if not bird:
			continue
		if bird not in by_bird:
			by_bird[bird] = []
			bird_order.append(bird)
		by_bird[bird].append(row)

	train: list[dict] = []
	test: list[dict] = []
	for bird in bird_order:
		rows = sorted(by_bird[bird], key=_manifest_row_order_key)
		n_rows = len(rows)
		n_test = int(round(n_rows * validation_fraction))
		if n_rows > 1:
			n_test = max(int(min_test_per_bird), n_test)
			n_test = min(n_test, n_rows - 1)
		else:
			n_test = 0
		test_indices = _evenly_spaced_seeded_indices(n_rows, n_test, bird, int(seed))
		for idx, row in enumerate(rows):
			payload = dict(row)
			if idx in test_indices:
				payload["split"] = "test"
				test.append(payload)
			else:
				payload["split"] = "train"
				train.append(payload)

	return {"train": train, "test": test}


def summarize_manifest(manifest: dict) -> dict:
	"""Summarize manifest rows by bird and split."""
	birds: dict[str, dict] = {}
	for split in ("train", "test"):
		for row in manifest.get(split, []):
			bird = str(row.get("bird_id_norm") or row.get("bird_id_raw") or "").strip().upper()
			if not bird:
				continue
			payload = birds.setdefault(
				bird,
				{
					"bird_id": bird,
					"manifest_rows": 0,
					"num_files": 0,
					"dph_values": [],
					"regimes": set(),
					"splits": set(),
					"split_counts": {},
				},
			)
			payload["manifest_rows"] += 1
			payload["num_files"] += int(row.get("num_files") or 0)
			dph = _optional_float(row.get("dph"))
			if dph is not None:
				payload["dph_values"].append(float(dph))
			if row.get("regime"):
				payload["regimes"].add(str(row["regime"]))
			payload["splits"].add(split)
			payload["split_counts"][split] = int(payload["split_counts"].get(split, 0)) + 1
	out = {}
	for bird, payload in sorted(birds.items()):
		dph_values = payload["dph_values"]
		out[bird] = {
			"bird_id": bird,
			"manifest_rows": int(payload["manifest_rows"]),
			"num_files": int(payload["num_files"]),
			"unique_dph": int(len(set(dph_values))),
			"dph_min": float(min(dph_values)) if dph_values else None,
			"dph_max": float(max(dph_values)) if dph_values else None,
			"regimes": sorted(payload["regimes"]),
			"splits": sorted(payload["splits"]),
			"split_counts": {
				key: int(value)
				for key, value in sorted(payload["split_counts"].items())
			},
		}
	return {
		"birds": out,
		"total_rows": int(sum(len(manifest.get(split, [])) for split in ("train", "test"))),
		"total_num_files": int(sum(row.get("num_files") or 0 for split in ("train", "test") for row in manifest.get(split, []))),
	}


def write_shotgun_manifests(
	manifest_path: Path,
	out_dir: Path,
	pilot_bird: str = "PK249",
	cohort_birds: Sequence[str] = DEFAULT_BIRD_IDS,
	dph_min: float = 33,
	dph_max: float = 90,
	cohort_validation_fraction: float = 0.0,
	validation_seed: int = 0,
	validation_min_per_bird: int = 1,
) -> dict:
	"""Write PK249 pilot and fixed-cohort manifests for shotgun VAE work."""
	manifest = load_manifest(manifest_path)
	out_dir.mkdir(parents=True, exist_ok=True)
	pilot_bird = str(pilot_bird).strip().upper()
	cohort_birds = [str(bird).strip().upper() for bird in cohort_birds]
	pilot = filter_manifest_by_birds_and_dph(
		manifest,
		[pilot_bird],
		dph_min=dph_min,
		dph_max=dph_max,
	)
	cohort = filter_manifest_by_birds_and_dph(
		manifest,
		cohort_birds,
		dph_min=dph_min,
		dph_max=dph_max,
	)
	if cohort_validation_fraction > 0:
		cohort = split_manifest_for_validation(
			cohort,
			validation_fraction=cohort_validation_fraction,
			seed=validation_seed,
			min_test_per_bird=validation_min_per_bird,
		)
	val_suffix = _validation_suffix(cohort_validation_fraction)
	paths = {
		"pilot_manifest": (out_dir / f"{pilot_bird.lower()}_{_format_dph(dph_min)}_{_format_dph(dph_max)}_manifest.json").as_posix(),
		"cohort_manifest": (out_dir / f"fixed_11bird_{_format_dph(dph_min)}_{_format_dph(dph_max)}_manifest{val_suffix}.json").as_posix(),
		"summary": (out_dir / "shotgun_cohort_manifest_summary.json").as_posix(),
	}
	_write_json(Path(paths["pilot_manifest"]), pilot)
	_write_json(Path(paths["cohort_manifest"]), cohort)
	summary = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"source_manifest": manifest_path.as_posix(),
		"dph_min": float(dph_min),
		"dph_max": float(dph_max),
		"pilot_bird": pilot_bird,
		"cohort_birds": cohort_birds,
		"cohort_validation": {
			"fraction": float(cohort_validation_fraction),
			"seed": int(validation_seed),
			"min_test_per_bird": int(validation_min_per_bird),
			"enabled": bool(cohort_validation_fraction > 0),
		},
		"pilot": summarize_manifest(pilot),
		"cohort": summarize_manifest(cohort),
		"artifacts": paths,
	}
	_write_json(Path(paths["summary"]), summary)
	return summary


def write_shotgun_config(
	base_config: Path,
	out_path: Path,
	epochs: int,
	min_freq: float = 300.0,
	spec_min_val: float = 1.0,
	kl_beta: float = 1.0,
	kl_warmup_epochs: int = 20,
	test_freq: Optional[int] = None,
	stopping_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
	"""Write a fixed-window shotgun VAE config with the selected overrides."""
	cfg = FixedWindowExperimentConfig.from_yaml(base_config.as_posix())
	cfg.preprocess.min_freq = float(min_freq)
	cfg.preprocess.spec_min_val = float(spec_min_val)
	cfg.training.kl_beta = float(kl_beta)
	cfg.training.kl_warmup_epochs = int(kl_warmup_epochs)
	cfg.training.epochs = int(epochs)
	if test_freq is not None:
		cfg.training.test_freq = int(test_freq)
	if stopping_kwargs is not None:
		cfg.training.stopping_kwargs = dict(stopping_kwargs)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cfg.to_yaml(out_path.as_posix())
	overrides = {
		"preprocess.min_freq": float(min_freq),
		"preprocess.spec_min_val": float(spec_min_val),
		"training.kl_beta": float(kl_beta),
		"training.kl_warmup_epochs": int(kl_warmup_epochs),
		"training.epochs": int(epochs),
	}
	if test_freq is not None:
		overrides["training.test_freq"] = int(test_freq)
	if stopping_kwargs is not None:
		overrides["training.stopping_kwargs"] = dict(stopping_kwargs)
	return {
		"base_config": base_config.as_posix(),
		"config": out_path.as_posix(),
		"epochs": int(epochs),
		"overrides": overrides,
	}


def compare_replication_metrics(
	baseline_per_bird: dict,
	shotgun_per_bird: dict,
	baseline_label: str = "ava_latent",
	shotgun_label: str = SHOTGUN_MODEL_ID,
) -> dict:
	"""Compare baseline and shotgun developmental replication metrics."""
	common = sorted(
		bird for bird in baseline_per_bird
		if bird in shotgun_per_bird
		and baseline_per_bird[bird].get("status") == "analyzed"
		and shotgun_per_bird[bird].get("status") == "analyzed"
	)
	rows = []
	criterion_deltas = []
	for bird in common:
		baseline = baseline_per_bird[bird]
		shotgun = shotgun_per_bird[bird]
		row = {
			"bird_id": bird,
			"baseline_events": int(baseline.get("events_analyzed") or 0),
			"shotgun_events": int(shotgun.get("events_analyzed") or 0),
			"metrics": {},
			"negative_controls": {},
		}
		for metric, expected in PRIMARY_METRICS.items():
			base_rho = _metric_rho(baseline, metric)
			shotgun_rho = _metric_rho(shotgun, metric)
			delta = (
				None if base_rho is None or shotgun_rho is None
				else float(shotgun_rho - base_rho)
			)
			expected_delta = (
				None if delta is None
				else float((shotgun_rho * expected) - (base_rho * expected))
			)
			if metric in CRITERION_METRICS and expected_delta is not None:
				criterion_deltas.append(expected_delta)
			row["metrics"][metric] = {
				"expected_sign": int(expected),
				"baseline_rho": base_rho,
				"shotgun_rho": shotgun_rho,
				"rho_delta": delta,
				"expected_direction_delta": expected_delta,
				"baseline_matches_expected": _matches(base_rho, expected),
				"shotgun_matches_expected": _matches(shotgun_rho, expected),
			}
		base_radius = _radius_rho(baseline)
		shotgun_radius = _radius_rho(shotgun)
		row["negative_controls"][NEGATIVE_CONTROL_METRIC] = {
			"baseline_rho": base_radius,
			"shotgun_rho": shotgun_radius,
			"rho_delta": (
				None if base_radius is None or shotgun_radius is None
				else float(shotgun_radius - base_radius)
			),
		}
		row["coverage"] = {
			"baseline_counts_available": bool(
				baseline.get("coverage_bias", {}).get("counts_available")
			),
			"shotgun_counts_available": bool(
				shotgun.get("coverage_bias", {}).get("counts_available")
			),
			"event_count_delta": int(row["shotgun_events"] - row["baseline_events"]),
		}
		rows.append(row)
	strength_delta = (
		float(np.mean(criterion_deltas)) if criterion_deltas else None
	)
	shotgun_sign_pass = _all_criterion_signs(rows, "shotgun_matches_expected")
	baseline_sign_pass = _all_criterion_signs(rows, "baseline_matches_expected")
	conclusion = _comparison_conclusion(
		strength_delta=strength_delta,
		baseline_sign_pass=baseline_sign_pass,
		shotgun_sign_pass=shotgun_sign_pass,
		n_common=len(common),
	)
	return {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"baseline_label": baseline_label,
		"shotgun_label": shotgun_label,
		"common_analyzed_birds": common,
		"n_common_analyzed_birds": int(len(common)),
		"mean_expected_direction_delta": strength_delta,
		"baseline_all_criterion_signs": bool(baseline_sign_pass),
		"shotgun_all_criterion_signs": bool(shotgun_sign_pass),
		"conclusion": conclusion,
		"per_bird": rows,
	}


def write_comparison_artifacts(
	comparison: dict,
	out_dir: Path,
	report_path: Path,
) -> dict:
	"""Write shotgun-vs-baseline metrics, figures, and report."""
	out_dir.mkdir(parents=True, exist_ok=True)
	fig_dir = out_dir / "figures"
	figures = make_comparison_figures(comparison, fig_dir)
	artifacts = {
		"metrics": (out_dir / "shotgun_comparison_metrics.json").as_posix(),
	}
	_write_json(Path(artifacts["metrics"]), comparison)
	write_comparison_report(report_path, comparison, figures, artifacts)
	return {
		"figures": figures,
		"artifacts": artifacts,
		"report_path": report_path.as_posix(),
	}


def make_comparison_figures(comparison: dict, out_dir: Path) -> dict[str, str]:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_dir.mkdir(parents=True, exist_ok=True)
	figures = {}
	rows = comparison.get("per_bird", [])
	metrics = list(CRITERION_METRICS)
	fig, ax = plt.subplots(figsize=(9, 4.5))
	if rows:
		x = np.arange(len(metrics), dtype=np.float64)
		values = []
		for metric in metrics:
			deltas = [
				row.get("metrics", {}).get(metric, {}).get("expected_direction_delta")
				for row in rows
			]
			deltas = [float(delta) for delta in deltas if delta is not None]
			values.append(float(np.mean(deltas)) if deltas else np.nan)
		ax.axhline(0, color="0.5", linewidth=1.0)
		ax.bar(x, values, color="#4C78A8")
		ax.set_xticks(x)
		ax.set_xticklabels([metric.replace("branch_", "").replace("_", "\n") for metric in metrics])
		ax.set_ylabel("shotgun minus baseline, expected direction")
	else:
		ax.text(0.5, 0.5, "no common analyzed birds", ha="center", va="center")
		ax.axis("off")
	ax.set_title("Shotgun branch metric deltas")
	_save_figure(fig, out_dir / "branch_metric_deltas")
	figures["branch_metric_deltas"] = (out_dir / "branch_metric_deltas.png").as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	if rows:
		birds = [row["bird_id"] for row in rows]
		base = [
			row.get("negative_controls", {}).get(NEGATIVE_CONTROL_METRIC, {}).get("baseline_rho")
			for row in rows
		]
		shot = [
			row.get("negative_controls", {}).get(NEGATIVE_CONTROL_METRIC, {}).get("shotgun_rho")
			for row in rows
		]
		x = np.arange(len(birds), dtype=np.float64)
		ax.axhline(0, color="0.5", linewidth=1.0)
		ax.scatter(x - 0.08, base, label=comparison.get("baseline_label", "baseline"), color="#4C78A8")
		ax.scatter(x + 0.08, shot, label=comparison.get("shotgun_label", "shotgun"), color="#F58518")
		ax.set_xticks(x)
		ax.set_xticklabels(birds, rotation=30)
		ax.set_ylabel("Poincare radius rho")
		ax.legend()
	else:
		ax.text(0.5, 0.5, "no common analyzed birds", ha="center", va="center")
		ax.axis("off")
	ax.set_title("Poincare radius negative control")
	_save_figure(fig, out_dir / "poincare_radius_comparison")
	figures["poincare_radius_comparison"] = (out_dir / "poincare_radius_comparison.png").as_posix()
	return figures


def write_comparison_report(
	report_path: Path,
	comparison: dict,
	figures: dict[str, str],
	artifacts: dict[str, str],
) -> None:
	report_path.parent.mkdir(parents=True, exist_ok=True)
	lines = [
		"# Shotgun VAE Developmental Branch Commitment Comparison",
		"",
		"## Summary",
		"",
		f"- Common analyzed birds: {comparison.get('n_common_analyzed_birds', 0)}.",
		f"- Mean expected-direction branch delta: {_fmt(comparison.get('mean_expected_direction_delta'))}.",
		f"- Conclusion: {comparison.get('conclusion', 'unavailable')}.",
		"",
		"## Figures",
		"",
		f"![Branch metric deltas]({_rel(figures['branch_metric_deltas'], report_path)})",
		"",
		f"![Poincare radius comparison]({_rel(figures['poincare_radius_comparison'], report_path)})",
		"",
		"## Per-Bird Summary",
		"",
	]
	for row in comparison.get("per_bird", []):
		conf = row["metrics"].get("branch_confidence", {})
		ent = row["metrics"].get("branch_entropy", {})
		dist = row["metrics"].get("branch_nearest_distance", {})
		margin = row["metrics"].get("branch_distance_margin", {})
		lines.append(
			f"- `{row['bird_id']}`: confidence delta {_fmt(conf.get('expected_direction_delta'))}, "
			f"entropy delta {_fmt(ent.get('expected_direction_delta'))}, "
			f"distance delta {_fmt(dist.get('expected_direction_delta'))}, "
			f"margin delta {_fmt(margin.get('expected_direction_delta'))}; "
			f"event delta {row.get('coverage', {}).get('event_count_delta')}."
		)
	if not comparison.get("per_bird"):
		lines.append("- No common analyzed birds.")
	lines.extend(["", "## Artifacts", ""])
	for name, path in artifacts.items():
		lines.append(f"- `{name}`: `{_rel(path, report_path)}`")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_rho(payload: dict, metric: str) -> Optional[float]:
	return _optional_float(
		payload.get("primary_metrics", {}).get(metric, {}).get("spearman_rho")
	)


def _radius_rho(payload: dict) -> Optional[float]:
	return _optional_float(
		payload.get("negative_controls", {})
		.get(NEGATIVE_CONTROL_METRIC, {})
		.get("radius_age", {})
		.get("spearman_rho")
	)


def _matches(value: Optional[float], expected: int) -> Optional[bool]:
	if value is None:
		return None
	return bool(float(value) * float(expected) > 0)


def _all_criterion_signs(rows: Sequence[dict], key: str) -> bool:
	if not rows:
		return False
	for row in rows:
		for metric in CRITERION_METRICS:
			if row.get("metrics", {}).get(metric, {}).get(key) is not True:
				return False
	return True


def _comparison_conclusion(
	strength_delta: Optional[float],
	baseline_sign_pass: bool,
	shotgun_sign_pass: bool,
	n_common: int,
) -> str:
	if n_common == 0 or strength_delta is None:
		return "unavailable: no common analyzed birds"
	if shotgun_sign_pass and strength_delta > 0.05:
		return "shotgun strengthens the branch-commitment claim"
	if baseline_sign_pass and (not shotgun_sign_pass or strength_delta < -0.05):
		return "shotgun weakens the branch-commitment claim"
	return "shotgun changes or approximately matches the branch-commitment claim"


def _optional_float(value: Any) -> Optional[float]:
	try:
		out = float(value)
	except (TypeError, ValueError):
		return None
	return out if math.isfinite(out) else None


def _manifest_row_order_key(row: dict) -> tuple:
	dph = _optional_float(row.get("dph"))
	return (
		float("inf") if dph is None else float(dph),
		str(row.get("audio_dir_rel") or row.get("audio_dir") or ""),
		str(row.get("roi_dir") or ""),
	)


def _evenly_spaced_seeded_indices(
	n_rows: int,
	n_select: int,
	bird: str,
	seed: int,
) -> set[int]:
	if n_select <= 0 or n_rows <= 0:
		return set()
	if n_select >= n_rows:
		return set(range(n_rows))
	digest = hashlib.sha256(f"{int(seed)}:{bird}".encode("utf-8")).digest()
	rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
	indices: set[int] = set()
	for bucket in np.array_split(np.arange(n_rows), int(n_select)):
		if bucket.size:
			indices.add(int(rng.choice(bucket)))
	return indices


def _validation_suffix(validation_fraction: float) -> str:
	if validation_fraction <= 0:
		return ""
	percent = float(validation_fraction) * 100.0
	if abs(percent - round(percent)) < 1e-9:
		return f"_val{int(round(percent)):02d}"
	return f"_val{validation_fraction:g}".replace(".", "p")


def _format_dph(value: float) -> str:
	return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _fmt(value: Any) -> str:
	value_f = _optional_float(value)
	return "n/a" if value_f is None else f"{value_f:.3f}"


def _rel(path: str, report_path: Path) -> str:
	return os.path.relpath(str(path), report_path.parent.as_posix())


def _write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_figure(fig, prefix: Path) -> None:
	fig.tight_layout()
	fig.savefig(Path(f"{prefix.as_posix()}.png"), dpi=180)
	fig.savefig(Path(f"{prefix.as_posix()}.pdf"))
	import matplotlib.pyplot as plt

	plt.close(fig)
