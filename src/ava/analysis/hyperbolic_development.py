"""Post-hoc hyperbolic developmental analyses for AVA latent exports.

The functions in this module operate on the canonical AVA latent sequence
exports (``.npz`` arrays plus ``.json`` metadata) and per-directory ROI bundles.
They aggregate fixed-window latent sequences back to ROI/protosyllable events,
fit late-age adult-like clusters, and produce a two-dimensional Poincare-ball
embedding for radius-vs-development analyses.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ava.data.manifest_paths import resolve_manifest_entry_paths


DEFAULT_CLUSTER_K_RANGE = (4, 12)
DEFAULT_RUN_BEAD = "hyperbolic-development"
EPS = 1e-8


@dataclass
class EventBuildResult:
	"""Event records plus coverage/sampling summary."""

	records: list[dict]
	summary: dict


@dataclass
class ClusterResult:
	"""Adult-like cluster fit and assignments."""

	best_k: int
	silhouette_scores: dict[str, float]
	centers: np.ndarray
	labels: np.ndarray
	distances: np.ndarray
	confidence: np.ndarray
	entropy: np.ndarray
	late_event_count: int


@dataclass
class EmbeddingResult:
	"""Poincare embedding coordinates and optimizer diagnostics."""

	points: np.ndarray
	loss_history: list[float]
	edge_count: int
	target_scale: float


def load_manifest(path: Path) -> dict:
	"""Load a birdsong manifest JSON file."""
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
	"""Write a JSON payload with numpy scalars converted to plain Python."""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
		encoding="utf-8",
	)


def _to_jsonable(value: Any) -> Any:
	if isinstance(value, dict):
		return {str(k): _to_jsonable(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_jsonable(v) for v in value]
	if isinstance(value, np.ndarray):
		return _to_jsonable(value.tolist())
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		value = float(value)
		return value if math.isfinite(value) else None
	if isinstance(value, float):
		return value if math.isfinite(value) else None
	return value


def _coerce_dph(value: Any) -> Optional[float]:
	if value is None:
		return None
	try:
		dph = float(value)
	except (TypeError, ValueError):
		return None
	return dph if math.isfinite(dph) else None


def _select_entries(
	manifest: dict,
	split: str,
	bird_id: str,
	dph_min: Optional[float],
	dph_max: Optional[float],
) -> list[dict]:
	if split == "train":
		entries = list(manifest.get("train", []))
	elif split == "test":
		entries = list(manifest.get("test", []))
	elif split == "all":
		entries = list(manifest.get("train", [])) + list(manifest.get("test", []))
	else:
		raise ValueError("split must be 'train', 'test', or 'all'.")

	bird_id = str(bird_id).strip().upper()
	selected = []
	for entry in entries:
		if str(entry.get("bird_id_norm", "")).strip().upper() != bird_id:
			continue
		dph = _coerce_dph(entry.get("dph"))
		if dph is None:
			continue
		if dph_min is not None and dph < float(dph_min):
			continue
		if dph_max is not None and dph > float(dph_max):
			continue
		selected.append(entry)
	return selected


def _list_wavs(audio_dir: str) -> list[str]:
	try:
		names = os.listdir(audio_dir)
	except FileNotFoundError:
		return []
	return [
		os.path.join(audio_dir, name)
		for name in sorted(names)
		if name.lower().endswith(".wav") and not name.startswith("._")
	]


def _clip_id_for_entry(entry: dict, clip_stem: str) -> str:
	audio_dir_rel = entry.get("audio_dir_rel") or "."
	if audio_dir_rel in ("", "."):
		return str(clip_stem)
	return (Path(str(audio_dir_rel)) / str(clip_stem)).as_posix()


def latent_export_paths(latent_root: Path, clip_id: str) -> tuple[Path, Path]:
	"""Return the expected ``.npz`` and ``.json`` paths for a clip id."""
	prefix = latent_root / Path(str(clip_id))
	return Path(f"{prefix.as_posix()}.npz"), Path(f"{prefix.as_posix()}.json")


def load_latent_export(latent_root: Path, clip_id: str) -> tuple[dict, dict]:
	"""Load one AVA latent export by clip id."""
	npz_path, json_path = latent_export_paths(latent_root, clip_id)
	if not npz_path.exists() or not json_path.exists():
		raise FileNotFoundError(f"Missing latent export for {clip_id}")
	with np.load(npz_path.as_posix()) as npz:
		latent = {key: npz[key].copy() for key in npz.files}
	with open(json_path, "r", encoding="utf-8") as handle:
		metadata = json.load(handle)
	return latent, metadata


def load_roi_parquet(path: Path) -> dict[str, list[tuple[float, float]]]:
	"""Load a per-directory ROI parquet bundle."""
	try:
		import pyarrow.parquet as pq  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError(
			"Parquet ROI analysis requires pyarrow. Install pyarrow or use txt ROIs."
		) from exc

	table = pq.read_table(
		path.as_posix(),
		columns=["clip_stem", "onsets_sec", "offsets_sec"],
	)
	payload = table.to_pydict()
	stems = payload.get("clip_stem") or []
	onsets = payload.get("onsets_sec") or []
	offsets = payload.get("offsets_sec") or []
	if not (len(stems) == len(onsets) == len(offsets)):
		raise ValueError(f"Malformed ROI parquet payload: {path.as_posix()}")

	out: dict[str, list[tuple[float, float]]] = {}
	for stem, ons, offs in zip(stems, onsets, offsets):
		pairs = []
		for onset, offset in zip(ons or [], offs or []):
			try:
				onset_f = float(onset)
				offset_f = float(offset)
			except (TypeError, ValueError):
				continue
			if math.isfinite(onset_f) and math.isfinite(offset_f) and offset_f > onset_f:
				pairs.append((onset_f, offset_f))
		out[str(stem)] = pairs
	return out


def load_roi_txt(path: Path) -> list[tuple[float, float]]:
	"""Load one text ROI file."""
	pairs = []
	try:
		lines = path.read_text(encoding="utf-8").splitlines()
	except FileNotFoundError:
		return pairs
	for line in lines:
		stripped = line.strip()
		if not stripped or stripped.startswith("#"):
			continue
		parts = stripped.split()
		if len(parts) < 2:
			continue
		try:
			onset = float(parts[0])
			offset = float(parts[1])
		except ValueError:
			continue
		if math.isfinite(onset) and math.isfinite(offset) and offset > onset:
			pairs.append((onset, offset))
	return pairs


def aggregate_latent_clip_to_events(
	latent: dict,
	rois: Sequence[tuple[float, float]],
	metadata: dict,
	entry: dict,
	clip_id: str,
	clip_stem: str,
) -> tuple[list[dict], dict]:
	"""Aggregate one clip's latent windows into ROI/protosyllable events."""
	start_times = np.asarray(latent["start_times_sec"], dtype=np.float64)
	window_length = float(np.asarray(latent["window_length_sec"]).reshape(()))
	centers = start_times + 0.5 * window_length
	mu = np.asarray(latent["mu"], dtype=np.float32)
	logvar = np.asarray(latent["logvar"], dtype=np.float32)
	energy = latent.get("energy")
	if energy is not None:
		energy = np.asarray(energy, dtype=np.float32)

	if mu.shape != logvar.shape:
		raise ValueError("mu/logvar shape mismatch.")
	if mu.shape[0] != start_times.shape[0]:
		raise ValueError("latent window count does not match start_times_sec.")
	if energy is not None and energy.shape[0] != start_times.shape[0]:
		raise ValueError("energy count does not match start_times_sec.")

	entry_dph = _coerce_dph(entry.get("dph"))
	records = []
	without_windows = 0
	for roi_index, (onset, offset) in enumerate(rois):
		left = int(np.searchsorted(centers, float(onset), side="left"))
		right = int(np.searchsorted(centers, float(offset), side="right"))
		if right <= left:
			without_windows += 1
			continue

		idx = slice(left, right)
		mu_sel = mu[idx]
		logvar_sel = logvar[idx]
		mu_mean = np.mean(mu_sel, axis=0)
		var_mean = np.mean(np.exp(logvar_sel), axis=0)
		mean_energy = None
		weight_sum = None
		mu_energy_weighted = None
		if energy is not None:
			weights = np.asarray(energy[idx], dtype=np.float64)
			weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
			mean_energy = float(np.mean(energy[idx]))
			weight_sum = float(np.sum(weights))
			if weight_sum > EPS:
				mu_energy_weighted = np.average(mu_sel, axis=0, weights=weights)

		duration = float(offset - onset)
		record = {
			"event_id": f"{clip_id}#{roi_index:05d}",
			"clip_id": str(clip_id),
			"clip_stem": str(clip_stem),
			"audio_dir_rel": entry.get("audio_dir_rel"),
			"bird_id_norm": entry.get("bird_id_norm"),
			"regime": entry.get("regime"),
			"split": entry.get("split"),
			"dph": entry_dph,
			"roi_index": int(roi_index),
			"onset_sec": float(onset),
			"offset_sec": float(offset),
			"duration_sec": duration,
			"n_windows": int(right - left),
			"start_time_first_sec": float(start_times[left]),
			"start_time_last_sec": float(start_times[right - 1]),
			"latent_dim": int(mu.shape[1]),
			"mean_energy": mean_energy,
			"energy_weight_sum": weight_sum,
			"mu": mu_mean.astype(np.float32).tolist(),
			"variance": var_mean.astype(np.float32).tolist(),
			"mu_energy_weighted": (
				None if mu_energy_weighted is None
				else np.asarray(mu_energy_weighted, dtype=np.float32).tolist()
			),
			"latent_norm": float(np.linalg.norm(mu_mean)),
			"variance_mean": float(np.mean(var_mean)),
			"latent_schema_version": metadata.get("schema_version"),
		}
		records.append(record)

	stats = {
		"roi_events_total": int(len(rois)),
		"roi_events_without_windows": int(without_windows),
		"events_written": int(len(records)),
	}
	return records, stats


def build_event_records_from_manifest(
	manifest: dict,
	latent_root: Path,
	bird_id: str = "PK249",
	dph_min: float = 33,
	dph_max: float = 90,
	split: str = "all",
	audio_root: Optional[Path] = None,
	roi_root: Optional[Path] = None,
	roi_format: str = "parquet",
	roi_parquet_name: str = "roi.parquet",
	max_events_per_dph: Optional[int] = 2000,
	seed: int = 0,
) -> EventBuildResult:
	"""Build a balanced event-level latent table from manifest-driven exports."""
	entries = _select_entries(manifest, split, bird_id, dph_min, dph_max)
	roi_format = str(roi_format).strip().lower()
	if roi_format not in {"parquet", "txt"}:
		raise ValueError("roi_format must be 'parquet' or 'txt'.")

	rng = np.random.default_rng(int(seed))
	records_by_dph: dict[float, list[dict]] = defaultdict(list)
	seen_by_dph: dict[float, int] = defaultdict(int)
	available_by_dph: dict[float, int] = defaultdict(int)
	by_dph_counts: dict[float, dict] = defaultdict(_empty_dph_counts)
	summary = {
		"bird_id": str(bird_id).strip().upper(),
		"dph_min": float(dph_min),
		"dph_max": float(dph_max),
		"split": split,
		"roi_format": roi_format,
		"entries_selected": int(len(entries)),
		"clips_seen": 0,
		"clips_missing_latent": 0,
		"clips_missing_roi": 0,
		"roi_parse_errors": 0,
		"roi_events_total": 0,
		"roi_events_without_windows": 0,
		"events_available": 0,
		"events_sampled": 0,
		"max_events_per_dph": max_events_per_dph,
		"by_dph": {},
	}

	for entry in entries:
		dph = _coerce_dph(entry.get("dph"))
		if dph is None:
			continue
		audio_dir, roi_dir = resolve_manifest_entry_paths(
			entry,
			audio_root=audio_root,
			roi_root=roi_root,
		)
		dph_counts = by_dph_counts[dph]
		wavs = _list_wavs(audio_dir)
		roi_index = None
		if roi_format == "parquet":
			roi_path = Path(roi_dir) / str(roi_parquet_name)
			try:
				roi_index = load_roi_parquet(roi_path)
			except FileNotFoundError:
				summary["clips_missing_roi"] += int(len(wavs))
				dph_counts["clips_missing_roi"] += int(len(wavs))
				continue
			except Exception:
				summary["roi_parse_errors"] += 1
				dph_counts["roi_parse_errors"] += 1
				continue

		for wav_path in wavs:
			summary["clips_seen"] += 1
			dph_counts["clips_seen"] += 1
			clip_stem = Path(wav_path).stem
			if roi_format == "parquet":
				rois = roi_index.get(clip_stem, []) if roi_index is not None else []
			else:
				rois = load_roi_txt(Path(roi_dir) / f"{clip_stem}.txt")
			if not rois:
				summary["clips_missing_roi"] += 1
				dph_counts["clips_missing_roi"] += 1
				continue

			clip_id = _clip_id_for_entry(entry, clip_stem)
			try:
				latent, metadata = load_latent_export(latent_root, clip_id)
			except FileNotFoundError:
				summary["clips_missing_latent"] += 1
				dph_counts["clips_missing_latent"] += 1
				continue

			clip_records, clip_stats = aggregate_latent_clip_to_events(
				latent=latent,
				rois=rois,
				metadata=metadata,
				entry=entry,
				clip_id=clip_id,
				clip_stem=clip_stem,
			)
			for key in ("roi_events_total", "roi_events_without_windows"):
				summary[key] += int(clip_stats[key])
				dph_counts[key] += int(clip_stats[key])
			for record in clip_records:
				available_by_dph[dph] += 1
				_reservoir_add(
					records_by_dph[dph],
					seen_by_dph,
					record,
					dph=dph,
					max_records=max_events_per_dph,
					rng=rng,
				)

	records = []
	for dph in sorted(records_by_dph):
		records.extend(
			sorted(records_by_dph[dph], key=lambda item: item["event_id"])
		)
	summary["events_available"] = int(sum(available_by_dph.values()))
	summary["events_sampled"] = int(len(records))
	for dph in sorted(set(available_by_dph) | set(records_by_dph) | set(by_dph_counts)):
		dph_counts = dict(by_dph_counts[dph])
		summary["by_dph"][str(_format_dph_key(dph))] = {
			**dph_counts,
			"events_available": int(available_by_dph.get(dph, 0)),
			"events_sampled": int(len(records_by_dph.get(dph, []))),
		}
	return EventBuildResult(records=records, summary=summary)


def _empty_dph_counts() -> dict:
	return {
		"clips_seen": 0,
		"clips_missing_latent": 0,
		"clips_missing_roi": 0,
		"roi_parse_errors": 0,
		"roi_events_total": 0,
		"roi_events_without_windows": 0,
	}


def _format_dph_key(dph: float) -> Any:
	return int(dph) if float(dph).is_integer() else float(dph)


def _reservoir_add(
	bucket: list[dict],
	seen_by_dph: dict[float, int],
	record: dict,
	dph: float,
	max_records: Optional[int],
	rng: np.random.Generator,
) -> None:
	seen = int(seen_by_dph[dph])
	seen_by_dph[dph] = seen + 1
	if max_records is None:
		bucket.append(record)
		return
	max_records = int(max_records)
	if max_records <= 0:
		return
	if len(bucket) < max_records:
		bucket.append(record)
		return
	j = int(rng.integers(0, seen + 1))
	if j < max_records:
		bucket[j] = record


def latent_matrix(records: Sequence[dict], use_energy_weighted: bool = False) -> np.ndarray:
	"""Return an event x latent-dimension matrix from event records."""
	rows = []
	for record in records:
		if use_energy_weighted and record.get("mu_energy_weighted") is not None:
			rows.append(record["mu_energy_weighted"])
		else:
			rows.append(record["mu"])
	if not rows:
		raise ValueError("No event records available.")
	return np.asarray(rows, dtype=np.float32)


def standardize_latents(x: np.ndarray) -> tuple[np.ndarray, dict]:
	"""Standardize latent coordinates for geometry/clustering."""
	scaler = StandardScaler()
	x_std = scaler.fit_transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
	return x_std, {
		"mean": scaler.mean_.astype(float).tolist(),
		"scale": scaler.scale_.astype(float).tolist(),
	}


def fit_adult_clusters(
	records: Sequence[dict],
	x_std: np.ndarray,
	late_dph_min: float = 80,
	min_k: int = DEFAULT_CLUSTER_K_RANGE[0],
	max_k: int = DEFAULT_CLUSTER_K_RANGE[1],
	seed: int = 0,
) -> ClusterResult:
	"""Fit adult-like clusters on late-age events and assign all events."""
	dph = np.asarray([float(record["dph"]) for record in records], dtype=np.float64)
	late_idx = np.flatnonzero(dph >= float(late_dph_min))
	if late_idx.size == 0:
		raise ValueError("No late-age events available for adult-like clustering.")

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
				best_k = k
				best_model = model

	if best_model is None:
		best_k = 1
		centers = np.mean(x_late, axis=0, keepdims=True)
	else:
		centers = best_model.cluster_centers_.astype(np.float32)

	distances = np.linalg.norm(x_std[:, None, :] - centers[None, :, :], axis=2)
	labels = np.argmin(distances, axis=1).astype(np.int32)
	if centers.shape[0] == 1:
		confidence = np.ones(x_std.shape[0], dtype=np.float32)
		entropy = np.zeros(x_std.shape[0], dtype=np.float32)
	else:
		scale = float(np.median(np.min(distances, axis=1)))
		if not math.isfinite(scale) or scale <= EPS:
			scale = 1.0
		logits = -distances / scale
		logits = logits - np.max(logits, axis=1, keepdims=True)
		probs = np.exp(logits)
		probs = probs / np.sum(probs, axis=1, keepdims=True)
		confidence = np.max(probs, axis=1).astype(np.float32)
		entropy_raw = -np.sum(probs * np.log(probs + EPS), axis=1)
		entropy = (entropy_raw / math.log(centers.shape[0])).astype(np.float32)

	return ClusterResult(
		best_k=int(best_k),
		silhouette_scores=scores,
		centers=np.asarray(centers, dtype=np.float32),
		labels=labels,
		distances=np.min(distances, axis=1).astype(np.float32),
		confidence=confidence,
		entropy=entropy,
		late_event_count=int(late_idx.size),
	)


def apply_cluster_assignments(records: Sequence[dict], result: ClusterResult) -> None:
	"""Mutate records with adult-like cluster assignment fields."""
	for i, record in enumerate(records):
		record["adult_cluster"] = int(result.labels[i])
		record["adult_cluster_distance"] = float(result.distances[i])
		record["adult_cluster_confidence"] = float(result.confidence[i])
		record["adult_cluster_entropy"] = float(result.entropy[i])


def select_balanced_indices(
	records: Sequence[dict],
	max_records: Optional[int],
	seed: int = 0,
) -> np.ndarray:
	"""Select a dph-balanced subset of record indices."""
	n = len(records)
	if max_records is None or int(max_records) >= n:
		return np.arange(n, dtype=np.int64)
	max_records = int(max_records)
	if max_records <= 0:
		raise ValueError("max_records must be positive.")

	by_dph: dict[float, list[int]] = defaultdict(list)
	for idx, record in enumerate(records):
		by_dph[float(record["dph"])].append(idx)

	rng = np.random.default_rng(int(seed))
	dph_values = sorted(by_dph)
	base = max(1, max_records // max(1, len(dph_values)))
	selected = []
	for dph in dph_values:
		indices = np.asarray(by_dph[dph], dtype=np.int64)
		keep = min(base, len(indices))
		selected.extend(rng.choice(indices, size=keep, replace=False).tolist())
	remaining = max_records - len(selected)
	if remaining > 0:
		pool = np.setdiff1d(np.arange(n, dtype=np.int64), np.asarray(selected, dtype=np.int64))
		if pool.size:
			selected.extend(
				rng.choice(pool, size=min(remaining, pool.size), replace=False).tolist()
			)
	selected = sorted(set(int(i) for i in selected))
	return np.asarray(selected[:max_records], dtype=np.int64)


def poincare_distance(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
	"""Compute Poincare-ball distances for curvature -1."""
	a = np.asarray(points_a, dtype=np.float64)
	b = np.asarray(points_b, dtype=np.float64)
	diff_sq = np.sum((a - b) ** 2, axis=-1)
	a_norm_sq = np.sum(a * a, axis=-1)
	b_norm_sq = np.sum(b * b, axis=-1)
	denom = np.maximum((1.0 - a_norm_sq) * (1.0 - b_norm_sq), EPS)
	arg = 1.0 + 2.0 * diff_sq / denom
	return np.arccosh(np.maximum(arg, 1.0))


def poincare_radius(points: np.ndarray) -> np.ndarray:
	"""Return hyperbolic distance from the origin."""
	norm = np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1)
	norm = np.clip(norm, 0.0, 1.0 - EPS)
	return 2.0 * np.arctanh(norm)


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Mobius addition in the unit Poincare ball, curvature -1."""
	x = np.asarray(x, dtype=np.float64)
	y = np.asarray(y, dtype=np.float64)
	x2 = np.sum(x * x, axis=-1, keepdims=True)
	y2 = np.sum(y * y, axis=-1, keepdims=True)
	xy = np.sum(x * y, axis=-1, keepdims=True)
	num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
	denom = 1.0 + 2.0 * xy + x2 * y2
	return num / np.maximum(denom, EPS)


def recenter_poincare(points: np.ndarray, center: np.ndarray) -> np.ndarray:
	"""Translate ``center`` to the origin with Mobius addition."""
	recentered = mobius_add(-np.asarray(center, dtype=np.float64), points)
	norm = np.linalg.norm(recentered, axis=1)
	mask = norm >= 1.0
	if np.any(mask):
		recentered[mask] = recentered[mask] / norm[mask, None] * (1.0 - 1e-6)
	return recentered.astype(np.float32)


def fit_poincare_embedding(
	x_std: np.ndarray,
	seed: int = 0,
	knn: int = 10,
	epochs: int = 300,
	lr: float = 0.01,
	max_edges: int = 200_000,
) -> EmbeddingResult:
	"""Fit a two-dimensional Poincare embedding from kNN edge distances."""
	try:
		import torch
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError("PyTorch is required for Poincare embedding optimization.") from exc

	x_std = np.asarray(x_std, dtype=np.float32)
	n = x_std.shape[0]
	if n < 2:
		raise ValueError("At least two events are required for Poincare embedding.")

	n_neighbors = min(max(2, int(knn) + 1), n)
	nn = NearestNeighbors(n_neighbors=n_neighbors)
	nn.fit(x_std)
	dists, neighbors = nn.kneighbors(x_std)
	src = []
	dst = []
	targets = []
	seen_edges = set()
	for i in range(n):
		for j, dist in zip(neighbors[i, 1:], dists[i, 1:]):
			a, b = sorted((int(i), int(j)))
			if a == b or (a, b) in seen_edges:
				continue
			seen_edges.add((a, b))
			src.append(a)
			dst.append(b)
			targets.append(float(dist))
	if not src:
		raise ValueError("No kNN edges were available for embedding.")

	rng = np.random.default_rng(int(seed))
	if max_edges is not None and len(src) > int(max_edges):
		keep = rng.choice(len(src), size=int(max_edges), replace=False)
		src = [src[int(i)] for i in keep]
		dst = [dst[int(i)] for i in keep]
		targets = [targets[int(i)] for i in keep]

	target_np = np.asarray(targets, dtype=np.float32)
	positive = target_np[target_np > EPS]
	target_scale = float(np.median(positive)) if positive.size else 1.0
	if not math.isfinite(target_scale) or target_scale <= EPS:
		target_scale = 1.0
	target_np = np.clip(target_np / target_scale, 0.0, 8.0)

	init = _initial_pca_coordinates(x_std, seed=seed)
	points_param = torch.tensor(init, dtype=torch.float32, requires_grad=True)
	src_t = torch.tensor(src, dtype=torch.long)
	dst_t = torch.tensor(dst, dtype=torch.long)
	target_t = torch.tensor(target_np, dtype=torch.float32)
	opt = torch.optim.Adam([points_param], lr=float(lr))
	loss_history: list[float] = []
	best_points = _project_poincare_torch(points_param).detach().clone()
	best_loss = float("inf")
	for _ in range(int(epochs)):
		opt.zero_grad()
		points = _project_poincare_torch(points_param)
		pred = _poincare_distance_torch(points[src_t], points[dst_t])
		loss = torch.mean((pred - target_t) ** 2)
		loss = loss + 1e-4 * torch.mean(torch.sum(points * points, dim=1))
		if not bool(torch.isfinite(loss).detach().cpu()):
			break
		loss_value = float(loss.detach().cpu())
		loss_history.append(loss_value)
		if loss_value < best_loss:
			best_loss = loss_value
			best_points = points.detach().clone()
		loss.backward()
		if not _all_finite_gradients([points_param]):
			break
		torch.nn.utils.clip_grad_norm_([points_param], max_norm=10.0)
		opt.step()
		with torch.no_grad():
			points_param.nan_to_num_(nan=0.0, posinf=0.95, neginf=-0.95)
			points_param.copy_(_project_poincare_torch(points_param))

	if not loss_history:
		raise ValueError("Poincare embedding optimization produced no finite losses.")

	points = best_points.detach().cpu().numpy().astype(np.float32)
	if not np.isfinite(points).all():
		raise ValueError("Poincare embedding produced non-finite coordinates.")
	return EmbeddingResult(
		points=points,
		loss_history=loss_history,
		edge_count=int(len(src)),
		target_scale=float(target_scale),
	)


def _initial_pca_coordinates(x_std: np.ndarray, seed: int) -> np.ndarray:
	if x_std.shape[0] < 3 or x_std.shape[1] < 2:
		coords = np.zeros((x_std.shape[0], 2), dtype=np.float32)
		if x_std.shape[1] >= 1:
			coords[:, 0] = x_std[:, 0]
	else:
		coords = PCA(n_components=2, random_state=int(seed)).fit_transform(x_std)
	coords = np.asarray(coords, dtype=np.float32)
	norm = np.linalg.norm(coords, axis=1)
	max_norm = float(np.max(norm)) if norm.size else 0.0
	if max_norm > EPS:
		coords = coords / max_norm * 0.05
	return coords.astype(np.float32)


def _raw_to_poincare_torch(raw):
	import torch

	norm = torch.linalg.norm(raw, dim=1, keepdim=True)
	direction = raw / torch.clamp(norm, min=EPS)
	radius = 0.95 * torch.tanh(norm)
	return direction * radius


def _project_poincare_torch(points, max_norm: float = 0.95):
	import torch

	points = torch.nan_to_num(points, nan=0.0, posinf=max_norm, neginf=-max_norm)
	norm = torch.linalg.norm(points, dim=1, keepdim=True)
	scale = torch.clamp(float(max_norm) / torch.clamp(norm, min=EPS), max=1.0)
	return points * scale


def _all_finite_gradients(parameters) -> bool:
	import torch

	for parameter in parameters:
		if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
			return False
	return True


def _poincare_distance_torch(a, b):
	import torch

	torch_eps = 1e-5
	diff_sq = torch.sum((a - b) ** 2, dim=1)
	a_norm_sq = torch.sum(a * a, dim=1)
	b_norm_sq = torch.sum(b * b, dim=1)
	denom = torch.clamp((1.0 - a_norm_sq) * (1.0 - b_norm_sq), min=EPS)
	arg = 1.0 + 2.0 * diff_sq / denom
	return torch.acosh(torch.clamp(arg, min=1.0 + torch_eps))


def attach_embedding_coordinates(
	records: Sequence[dict],
	points: np.ndarray,
	early_dph_max: float = 45,
) -> np.ndarray:
	"""Recenter an embedding on early events and attach radius coordinates."""
	dph = np.asarray([float(record["dph"]) for record in records], dtype=np.float64)
	early = dph <= float(early_dph_max)
	if not np.any(early):
		raise ValueError("No early events available for Poincare recentering.")
	center = np.mean(points[early], axis=0)
	center_norm = float(np.linalg.norm(center))
	if center_norm >= 1.0:
		center = center / center_norm * (1.0 - 1e-6)
	recentered = recenter_poincare(points, center)
	radii = poincare_radius(recentered)
	for record, point, radius in zip(records, recentered, radii):
		record["poincare_x"] = float(point[0])
		record["poincare_y"] = float(point[1])
		record["poincare_radius"] = float(radius)
	return recentered


def attach_control_coordinates(
	records: Sequence[dict],
	x_raw: np.ndarray,
	x_std: np.ndarray,
	early_dph_max: float = 45,
	seed: int = 0,
	compute_umap: bool = True,
) -> dict:
	"""Attach Euclidean/PCA/optional UMAP radius controls to records."""
	dph = np.asarray([float(record["dph"]) for record in records], dtype=np.float64)
	early = dph <= float(early_dph_max)
	if not np.any(early):
		early = np.ones(len(records), dtype=bool)

	raw_norm = np.linalg.norm(x_raw, axis=1)
	pca_coords = _initial_pca_coordinates(x_std, seed=seed)
	pca_center = np.mean(pca_coords[early], axis=0)
	pca_radius = np.linalg.norm(pca_coords - pca_center[None, :], axis=1)
	umap_status = {"available": False, "error": None}
	umap_coords = None
	umap_radius = None
	if compute_umap and len(records) >= 4:
		try:
			import umap  # type: ignore

			n_neighbors = min(15, max(2, len(records) - 1))
			reducer = umap.UMAP(
				n_components=2,
				n_neighbors=n_neighbors,
				min_dist=0.1,
				random_state=int(seed),
			)
			umap_coords = reducer.fit_transform(x_std)
			umap_center = np.mean(umap_coords[early], axis=0)
			umap_radius = np.linalg.norm(umap_coords - umap_center[None, :], axis=1)
			umap_status["available"] = True
		except Exception as exc:  # pragma: no cover - depends on optional numba/umap env
			umap_status["error"] = str(exc)

	for i, record in enumerate(records):
		record["euclidean_latent_norm"] = float(raw_norm[i])
		record["pca_x"] = float(pca_coords[i, 0])
		record["pca_y"] = float(pca_coords[i, 1])
		record["pca_radius"] = float(pca_radius[i])
		if umap_coords is not None and umap_radius is not None:
			record["umap_x"] = float(umap_coords[i, 0])
			record["umap_y"] = float(umap_coords[i, 1])
			record["umap_radius"] = float(umap_radius[i])
		else:
			record["umap_x"] = None
			record["umap_y"] = None
			record["umap_radius"] = None
	return {"umap": umap_status}


def radius_age_metrics(
	records: Sequence[dict],
	radius_key: str = "poincare_radius",
	bootstrap_iterations: int = 1000,
	seed: int = 0,
) -> dict:
	"""Compute per-dph radius summaries and Spearman/block-bootstrap stats."""
	grouped: dict[float, list[float]] = defaultdict(list)
	for record in records:
		radius = record.get(radius_key)
		dph = _coerce_dph(record.get("dph"))
		if dph is None or radius is None:
			continue
		radius = float(radius)
		if math.isfinite(radius):
			grouped[dph].append(radius)
	if len(grouped) < 2:
		raise ValueError("At least two dph groups are required for radius metrics.")

	dph_values = np.asarray(sorted(grouped), dtype=np.float64)
	medians = np.asarray([np.median(grouped[dph]) for dph in dph_values], dtype=np.float64)
	p25 = np.asarray([np.percentile(grouped[dph], 25) for dph in dph_values], dtype=np.float64)
	p75 = np.asarray([np.percentile(grouped[dph], 75) for dph in dph_values], dtype=np.float64)
	n = np.asarray([len(grouped[dph]) for dph in dph_values], dtype=np.int64)
	rho, p_value = spearmanr(dph_values, medians)
	boot = _bootstrap_spearman(dph_values, medians, bootstrap_iterations, seed)
	return {
		"radius_key": radius_key,
		"spearman_rho": float(rho),
		"spearman_p_value": float(p_value),
		"bootstrap_iterations": int(bootstrap_iterations),
		"bootstrap_rho_mean": float(np.mean(boot)) if boot.size else None,
		"bootstrap_rho_ci95": (
			[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
			if boot.size else None
		),
		"per_dph": [
			{
				"dph": float(dph),
				"n": int(count),
				"median": float(median),
				"p25": float(lo),
				"p75": float(hi),
			}
			for dph, count, median, lo, hi in zip(dph_values, n, medians, p25, p75)
		],
	}


def _bootstrap_spearman(
	dph_values: np.ndarray,
	medians: np.ndarray,
	iterations: int,
	seed: int,
) -> np.ndarray:
	if iterations <= 0 or dph_values.size < 2:
		return np.zeros((0,), dtype=np.float64)
	rng = np.random.default_rng(int(seed))
	out = []
	for _ in range(int(iterations)):
		idx = rng.integers(0, dph_values.size, size=dph_values.size)
		if len(set(idx.tolist())) < 2:
			continue
		rho, _ = spearmanr(dph_values[idx], medians[idx])
		if math.isfinite(float(rho)):
			out.append(float(rho))
	return np.asarray(out, dtype=np.float64)


def grouped_median_metric(records: Sequence[dict], key: str) -> list[dict]:
	"""Summarize a scalar record field by dph."""
	grouped: dict[float, list[float]] = defaultdict(list)
	for record in records:
		value = record.get(key)
		dph = _coerce_dph(record.get("dph"))
		if dph is None or value is None:
			continue
		value = float(value)
		if math.isfinite(value):
			grouped[dph].append(value)
	return [
		{"dph": float(dph), "n": len(values), "median": float(np.median(values))}
		for dph, values in sorted(grouped.items())
	]


def early_late_metric_contrast(
	records: Sequence[dict],
	key: str,
	early_dph_max: float,
	late_dph_min: float,
) -> dict:
	"""Summarize early-vs-late medians for a scalar embedding record field."""
	early = []
	late = []
	for record in records:
		value = record.get(key)
		dph = _coerce_dph(record.get("dph"))
		if dph is None or value is None:
			continue
		value = float(value)
		if not math.isfinite(value):
			continue
		if dph <= float(early_dph_max):
			early.append(value)
		if dph >= float(late_dph_min):
			late.append(value)
	early_arr = np.asarray(early, dtype=np.float64)
	late_arr = np.asarray(late, dtype=np.float64)
	early_median = float(np.median(early_arr)) if early_arr.size else None
	late_median = float(np.median(late_arr)) if late_arr.size else None
	delta = (
		float(late_median - early_median)
		if early_median is not None and late_median is not None
		else None
	)
	return {
		"key": key,
		"early_dph_max": float(early_dph_max),
		"late_dph_min": float(late_dph_min),
		"early_n": int(early_arr.size),
		"late_n": int(late_arr.size),
		"early_median": early_median,
		"late_median": late_median,
		"late_minus_early": delta,
	}


def cluster_payload(result: ClusterResult, late_dph_min: float) -> dict:
	"""Return a serializable cluster-fit payload."""
	return {
		"late_dph_min": float(late_dph_min),
		"late_event_count": int(result.late_event_count),
		"best_k": int(result.best_k),
		"silhouette_scores": result.silhouette_scores,
		"centers": result.centers.astype(float).tolist(),
	}


def write_records_parquet(records: Sequence[dict], path: Path) -> None:
	"""Write event/embedding records to parquet."""
	try:
		import pyarrow as pa  # type: ignore
		import pyarrow.parquet as pq  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError("Writing parquet analysis artifacts requires pyarrow.") from exc

	if not records:
		raise ValueError("Cannot write an empty record table.")
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
		"adult_cluster",
		"adult_cluster_distance",
		"adult_cluster_confidence",
		"adult_cluster_entropy",
		"poincare_x",
		"poincare_y",
		"poincare_radius",
		"euclidean_latent_norm",
		"pca_x",
		"pca_y",
		"pca_radius",
		"umap_x",
		"umap_y",
		"umap_radius",
	]
	list_fields = ["mu", "variance", "mu_energy_weighted"]
	arrays = []
	names = []
	for field in scalar_fields:
		values = [record.get(field) for record in records]
		arrays.append(pa.array(values))
		names.append(field)
	for field in list_fields:
		values = [record.get(field) for record in records]
		arrays.append(pa.array(values, type=pa.list_(pa.float32())))
		names.append(field)
	table = pa.Table.from_arrays(arrays, names=names)
	pq.write_table(table, path.as_posix())


def make_figures(records: Sequence[dict], metrics: dict, out_dir: Path) -> dict[str, str]:
	"""Create report figures and return paths keyed by figure name."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_dir.mkdir(parents=True, exist_ok=True)
	figures: dict[str, str] = {}
	dph = np.asarray([float(record["dph"]) for record in records], dtype=np.float64)
	fig, ax = plt.subplots(figsize=(6, 6))
	x = np.asarray([float(record["poincare_x"]) for record in records])
	y = np.asarray([float(record["poincare_y"]) for record in records])
	scatter = ax.scatter(x, y, c=dph, s=10, alpha=0.65, cmap="viridis", linewidths=0)
	ax.add_patch(plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=1.0))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(-1.02, 1.02)
	ax.set_ylim(-1.02, 1.02)
	ax.set_xlabel("Poincare x")
	ax.set_ylabel("Poincare y")
	ax.set_title("Hyperbolic embedding by age")
	fig.colorbar(scatter, ax=ax, label="dph")
	_save_figure(fig, out_dir / "poincare_disk")
	figures["poincare_disk"] = (out_dir / "poincare_disk.png").as_posix()

	per_dph = metrics["radius_age"]["per_dph"]
	dph_line = np.asarray([row["dph"] for row in per_dph], dtype=np.float64)
	median = np.asarray([row["median"] for row in per_dph], dtype=np.float64)
	p25 = np.asarray([row["p25"] for row in per_dph], dtype=np.float64)
	p75 = np.asarray([row["p75"] for row in per_dph], dtype=np.float64)
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.fill_between(dph_line, p25, p75, color="#4C78A8", alpha=0.2, label="IQR")
	ax.plot(dph_line, median, color="#4C78A8", marker="o", linewidth=1.5, label="median")
	ax.set_xlabel("dph")
	ax.set_ylabel("hyperbolic radius")
	ax.set_title("Hyperbolic radius by developmental age")
	ax.legend(loc="best")
	_save_figure(fig, out_dir / "radius_by_dph")
	figures["radius_by_dph"] = (out_dir / "radius_by_dph.png").as_posix()

	fig, ax = plt.subplots(figsize=(10, 4.5))
	_plot_cluster_composition(records, ax)
	ax.set_title("Adult-like branch composition by age")
	_save_figure(fig, out_dir / "adult_branch_composition")
	figures["adult_branch_composition"] = (
		out_dir / "adult_branch_composition.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	for key, label, color in [
		("adult_cluster_confidence", "confidence", "#54A24B"),
		("adult_cluster_entropy", "entropy", "#E45756"),
	]:
		grouped = grouped_median_metric(records, key)
		if grouped:
			ax.plot(
				[row["dph"] for row in grouped],
				[row["median"] for row in grouped],
				marker="o",
				linewidth=1.5,
				label=label,
				color=color,
			)
	ax.set_xlabel("dph")
	ax.set_ylabel("median value")
	ax.set_title("Adult-like branch confidence and entropy")
	ax.legend(loc="best")
	_save_figure(fig, out_dir / "adult_confidence_entropy")
	figures["adult_confidence_entropy"] = (
		out_dir / "adult_confidence_entropy.png"
	).as_posix()

	fig, ax = plt.subplots(figsize=(8, 4.5))
	for key, label in [
		("euclidean_latent_norm", "Euclidean latent norm"),
		("pca_radius", "PCA radius"),
		("umap_radius", "UMAP radius"),
	]:
		grouped = grouped_median_metric(records, key)
		if grouped:
			ax.plot(
				[row["dph"] for row in grouped],
				[row["median"] for row in grouped],
				marker="o",
				linewidth=1.2,
				label=label,
			)
	ax.set_xlabel("dph")
	ax.set_ylabel("median control radius")
	ax.set_title("Euclidean and projection controls")
	ax.legend(loc="best")
	_save_figure(fig, out_dir / "radius_controls")
	figures["radius_controls"] = (out_dir / "radius_controls.png").as_posix()
	return figures


def _save_figure(fig, prefix: Path) -> None:
	fig.tight_layout()
	fig.savefig(Path(f"{prefix.as_posix()}.png"), dpi=180)
	fig.savefig(Path(f"{prefix.as_posix()}.pdf"))
	import matplotlib.pyplot as plt

	plt.close(fig)


def _plot_cluster_composition(records: Sequence[dict], ax) -> None:
	grouped: dict[float, list[int]] = defaultdict(list)
	for record in records:
		cluster = record.get("adult_cluster")
		if cluster is None:
			continue
		grouped[float(record["dph"])].append(int(cluster))
	if not grouped:
		return
	dph_values = sorted(grouped)
	clusters = sorted({cluster for values in grouped.values() for cluster in values})
	bottom = np.zeros(len(dph_values), dtype=np.float64)
	for cluster in clusters:
		values = []
		for dph in dph_values:
			arr = np.asarray(grouped[dph], dtype=np.int32)
			values.append(float(np.mean(arr == cluster)) if arr.size else 0.0)
		ax.bar(dph_values, values, bottom=bottom, width=0.8, label=f"C{cluster}")
		bottom += np.asarray(values)
	ax.set_xlabel("dph")
	ax.set_ylabel("fraction of events")
	ax.set_ylim(0.0, 1.0)
	ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def write_report(
	report_path: Path,
	metrics: dict,
	figures: dict[str, str],
	artifacts: dict[str, str],
) -> None:
	"""Write a figure-ready markdown report."""
	report_path.parent.mkdir(parents=True, exist_ok=True)
	radius = metrics["radius_age"]
	ci = radius.get("bootstrap_rho_ci95")
	ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
	confidence_age = metrics.get("adult_confidence_age", {})
	entropy_age = metrics.get("adult_entropy_age", {})
	conf_ci = confidence_age.get("bootstrap_rho_ci95")
	ent_ci = entropy_age.get("bootstrap_rho_ci95")
	conf_ci_text = "n/a" if conf_ci is None else f"[{conf_ci[0]:.3f}, {conf_ci[1]:.3f}]"
	ent_ci_text = "n/a" if ent_ci is None else f"[{ent_ci[0]:.3f}, {ent_ci[1]:.3f}]"
	sensitivity = metrics.get("sensitivity", {})
	sens_radius = sensitivity.get("radius_age", {})
	sens_text = "not run"
	if sens_radius:
		sens_text = (
			f"rho={sens_radius.get('spearman_rho'):.3f}, "
			f"CI={sens_radius.get('bootstrap_rho_ci95')}"
		)

	def rel(path: str) -> str:
		return os.path.relpath(path, report_path.parent.as_posix())

	lines = [
		"# Hyperbolic Developmental AVA Report",
		"",
		"## Summary",
		"",
		f"- Events sampled: {metrics['event_summary']['events_sampled']} "
		f"of {metrics['event_summary']['events_available']} available ROI events.",
		f"- Primary radius-age Spearman rho: {radius['spearman_rho']:.3f} "
		f"(p={radius['spearman_p_value']:.3g}, bootstrap CI {ci_text}).",
		f"- Adult-like clusters: K={metrics['adult_clusters']['best_k']} "
		f"from {metrics['adult_clusters']['late_event_count']} late-age events.",
		f"- Adult-cluster confidence-age Spearman rho: "
		f"{confidence_age.get('spearman_rho', float('nan')):.3f} "
		f"(bootstrap CI {conf_ci_text}); entropy-age rho: "
		f"{entropy_age.get('spearman_rho', float('nan')):.3f} "
		f"(bootstrap CI {ent_ci_text}).",
		_early_late_summary(metrics),
		f"- Sensitivity run: {sens_text}.",
		"",
		"## Figures",
		"",
		f"![Poincare disk]({rel(figures['poincare_disk'])})",
		"",
		f"![Radius by dph]({rel(figures['radius_by_dph'])})",
		"",
		f"![Adult branch composition]({rel(figures['adult_branch_composition'])})",
		"",
		f"![Adult confidence and entropy]({rel(figures['adult_confidence_entropy'])})",
		"",
		f"![Radius controls]({rel(figures['radius_controls'])})",
		"",
		"## Interpretation",
		"",
		_interpret_radius(radius),
		"",
		_interpret_branch_confidence(metrics),
		"",
		"## Skip And Coverage Summary",
		"",
		"```json",
		json.dumps(_to_jsonable(metrics["event_summary"]), indent=2, sort_keys=True),
		"```",
		"",
		"## Artifacts",
		"",
	]
	for name, path in artifacts.items():
		lines.append(f"- `{name}`: `{rel(path)}`")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional(value: Any, digits: int = 4) -> str:
	if value is None:
		return "n/a"
	try:
		value = float(value)
	except (TypeError, ValueError):
		return "n/a"
	return f"{value:.{digits}f}" if math.isfinite(value) else "n/a"


def _early_late_summary(metrics: dict) -> str:
	contrasts = metrics.get("early_late_contrasts", {})
	radius = contrasts.get("poincare_radius", {})
	confidence = contrasts.get("adult_cluster_confidence", {})
	entropy = contrasts.get("adult_cluster_entropy", {})
	return (
		"- Early/late medians: radius "
		f"{_fmt_optional(radius.get('early_median'))} -> "
		f"{_fmt_optional(radius.get('late_median'))}; confidence "
		f"{_fmt_optional(confidence.get('early_median'))} -> "
		f"{_fmt_optional(confidence.get('late_median'))}; entropy "
		f"{_fmt_optional(entropy.get('early_median'))} -> "
		f"{_fmt_optional(entropy.get('late_median'))}."
	)


def _interpret_radius(radius_metrics: dict) -> str:
	rho = float(radius_metrics["spearman_rho"])
	ci = radius_metrics.get("bootstrap_rho_ci95")
	if ci and ci[0] > 0:
		return (
			"The primary radius-age test supports the developmental hypothesis: "
			"median hyperbolic radius increases monotonically across dph groups."
		)
	if rho > 0:
		return (
			"The primary radius-age test is directionally consistent with the "
			"developmental hypothesis, but the bootstrap interval does not clearly "
			"exclude weak or unstable effects."
		)
	return (
		"The primary radius-age test does not support the current hyperbolic "
		"developmental hypothesis in this run."
	)


def _interpret_branch_confidence(metrics: dict) -> str:
	contrasts = metrics.get("early_late_contrasts", {})
	radius = contrasts.get("poincare_radius", {})
	confidence = contrasts.get("adult_cluster_confidence", {})
	entropy = contrasts.get("adult_cluster_entropy", {})
	radius_delta = radius.get("late_minus_early")
	conf_delta = confidence.get("late_minus_early")
	entropy_delta = entropy.get("late_minus_early")

	branch_confident = (
		conf_delta is not None and conf_delta > 0
		and entropy_delta is not None and entropy_delta < 0
	)
	radius_distal = radius_delta is not None and radius_delta > 0
	if branch_confident and radius_distal:
		return (
			"Late events occupy more confident, lower-entropy adult-like branches "
			"and are more distal in the recentered Poincare embedding."
		)
	if branch_confident:
		return (
			"Late events are more confidently assigned to adult-like branches and "
			"have lower cluster entropy, but they are not more distal by the primary "
			"hyperbolic radius coordinate."
		)
	return (
		"Late events do not show the paired increase in adult-cluster confidence "
		"and decrease in cluster entropy expected for branch-like differentiation."
	)


def run_hyperbolic_development_analysis(
	manifest_path: Path,
	latent_root: Path,
	artifact_dir: Path,
	report_path: Path,
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
	embedding_max_events: int = 6000,
	knn: int = 10,
	embedding_epochs: int = 300,
	embedding_lr: float = 0.05,
	cluster_min_k: int = 4,
	cluster_max_k: int = 12,
	bootstrap_iterations: int = 1000,
	seed: int = 0,
	use_energy_weighted: bool = False,
	compute_umap: bool = True,
	run_sensitivity: bool = True,
) -> dict:
	"""Run the full post-hoc hyperbolic developmental analysis."""
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
	if not build.records:
		raise ValueError("No ROI event latents were available after filtering.")

	x_raw = latent_matrix(build.records, use_energy_weighted=use_energy_weighted)
	x_std, standardization = standardize_latents(x_raw)
	clusters = fit_adult_clusters(
		build.records,
		x_std,
		late_dph_min=late_dph_min,
		min_k=cluster_min_k,
		max_k=cluster_max_k,
		seed=seed,
	)
	apply_cluster_assignments(build.records, clusters)

	embedding_idx = select_balanced_indices(
		build.records,
		max_records=embedding_max_events,
		seed=seed,
	)
	embedding_records = [dict(build.records[int(i)]) for i in embedding_idx]
	x_embed_raw = x_raw[embedding_idx]
	x_embed_std = x_std[embedding_idx]
	embedding = fit_poincare_embedding(
		x_embed_std,
		seed=seed,
		knn=knn,
		epochs=embedding_epochs,
		lr=embedding_lr,
	)
	points = attach_embedding_coordinates(
		embedding_records,
		embedding.points,
		early_dph_max=early_dph_max,
	)
	control_status = attach_control_coordinates(
		embedding_records,
		x_embed_raw,
		x_embed_std,
		early_dph_max=early_dph_max,
		seed=seed,
		compute_umap=compute_umap,
	)

	radius_metrics = radius_age_metrics(
		embedding_records,
		radius_key="poincare_radius",
		bootstrap_iterations=bootstrap_iterations,
		seed=seed,
	)
	metrics = {
		"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
		"inputs": {
			"manifest_path": manifest_path.as_posix(),
			"latent_root": latent_root.as_posix(),
			"bird_id": bird_id,
			"dph_min": float(dph_min),
			"dph_max": float(dph_max),
			"early_dph_max": float(early_dph_max),
			"late_dph_min": float(late_dph_min),
			"split": split,
			"use_energy_weighted": bool(use_energy_weighted),
		},
		"event_summary": build.summary,
		"standardization": standardization,
		"adult_clusters": cluster_payload(clusters, late_dph_min),
		"embedding": {
			"events_embedded": int(len(embedding_records)),
			"events_available_for_embedding": int(len(build.records)),
			"knn": int(knn),
			"epochs": int(embedding_epochs),
			"epochs_completed": int(len(embedding.loss_history)),
			"lr": float(embedding_lr),
			"edge_count": int(embedding.edge_count),
			"target_scale": float(embedding.target_scale),
			"loss_initial": float(embedding.loss_history[0]),
			"loss_final": float(embedding.loss_history[-1]),
			"loss_best": float(min(embedding.loss_history)),
		},
		"controls": control_status,
		"radius_age": radius_metrics,
		"adult_confidence_by_dph": grouped_median_metric(
			embedding_records,
			"adult_cluster_confidence",
		),
		"adult_entropy_by_dph": grouped_median_metric(
			embedding_records,
			"adult_cluster_entropy",
		),
		"adult_confidence_age": radius_age_metrics(
			embedding_records,
			radius_key="adult_cluster_confidence",
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
		),
		"adult_entropy_age": radius_age_metrics(
			embedding_records,
			radius_key="adult_cluster_entropy",
			bootstrap_iterations=bootstrap_iterations,
			seed=seed,
		),
		"early_late_contrasts": {
			"poincare_radius": early_late_metric_contrast(
				embedding_records,
				"poincare_radius",
				early_dph_max,
				late_dph_min,
			),
			"adult_cluster_confidence": early_late_metric_contrast(
				embedding_records,
				"adult_cluster_confidence",
				early_dph_max,
				late_dph_min,
			),
			"adult_cluster_entropy": early_late_metric_contrast(
				embedding_records,
				"adult_cluster_entropy",
				early_dph_max,
				late_dph_min,
			),
		},
		"control_radius_by_dph": {
			"euclidean_latent_norm": grouped_median_metric(
				embedding_records,
				"euclidean_latent_norm",
			),
			"pca_radius": grouped_median_metric(embedding_records, "pca_radius"),
			"umap_radius": grouped_median_metric(embedding_records, "umap_radius"),
		},
	}

	if run_sensitivity:
		sensitivity_epochs = max(5, int(math.ceil(int(embedding_epochs) / 3.0)))
		sensitivity = fit_poincare_embedding(
			x_embed_std,
			seed=int(seed) + 1,
			knn=knn,
			epochs=sensitivity_epochs,
			lr=embedding_lr,
		)
		sensitivity_records = [dict(record) for record in embedding_records]
		attach_embedding_coordinates(
			sensitivity_records,
			sensitivity.points,
			early_dph_max=early_dph_max,
		)
		metrics["sensitivity"] = {
			"seed": int(seed) + 1,
			"embedding": {
				"epochs": int(sensitivity_epochs),
				"epochs_completed": int(len(sensitivity.loss_history)),
				"lr": float(embedding_lr),
				"edge_count": int(sensitivity.edge_count),
				"loss_initial": float(sensitivity.loss_history[0]),
				"loss_final": float(sensitivity.loss_history[-1]),
				"loss_best": float(min(sensitivity.loss_history)),
			},
			"radius_age": radius_age_metrics(
				sensitivity_records,
				radius_key="poincare_radius",
				bootstrap_iterations=bootstrap_iterations,
				seed=int(seed) + 1,
			),
		}

	artifact_dir.mkdir(parents=True, exist_ok=True)
	event_table = artifact_dir / "event_latents.parquet"
	embedding_table = artifact_dir / "hyperbolic_embedding.parquet"
	cluster_path = artifact_dir / "adult_clusters.json"
	metrics_path = artifact_dir / "metrics.json"
	fig_dir = artifact_dir / "figures"
	write_records_parquet(build.records, event_table)
	write_records_parquet(embedding_records, embedding_table)
	write_json(cluster_path, metrics["adult_clusters"])
	figures = make_figures(embedding_records, metrics, fig_dir)
	artifacts = {
		"event_latents": event_table.as_posix(),
		"hyperbolic_embedding": embedding_table.as_posix(),
		"adult_clusters": cluster_path.as_posix(),
		"metrics": metrics_path.as_posix(),
	}
	metrics["artifacts"] = artifacts
	metrics["figures"] = figures
	metrics["report_path"] = report_path.as_posix()
	write_json(metrics_path, metrics)
	write_report(report_path, metrics, figures, artifacts)
	return metrics
