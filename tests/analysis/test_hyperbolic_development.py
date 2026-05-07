import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ava.analysis.hyperbolic_development import (
	aggregate_latent_clip_to_events,
	fit_adult_clusters,
	fit_poincare_embedding,
	latent_matrix,
	poincare_distance,
	poincare_radius,
	recenter_poincare,
	standardize_latents,
)


def test_aggregate_latent_clip_to_events_assigns_windows_and_metadata():
	latent = {
		"start_times_sec": np.asarray([0.0, 0.03, 0.06, 0.09], dtype=np.float64),
		"window_length_sec": np.asarray(0.03, dtype=np.float64),
		"mu": np.asarray(
			[
				[1.0, 0.0],
				[3.0, 0.0],
				[0.0, 2.0],
				[0.0, 4.0],
			],
			dtype=np.float32,
		),
		"logvar": np.zeros((4, 2), dtype=np.float32),
		"energy": np.asarray([1.0, 3.0, 1.0, 1.0], dtype=np.float32),
	}
	entry = {
		"audio_dir_rel": "day43 Bells/pk249/33",
		"bird_id_norm": "PK249",
		"regime": "bells",
		"dph": 33,
		"split": "train",
	}

	records, stats = aggregate_latent_clip_to_events(
		latent=latent,
		rois=[(0.0, 0.06), (0.06, 0.12), (0.2, 0.3)],
		metadata={"schema_version": "ava_latent_sequence_v1"},
		entry=entry,
		clip_id="day43 Bells/pk249/33/clip_a",
		clip_stem="clip_a",
	)

	assert stats["roi_events_total"] == 3
	assert stats["roi_events_without_windows"] == 1
	assert len(records) == 2
	assert records[0]["bird_id_norm"] == "PK249"
	assert records[0]["regime"] == "bells"
	assert records[0]["dph"] == 33.0
	assert records[0]["clip_id"] == "day43 Bells/pk249/33/clip_a"
	assert records[0]["n_windows"] == 2
	assert np.allclose(records[0]["mu"], [2.0, 0.0])
	assert np.allclose(records[0]["variance"], [1.0, 1.0])
	assert np.allclose(records[0]["mu_energy_weighted"], [2.5, 0.0])
	assert np.allclose(records[1]["mu"], [0.0, 3.0])


def test_poincare_distance_recenter_and_radius_are_finite():
	points = np.asarray(
		[
			[0.0, 0.0],
			[0.2, 0.0],
			[0.0, 0.3],
		],
		dtype=np.float32,
	)

	dist = poincare_distance(points[:1], points[1:2])
	recentered = recenter_poincare(points, np.asarray([0.1, 0.0], dtype=np.float32))
	radius = poincare_radius(recentered)

	assert np.isfinite(dist).all()
	assert np.isfinite(recentered).all()
	assert np.isfinite(radius).all()
	assert (np.linalg.norm(recentered, axis=1) < 1.0).all()
	assert (radius >= 0.0).all()


def test_adult_clusters_are_deterministic_with_fixed_seed():
	records = []
	for dph, coords in [
		(33, [0.0, 0.0]),
		(40, [0.2, 0.1]),
		(80, [4.0, 0.0]),
		(82, [4.2, 0.1]),
		(85, [0.0, 4.0]),
		(88, [0.1, 4.2]),
	]:
		records.append({"dph": float(dph), "mu": coords})
	x_std, _ = standardize_latents(latent_matrix(records))

	first = fit_adult_clusters(
		records,
		x_std,
		late_dph_min=80,
		min_k=2,
		max_k=2,
		seed=11,
	)
	second = fit_adult_clusters(
		records,
		x_std,
		late_dph_min=80,
		min_k=2,
		max_k=2,
		seed=11,
	)

	assert first.best_k == 2
	assert second.best_k == 2
	assert np.array_equal(first.labels, second.labels)
	assert np.allclose(first.confidence, second.confidence)


def test_poincare_embedding_returns_best_finite_coordinates():
	pytest.importorskip("torch")

	rng = np.random.default_rng(4)
	x = rng.normal(size=(32, 6)).astype(np.float32)
	result = fit_poincare_embedding(x, seed=0, knn=4, epochs=12, lr=1.0)

	assert result.loss_history
	assert np.isfinite(result.loss_history).all()
	assert np.isfinite(result.points).all()
	assert (np.linalg.norm(result.points, axis=1) < 1.0).all()


def test_hyperbolic_development_cli_smoke(tmp_path: Path):
	pytest.importorskip("pyarrow")
	pytest.importorskip("torch")
	import pyarrow as pa
	import pyarrow.parquet as pq

	repo_root = Path(__file__).resolve().parents[2]
	manifest = {"train": [], "test": []}
	latent_root = tmp_path / "latent"
	audio_root = tmp_path / "audio"
	roi_root = tmp_path / "roi"
	latent_root.mkdir()
	rng = np.random.default_rng(0)

	for dph in [33, 40, 80, 85]:
		audio_dir_rel = f"day43 Bells/pk249/{dph}"
		audio_dir = audio_root / audio_dir_rel
		roi_dir = roi_root / audio_dir_rel
		audio_dir.mkdir(parents=True)
		roi_dir.mkdir(parents=True)
		stems = []
		onsets = []
		offsets = []
		for idx in range(2):
			stem = f"clip_{dph}_{idx}"
			stems.append(stem)
			onsets.append([0.0])
			offsets.append([0.11])
			(audio_dir / f"{stem}.wav").write_bytes(b"")
			clip_id = f"{audio_dir_rel}/{stem}"
			latent_dir = latent_root / audio_dir_rel
			latent_dir.mkdir(parents=True, exist_ok=True)
			if dph < 80:
				base = np.asarray([0.1 * idx, 0.0, 0.0, 0.0], dtype=np.float32)
			elif idx == 0:
				base = np.asarray([4.0 + 0.1 * (dph - 80), 0.0, 0.0, 0.0], dtype=np.float32)
			else:
				base = np.asarray([0.0, 4.0 + 0.1 * (dph - 80), 0.0, 0.0], dtype=np.float32)
			mu = np.stack([base, base + rng.normal(0.0, 0.01, size=4)], axis=0)
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
		table = pa.table(
			{
				"clip_stem": stems,
				"onsets_sec": onsets,
				"offsets_sec": offsets,
			}
		)
		pq.write_table(table, roi_dir / "roi.parquet")
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
			str(repo_root / "scripts" / "analyze_hyperbolic_development.py"),
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
			"--embedding-max-events",
			"8",
			"--embedding-epochs",
			"5",
			"--bootstrap-iterations",
			"10",
			"--knn",
			"2",
			"--skip-umap",
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

	assert (out_dir / "event_latents.parquet").exists()
	assert (out_dir / "hyperbolic_embedding.parquet").exists()
	assert (out_dir / "adult_clusters.json").exists()
	assert (out_dir / "metrics.json").exists()
	assert (out_dir / "figures" / "poincare_disk.png").exists()
	assert report_path.exists()
	metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
	assert metrics["event_summary"]["events_sampled"] == 8
	assert metrics["event_summary"]["by_dph"]["33"]["clips_seen"] == 2
	assert metrics["event_summary"]["by_dph"]["33"]["roi_events_without_windows"] == 0
	assert metrics["adult_clusters"]["best_k"] == 2
	assert "spearman_rho" in metrics["radius_age"]
