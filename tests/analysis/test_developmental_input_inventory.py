import json
import subprocess
import sys
from pathlib import Path

from ava.analysis.developmental_input_inventory import build_developmental_input_inventory


def _manifest() -> dict:
	return {
		"train": [
			{
				"audio_dir_rel": "bells/PKA/33",
				"bird_id_norm": "PKA",
				"regime": "bells",
				"dph": 33,
				"num_files": 2,
			},
			{
				"audio_dir_rel": "bells/PKA/80",
				"bird_id_norm": "PKA",
				"regime": "bells",
				"dph": 80,
				"num_files": 2,
			},
			{
				"audio_dir_rel": "bells/PKB/33",
				"bird_id_norm": "PKB",
				"regime": "bells",
				"dph": 33,
				"num_files": 1,
			},
			{
				"audio_dir_rel": "bells/PKB/80",
				"bird_id_norm": "PKB",
				"regime": "bells",
				"dph": 80,
				"num_files": 1,
			},
		],
		"test": [],
	}


def _touch_inputs(root: Path, rel: str, with_latent: bool) -> None:
	audio_dir = root / "audio" / rel
	roi_dir = root / "roi" / rel
	latent_dir = root / "latent" / rel
	audio_dir.mkdir(parents=True, exist_ok=True)
	roi_dir.mkdir(parents=True, exist_ok=True)
	(audio_dir / "clip.wav").write_bytes(b"")
	(roi_dir / "roi.parquet").write_bytes(b"parquet")
	if with_latent:
		latent_dir.mkdir(parents=True, exist_ok=True)
		(latent_dir / "clip.npz").write_bytes(b"npz")
		(latent_dir / "clip.json").write_text("{}", encoding="utf-8")


def test_input_inventory_summarizes_ready_and_missing_latent_birds(tmp_path: Path):
	for rel in ("bells/PKA/33", "bells/PKA/80"):
		_touch_inputs(tmp_path, rel, with_latent=True)
	for rel in ("bells/PKB/33", "bells/PKB/80"):
		_touch_inputs(tmp_path, rel, with_latent=False)
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

	inventory = build_developmental_input_inventory(
		manifest_path=manifest_path,
		bird_ids=["PKA", "PKB", "PKC"],
		audio_root=tmp_path / "audio",
		roi_root=tmp_path / "roi",
		latent_roots=[tmp_path / "latent"],
	)

	assert inventory["birds"]["PKA"]["status"] == "ready_for_full_rebuild"
	assert inventory["birds"]["PKA"]["latent_npz_count"] == 2
	assert inventory["birds"]["PKB"]["status"] == "missing_latents"
	assert inventory["birds"]["PKC"]["status"] == "missing_manifest_rows"
	assert inventory["summary"]["birds_ready_for_full_rebuild"] == 1
	assert inventory["summary"]["birds_missing_latents"] == ["PKB"]
	assert inventory["summary"]["birds_missing_manifest_rows"] == ["PKC"]


def test_input_inventory_cli_smoke(tmp_path: Path):
	for rel in ("bells/PKA/33", "bells/PKA/80"):
		_touch_inputs(tmp_path, rel, with_latent=True)
	manifest_path = tmp_path / "manifest.json"
	manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
	repo_root = Path(__file__).resolve().parents[2]
	out_dir = tmp_path / "out"
	report_path = tmp_path / "report.md"

	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "inventory_developmental_replication_inputs.py"),
			"--manifest",
			str(manifest_path),
			"--bird-ids",
			"PKA,PKB",
			"--audio-root",
			str(tmp_path / "audio"),
			"--roi-root",
			str(tmp_path / "roi"),
			"--latent-root",
			str(tmp_path / "latent"),
			"--out-dir",
			str(out_dir),
			"--report-out",
			str(report_path),
		],
		cwd=repo_root,
		capture_output=True,
		text=True,
	)
	if result.returncode != 0:
		raise AssertionError(f"CLI failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

	assert (out_dir / "developmental_input_inventory.json").exists()
	assert (out_dir / "developmental_cohort_manifest.json").exists()
	assert report_path.exists()
