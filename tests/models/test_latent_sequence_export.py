import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ava.models.fixed_window_config import (  # noqa: E402
	FixedWindowAugmentationConfig,
	FixedWindowDataConfig,
	FixedWindowExperimentConfig,
	FixedWindowPreprocessConfig,
)
from ava.models.latent_sequence import (  # noqa: E402
	LatentSequenceEncoder,
	encode_clip_to_latent_sequence,
)
from ava.models.vae import VAE  # noqa: E402


def _write_config(path: Path, fs: int, input_shape: tuple[int, int]) -> FixedWindowExperimentConfig:
	config = FixedWindowExperimentConfig(
		preprocess=FixedWindowPreprocessConfig(
			fs=fs,
			num_freq_bins=input_shape[0],
			num_time_bins=input_shape[1],
			nperseg=256,
			noverlap=128,
			window_length=0.1,
			min_freq=300.0,
			max_freq=8000.0,
			spec_min_val=2.0,
			spec_max_val=6.5,
			mel=True,
			time_stretch=False,
			within_syll_normalize=False,
			normalization_mode="none",
		),
		augmentations=FixedWindowAugmentationConfig(enabled=False),
		data=FixedWindowDataConfig(batch_size=2, num_workers=0),
	)
	config.to_yaml(path.as_posix())
	return config


def _write_checkpoint(save_dir: Path, input_shape: tuple[int, int], z_dim: int = 4) -> Path:
	model = VAE(
		save_dir=save_dir.as_posix(),
		device_name="cpu",
		input_shape=input_shape,
		z_dim=z_dim,
		build_optimizer=False,
	)
	layers = model._get_layers()
	state = {name: layer.state_dict() for name, layer in layers.items()}
	state.update(
		{
			"loss": {"train": {}, "test": {}},
			"z_dim": model.z_dim,
			"input_shape": model.input_shape,
			"posterior_type": model.posterior_type,
			"epoch": model.epoch,
			"lr": model.lr,
			"save_dir": model.save_dir,
			"conv_arch": model.conv_arch,
			"model_precision": model.model_precision,
			"log_precision": float(model.log_precision.detach().cpu()),
			"learn_observation_scale": model.learn_observation_scale,
			"decoder_type": model.decoder_type,
		}
	)
	checkpoint = save_dir / "checkpoint_000.tar"
	torch.save(state, checkpoint)
	return checkpoint


def _assert_strictly_increasing(values: np.ndarray) -> None:
	values = np.asarray(values, dtype=np.float64)
	assert values.ndim == 1
	deltas = np.diff(values)
	assert (deltas > 0).all()


def test_encode_clip_to_latent_sequence_invariants(tmp_path):
	repo_root = Path(__file__).resolve().parents[2]
	audio_path = tmp_path / "test.wav"
	shutil.copy(repo_root / "tests" / "data" / "test.wav", audio_path)
	roi_path = tmp_path / "test.txt"
	roi_path.write_text("0.0 0.5\n", encoding="utf-8")

	input_shape = (32, 32)
	config_path = tmp_path / "config.yaml"
	config = _write_config(config_path, fs=32000, input_shape=input_shape)
	ckpt_dir = tmp_path / "checkpoint"
	ckpt_dir.mkdir()
	checkpoint = _write_checkpoint(
		ckpt_dir,
		(config.preprocess.num_freq_bins, config.preprocess.num_time_bins),
		z_dim=4,
	)

	seq = encode_clip_to_latent_sequence(
		checkpoint_path=checkpoint,
		config=config_path,
		audio_path=audio_path,
		roi_path=roi_path,
		device="cpu",
		batch_size=2,
	)

	assert seq.mu.shape == seq.logvar.shape
	assert seq.mu.shape[0] == seq.start_times_sec.shape[0]
	assert seq.mu.shape[1] == 4
	_assert_strictly_increasing(seq.start_times_sec)
	assert np.isfinite(seq.mu).all()
	assert np.isfinite(seq.logvar).all()
	assert seq.metadata["schema_version"] == "ava_latent_sequence_v1"
	assert seq.metadata["clip_id"] == audio_path.stem


def test_latent_sequence_encoder_supports_energy(tmp_path):
	repo_root = Path(__file__).resolve().parents[2]
	audio_path = tmp_path / "test.wav"
	shutil.copy(repo_root / "tests" / "data" / "test.wav", audio_path)
	roi_path = tmp_path / "test.txt"
	roi_path.write_text("0.0 0.5\n", encoding="utf-8")

	input_shape = (32, 32)
	config_path = tmp_path / "config.yaml"
	config = _write_config(config_path, fs=32000, input_shape=input_shape)
	ckpt_dir = tmp_path / "checkpoint"
	ckpt_dir.mkdir()
	checkpoint = _write_checkpoint(
		ckpt_dir,
		(config.preprocess.num_freq_bins, config.preprocess.num_time_bins),
		z_dim=4,
	)

	encoder = LatentSequenceEncoder(
		checkpoint_path=checkpoint,
		config=config_path,
		device="cpu",
	)
	seq = encoder.encode(
		audio_path=audio_path,
		roi_path=roi_path,
		batch_size=2,
		return_energy=True,
	)

	assert seq.energy is not None
	assert seq.energy.shape == (seq.start_times_sec.shape[0],)
	assert np.isfinite(seq.energy).all()


def test_export_latent_sequences_cli_writes_npz_and_json(tmp_path):
	repo_root = Path(__file__).resolve().parents[2]
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	shutil.copy(repo_root / "tests" / "data" / "test.wav", audio_dir / "test.wav")
	(roi_dir / "test.txt").write_text("0.0 0.5\n", encoding="utf-8")

	manifest_path = tmp_path / "manifest.json"
	manifest = {
		"train": [
			{
				"audio_dir_rel": ".",
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"bird_id_norm": "BIRD1",
				"bird_id_raw": "bird1",
				"regime": "test",
				"dph": None,
				"session_label": None,
				"num_files": 1,
				"split": "train",
			}
		],
		"test": [],
	}
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

	input_shape = (32, 32)
	config_path = tmp_path / "config.yaml"
	config = _write_config(config_path, fs=32000, input_shape=input_shape)
	ckpt_dir = tmp_path / "checkpoint"
	ckpt_dir.mkdir()
	checkpoint = _write_checkpoint(
		ckpt_dir,
		(config.preprocess.num_freq_bins, config.preprocess.num_time_bins),
		z_dim=4,
	)

	out_dir = tmp_path / "out"
	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "export_latent_sequences.py"),
			"--manifest",
			str(manifest_path),
			"--split",
			"train",
			"--config",
			str(config_path),
			"--checkpoint",
			str(checkpoint),
			"--out-dir",
			str(out_dir),
			"--device",
			"cpu",
			"--batch-size",
			"2",
			"--max-clips",
			"1",
			"--report-every",
			"1",
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

	npz_path = out_dir / "test.npz"
	json_path = out_dir / "test.json"
	assert npz_path.exists()
	assert json_path.exists()

	npz = np.load(npz_path.as_posix())
	assert "start_times_sec" in npz
	assert "window_length_sec" in npz
	assert "hop_length_sec" in npz
	assert "mu" in npz
	assert "logvar" in npz
	assert np.isfinite(npz["mu"]).all()
	assert np.isfinite(npz["logvar"]).all()

	meta = json.loads(json_path.read_text(encoding="utf-8"))
	assert meta["schema_version"] == "ava_latent_sequence_v1"
	assert meta["clip_id"] == "test"


def test_export_latent_sequences_cli_can_export_energy(tmp_path):
	repo_root = Path(__file__).resolve().parents[2]
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()
	shutil.copy(repo_root / "tests" / "data" / "test.wav", audio_dir / "test.wav")
	(roi_dir / "test.txt").write_text("0.0 0.5\n", encoding="utf-8")

	manifest_path = tmp_path / "manifest.json"
	manifest = {
		"train": [
			{
				"audio_dir_rel": ".",
				"audio_dir": audio_dir.as_posix(),
				"roi_dir": roi_dir.as_posix(),
				"bird_id_norm": "BIRD1",
				"bird_id_raw": "bird1",
				"regime": "test",
				"dph": None,
				"session_label": None,
				"num_files": 1,
				"split": "train",
			}
		],
		"test": [],
	}
	manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

	input_shape = (32, 32)
	config_path = tmp_path / "config.yaml"
	config = _write_config(config_path, fs=32000, input_shape=input_shape)
	ckpt_dir = tmp_path / "checkpoint"
	ckpt_dir.mkdir()
	checkpoint = _write_checkpoint(
		ckpt_dir,
		(config.preprocess.num_freq_bins, config.preprocess.num_time_bins),
		z_dim=4,
	)

	out_dir = tmp_path / "out"
	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "export_latent_sequences.py"),
			"--manifest",
			str(manifest_path),
			"--split",
			"train",
			"--config",
			str(config_path),
			"--checkpoint",
			str(checkpoint),
			"--out-dir",
			str(out_dir),
			"--device",
			"cpu",
			"--batch-size",
			"2",
			"--max-clips",
			"1",
			"--report-every",
			"1",
			"--export-energy",
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

	npz_path = out_dir / "test.npz"
	assert npz_path.exists()
	npz = np.load(npz_path.as_posix())
	assert "energy" in npz
	assert np.isfinite(npz["energy"]).all()
