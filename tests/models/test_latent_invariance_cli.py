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
from ava.models.vae import VAE  # noqa: E402


def _write_config(path: Path, fs: int) -> FixedWindowExperimentConfig:
	config = FixedWindowExperimentConfig(
		preprocess=FixedWindowPreprocessConfig(
			fs=fs,
			num_freq_bins=32,
			num_time_bins=32,
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
		augmentations=FixedWindowAugmentationConfig(
			enabled=True,
			seed=123,
			amplitude_scale=(0.9, 1.1),
			noise_std=0.02,
			time_shift_max_bins=1,
			freq_shift_max_bins=1,
			time_mask_max_bins=2,
			time_mask_count=1,
			freq_mask_max_bins=2,
			freq_mask_count=1,
		),
		data=FixedWindowDataConfig(
			batch_size=2,
			num_workers=0,
			shuffle_train=False,
			shuffle_test=False,
		),
	)
	config.to_yaml(path.as_posix())
	return config


def _write_checkpoint(save_dir: Path, input_shape: tuple[int, int]) -> Path:
	model = VAE(
		save_dir=save_dir.as_posix(),
		device_name="cpu",
		input_shape=input_shape,
		z_dim=4,
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
			"decoder_type": "upsample",
		}
	)
	checkpoint = save_dir / "checkpoint_000.tar"
	torch.save(state, checkpoint)
	return checkpoint


def _assert_all_finite(payload):
	if isinstance(payload, dict):
		for value in payload.values():
			_assert_all_finite(value)
		return
	if isinstance(payload, list):
		for value in payload:
			_assert_all_finite(value)
		return
	if isinstance(payload, (int, float)):
		assert np.isfinite(payload), f"Non-finite value: {payload}"


def test_latent_invariance_cli_runs_without_nans(tmp_path):
	repo_root = Path(__file__).resolve().parents[2]
	audio_dir = tmp_path / "audio"
	roi_dir = tmp_path / "rois"
	audio_dir.mkdir()
	roi_dir.mkdir()

	source_wav = repo_root / "tests" / "data" / "test.wav"
	shutil.copy(source_wav, audio_dir / "test.wav")
	(roi_dir / "test.txt").write_text("0.0 0.5\n", encoding="utf-8")

	config_path = tmp_path / "config.yaml"
	config = _write_config(config_path, fs=32000)
	ckpt_dir = tmp_path / "checkpoint"
	ckpt_dir.mkdir()
	checkpoint = _write_checkpoint(
		ckpt_dir,
		(config.preprocess.num_freq_bins, config.preprocess.num_time_bins),
	)

	output_path = tmp_path / "metrics.json"
	result = subprocess.run(
		[
			sys.executable,
			str(repo_root / "scripts" / "evaluate_latent_metrics.py"),
			"--config",
			str(config_path),
			"--checkpoint",
			str(checkpoint),
			"--audio-dir",
			str(audio_dir),
			"--roi-dir",
			str(roi_dir),
			"--split",
			"train",
			"--train-fraction",
			"1.0",
			"--max-samples",
			"4",
			"--batch-size",
			"2",
			"--num-workers",
			"0",
			"--chunk-size",
			"8",
			"--device",
			"cpu",
			"--output",
			str(output_path),
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

	payload = json.loads(output_path.read_text(encoding="utf-8"))
	assert payload["num_pairs"] > 0
	_assert_all_finite(payload)
