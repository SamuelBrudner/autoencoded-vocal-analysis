import pytest


optuna = pytest.importorskip("optuna")
pytest.importorskip("torch")

from ava.models.optuna_sweep import suggest_vae_hyperparameters


def test_suggest_vae_hyperparameters_fixed_trial():
	trial = optuna.trial.FixedTrial(
		{"lr": 1e-3, "z_dim": 16, "model_precision": 10.0}
	)
	params = suggest_vae_hyperparameters(trial)
	assert params["lr"] == 1e-3
	assert params["z_dim"] == 16
	assert params["model_precision"] == 10.0
