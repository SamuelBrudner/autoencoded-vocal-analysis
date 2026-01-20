"""
Optuna sweep utilities for VAE hyperparameters.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional

try:
	import optuna
except ImportError as exc:  # pragma: no cover - optional dependency
	raise ImportError(
		"Optuna is required for ava.models.optuna_sweep. "
		"Install with `pip install optuna`."
	) from exc

try:
	import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - optional dependency
	raise ImportError(
		"PyTorch Lightning is required for ava.models.optuna_sweep. "
		"Install with `pip install pytorch-lightning`."
	) from exc

import torch

from ava.models.lightning_vae import train_vae


DEFAULT_VAE_SEARCH_SPACE = {
	"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
	"z_dim": {"type": "int", "low": 8, "high": 64, "step": 8},
	"model_precision": {"type": "float", "low": 1.0, "high": 100.0, "log": True},
}
"""Default Optuna search space for VAE hyperparameters."""


def _suggest_param(trial: "optuna.trial.Trial", name: str,
	spec: Dict[str, Any]):
	param_type = spec.get("type")
	if param_type == "float":
		return trial.suggest_float(
			name,
			spec["low"],
			spec["high"],
			log=spec.get("log", False),
			step=spec.get("step"),
		)
	if param_type == "int":
		return trial.suggest_int(
			name,
			spec["low"],
			spec["high"],
			step=spec.get("step", 1),
		)
	if param_type == "categorical":
		return trial.suggest_categorical(name, spec["choices"])
	raise ValueError(f"Unsupported search space type for {name}: {param_type}")


def suggest_vae_hyperparameters(trial: "optuna.trial.Trial",
	search_space: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
	"""Suggest VAE hyperparameters from a configurable search space."""
	if search_space is None:
		search_space = DEFAULT_VAE_SEARCH_SPACE
	params = {}
	for name, spec in search_space.items():
		params[name] = _suggest_param(trial, name, spec)
	return params


def _metric_value(metrics: Dict[str, Any], name: str) -> Optional[float]:
	value = metrics.get(name)
	if value is None:
		return None
	if isinstance(value, torch.Tensor):
		value = value.detach()
		if value.numel() != 1:
			value = value.mean()
		return value.item()
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _latest_loss(loss_history: Dict[int, float]) -> Optional[float]:
	if not loss_history:
		return None
	epoch = max(loss_history)
	return loss_history[epoch]


def _resolve_objective_metric(trainer, module, metric: str) -> float:
	value = _metric_value(trainer.callback_metrics, metric)
	if value is None and metric == "val_loss":
		value = _latest_loss(module.vae.loss.get("test", {}))
	if value is None and metric == "train_loss":
		value = _latest_loss(module.vae.loss.get("train", {}))
	if value is None or not math.isfinite(value):
		raise ValueError(f"Objective metric '{metric}' was unavailable.")
	return value


class OptunaPruningCallback(pl.Callback):
	"""Report metrics to Optuna and prune unpromising trials."""

	def __init__(self, trial: "optuna.trial.Trial", metric: str = "val_loss",
		min_epochs: int = 0):
		self.trial = trial
		self.metric = metric
		self.min_epochs = min_epochs
		self._reported_epochs = set()

	def _maybe_report(self, trainer, epoch: int):
		if epoch < self.min_epochs or epoch in self._reported_epochs:
			return
		value = _metric_value(trainer.callback_metrics, self.metric)
		if value is None:
			return
		self._reported_epochs.add(epoch)
		self.trial.report(value, step=epoch)
		if self.trial.should_prune():
			raise optuna.TrialPruned(
				f"Trial pruned at epoch {epoch} with {self.metric}={value:.4f}"
			)

	def on_train_epoch_end(self, trainer, pl_module):
		if self.metric.startswith("train_"):
			epoch = int(trainer.current_epoch)
			self._maybe_report(trainer, epoch)

	def on_validation_epoch_end(self, trainer, pl_module):
		if not self.metric.startswith("train_"):
			epoch = int(trainer.current_epoch)
			self._maybe_report(trainer, epoch)


def _write_trial_summary(trial_dir: str, trial: "optuna.trial.Trial",
	params: Dict[str, Any], metric: str, value: float) -> None:
	summary = {
		"trial": trial.number,
		"objective_metric": metric,
		"objective_value": value,
		"params": params,
	}
	path = os.path.join(trial_dir, "trial_summary.json")
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2, sort_keys=True)


def _write_best_summary(save_dir: str, study: "optuna.study.Study") -> None:
	best = study.best_trial
	summary = {
		"trial": best.number,
		"value": best.value,
		"params": best.params,
		"user_attrs": dict(best.user_attrs),
	}
	path = os.path.join(save_dir, "best_trial.json")
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(summary, handle, indent=2, sort_keys=True)


def run_optuna_sweep(loaders: dict, save_dir: str, n_trials: int = 20,
	epochs: int = 25, test_freq: Optional[int] = 1,
	save_freq: Optional[int] = None, vis_freq: Optional[int] = None,
	objective_metric: Optional[str] = None,
	search_space: Optional[Dict[str, Dict[str, Any]]] = None,
	study_name: Optional[str] = None, storage: Optional[str] = None,
	direction: str = "minimize",
	sampler: Optional["optuna.samplers.BaseSampler"] = None,
	pruner: Optional["optuna.pruners.BasePruner"] = None,
	trainer_kwargs: Optional[Dict[str, Any]] = None,
	stopping_kwargs: Optional[Dict[str, Any]] = None) -> "optuna.study.Study":
	"""Run an Optuna sweep for VAE hyperparameters using Lightning training."""
	if objective_metric is None:
		objective_metric = (
			"val_loss" if test_freq is not None and loaders.get("test") is not None
			else "train_loss"
		)
	if trainer_kwargs is None:
		trainer_kwargs = {}
	trainer_kwargs = dict(trainer_kwargs)
	trainer_kwargs.setdefault("enable_progress_bar", False)
	trainer_kwargs.setdefault("enable_model_summary", False)
	if pruner is None:
		pruner = optuna.pruners.NopPruner()
	study = optuna.create_study(
		study_name=study_name,
		storage=storage,
		direction=direction,
		sampler=sampler,
		pruner=pruner,
	)

	def objective(trial: "optuna.trial.Trial") -> float:
		params = suggest_vae_hyperparameters(trial, search_space)
		trial_dir = os.path.join(save_dir, f"trial_{trial.number:03d}")
		os.makedirs(trial_dir, exist_ok=True)
		callbacks = [OptunaPruningCallback(trial, metric=objective_metric)]
		module, trainer = train_vae(
			loaders,
			save_dir=trial_dir,
			lr=params["lr"],
			z_dim=int(params["z_dim"]),
			model_precision=params["model_precision"],
			epochs=epochs,
			test_freq=test_freq,
			save_freq=save_freq,
			vis_freq=vis_freq,
			trainer_kwargs=trainer_kwargs,
			stopping_kwargs=stopping_kwargs,
			extra_callbacks=callbacks,
		)
		value = _resolve_objective_metric(trainer, module, objective_metric)
		trial.set_user_attr("save_dir", trial_dir)
		_write_trial_summary(trial_dir, trial, params, objective_metric, value)
		return value

	study.optimize(objective, n_trials=n_trials)
	_write_best_summary(save_dir, study)
	return study
