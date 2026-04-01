"""
Variational recurrent model for contiguous spectrogram-window sequences.

The model keeps the shotgun-VAE window representation, but replaces the
independent-window latent prior with a recurrent latent dynamics model:

    p(z_t | h_{t-1}) p(x_t | z_t, h_{t-1})
    q(z_t | x_t, h_{t-1})

where ``h_t`` is a deterministic GRU state and ``z_t`` is the stochastic
latent state at each time step.
"""
from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError as exc:  # pragma: no cover - optional in some envs
    torch = None
    Adam = None
    _TORCH_IMPORT_ERROR = exc
    nn = SimpleNamespace(Module=object)
    F = None
else:
    _TORCH_IMPORT_ERROR = None

from ava.models.vae import DEFAULT_INPUT_SHAPE


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> "nn.Sequential":
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


class VRNN(nn.Module):
    """Variational recurrent model over contiguous spectrogram windows."""

    def __init__(
        self,
        save_dir: str = "",
        lr: float = 1e-3,
        z_dim: int = 32,
        hidden_dim: int = 256,
        x_feature_dim: int = 256,
        z_feature_dim: int = 128,
        model_precision: float = 10.0,
        learn_observation_scale: bool = False,
        device_name: str = "auto",
        input_shape: tuple[int, int] = DEFAULT_INPUT_SHAPE,
        build_optimizer: bool = True,
        log_precision_min: Optional[float] = None,
        log_precision_max: Optional[float] = None,
        posterior_logvar_min: Optional[float] = None,
        posterior_logvar_max: Optional[float] = None,
    ) -> None:
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "PyTorch is required for ava.models.vrnn. "
                "Install with `pip install torch`."
            ) from _TORCH_IMPORT_ERROR
        super().__init__()
        self.save_dir = save_dir
        self.lr = float(lr)
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.x_feature_dim = int(x_feature_dim)
        self.z_feature_dim = int(z_feature_dim)
        self.learn_observation_scale = bool(learn_observation_scale)
        self.input_shape = self._normalize_input_shape(input_shape)
        self.input_dim = int(np.prod(self.input_shape))
        model_precision = self._coerce_positive_float(model_precision, "model_precision")
        log_precision = math.log(model_precision)
        log_precision_tensor = torch.tensor(
            log_precision,
            dtype=torch.get_default_dtype(),
        )
        if self.learn_observation_scale:
            self.log_precision = nn.Parameter(log_precision_tensor)
        else:
            self.register_buffer("log_precision", log_precision_tensor)
        self.set_log_precision_bounds(
            log_precision_min=log_precision_min,
            log_precision_max=log_precision_max,
        )
        self.set_logvar_bounds(
            posterior_logvar_min=posterior_logvar_min,
            posterior_logvar_max=posterior_logvar_max,
        )
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        self._requested_device = torch.device(device_name)
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._build_network()
        self.optimizer = None
        if build_optimizer:
            self._ensure_optimizer()
        self.epoch = 0
        self.loss = {"train": {}, "test": {}}
        self.to(self._requested_device)

    @staticmethod
    def _normalize_input_shape(value) -> tuple[int, int]:
        if value is None:
            return tuple(DEFAULT_INPUT_SHAPE)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == 2:
            height, width = int(value[0]), int(value[1])
            if height <= 0 or width <= 0:
                raise ValueError("input_shape must contain positive integers.")
            return (height, width)
        raise ValueError("input_shape must be a 2-tuple of positive integers.")

    @staticmethod
    def _coerce_positive_float(value, name: str) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive float.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive float.")
        return value

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def model_precision(self) -> float:
        return float(torch.exp(self._log_precision_tensor()).detach().cpu())

    @model_precision.setter
    def model_precision(self, value: float) -> None:
        value = self._coerce_positive_float(value, "model_precision")
        with torch.no_grad():
            self.log_precision.copy_(self.log_precision.new_tensor(math.log(value)))

    def set_log_precision_bounds(
        self,
        log_precision_min: Optional[float] = None,
        log_precision_max: Optional[float] = None,
    ) -> None:
        if log_precision_min is not None:
            log_precision_min = float(log_precision_min)
        if log_precision_max is not None:
            log_precision_max = float(log_precision_max)
        if (
            log_precision_min is not None
            and log_precision_max is not None
            and log_precision_min > log_precision_max
        ):
            raise ValueError("log_precision_min cannot exceed log_precision_max.")
        self.log_precision_min = log_precision_min
        self.log_precision_max = log_precision_max

    def set_logvar_bounds(
        self,
        posterior_logvar_min: Optional[float] = None,
        posterior_logvar_max: Optional[float] = None,
    ) -> None:
        if posterior_logvar_min is not None:
            posterior_logvar_min = float(posterior_logvar_min)
        if posterior_logvar_max is not None:
            posterior_logvar_max = float(posterior_logvar_max)
        if (
            posterior_logvar_min is not None
            and posterior_logvar_max is not None
            and posterior_logvar_min > posterior_logvar_max
        ):
            raise ValueError("posterior_logvar_min cannot exceed posterior_logvar_max.")
        self.posterior_logvar_min = posterior_logvar_min
        self.posterior_logvar_max = posterior_logvar_max

    def _apply_logvar_bounds(self, logvar: "torch.Tensor") -> "torch.Tensor":
        if self.posterior_logvar_min is not None:
            logvar = torch.maximum(
                logvar,
                logvar.new_tensor(self.posterior_logvar_min),
            )
        if self.posterior_logvar_max is not None:
            logvar = torch.minimum(
                logvar,
                logvar.new_tensor(self.posterior_logvar_max),
            )
        return logvar

    def _log_precision_tensor(self) -> "torch.Tensor":
        log_precision = self.log_precision
        if self.log_precision_min is not None:
            log_precision = torch.maximum(
                log_precision,
                log_precision.new_tensor(self.log_precision_min),
            )
        if self.log_precision_max is not None:
            log_precision = torch.minimum(
                log_precision,
                log_precision.new_tensor(self.log_precision_max),
            )
        return log_precision

    def _precision_tensor(self) -> "torch.Tensor":
        return torch.exp(self._log_precision_tensor())

    def _build_network(self) -> None:
        self.x_encoder = _build_mlp(self.input_dim, self.hidden_dim, self.x_feature_dim)
        self.z_encoder = _build_mlp(self.z_dim, self.hidden_dim, self.z_feature_dim)
        self.prior_hidden = _build_mlp(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.prior_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        self.posterior_hidden = _build_mlp(
            self.hidden_dim + self.x_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.posterior_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.posterior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        self.decoder_hidden = _build_mlp(
            self.hidden_dim + self.z_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
        )
        self.decoder_out = nn.Linear(self.hidden_dim, self.input_dim)
        self.rnn = nn.GRUCell(self.x_feature_dim + self.z_feature_dim, self.hidden_dim)

    def _ensure_optimizer(self) -> None:
        if self.optimizer is None:
            self.optimizer = Adam(self.parameters(), lr=self.lr)

    def _split_sequence(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        if x.dim() == 4:
            batch, steps = x.shape[:2]
            x = x.reshape(batch, steps, -1)
        elif x.dim() != 3:
            raise ValueError(
                "Expected x to have shape [batch, steps, features] or "
                "[batch, steps, height, width]."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final dimension {self.input_dim}, got {x.shape[-1]}."
            )
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(x.shape[:2]):
                raise ValueError("mask must have shape [batch, steps].")
        return x, mask

    @staticmethod
    def _reparameterize(mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        sample_posterior: bool = True,
    ) -> dict[str, "torch.Tensor"]:
        x, mask = self._split_sequence(x, mask=mask)
        batch_size, steps, _ = x.shape
        hidden = x.new_zeros((batch_size, self.hidden_dim))
        recon_steps = []
        posterior_mu_steps = []
        posterior_logvar_steps = []
        prior_mu_steps = []
        prior_logvar_steps = []
        z_steps = []
        hidden_steps = []

        for step in range(steps):
            step_mask = mask[:, step : step + 1].to(dtype=x.dtype)
            x_t = x[:, step, :]
            x_features = self.x_encoder(x_t)

            prior_features = self.prior_hidden(hidden)
            prior_mu = self.prior_mu(prior_features)
            prior_logvar = self._apply_logvar_bounds(self.prior_logvar(prior_features))

            posterior_input = torch.cat([x_features, hidden], dim=-1)
            posterior_features = self.posterior_hidden(posterior_input)
            posterior_mu = self.posterior_mu(posterior_features)
            posterior_logvar = self._apply_logvar_bounds(
                self.posterior_logvar(posterior_features)
            )
            if sample_posterior:
                z_t = self._reparameterize(posterior_mu, posterior_logvar)
            else:
                z_t = posterior_mu
            z_features = self.z_encoder(z_t)

            decoder_input = torch.cat([z_features, hidden], dim=-1)
            decoder_features = self.decoder_hidden(decoder_input)
            recon_t = torch.sigmoid(self.decoder_out(decoder_features))

            hidden_candidate = self.rnn(torch.cat([x_features, z_features], dim=-1), hidden)
            hidden = step_mask * hidden_candidate + (1.0 - step_mask) * hidden

            recon_steps.append(recon_t)
            posterior_mu_steps.append(posterior_mu)
            posterior_logvar_steps.append(posterior_logvar)
            prior_mu_steps.append(prior_mu)
            prior_logvar_steps.append(prior_logvar)
            z_steps.append(z_t)
            hidden_steps.append(hidden)

        recon = torch.stack(recon_steps, dim=1).view(batch_size, steps, *self.input_shape)
        return {
            "recon": recon,
            "posterior_mu": torch.stack(posterior_mu_steps, dim=1),
            "posterior_logvar": torch.stack(posterior_logvar_steps, dim=1),
            "prior_mu": torch.stack(prior_mu_steps, dim=1),
            "prior_logvar": torch.stack(prior_logvar_steps, dim=1),
            "z": torch.stack(z_steps, dim=1),
            "hidden": torch.stack(hidden_steps, dim=1),
            "mask": mask,
        }

    def encode(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        outputs = self.forward(x, mask=mask, sample_posterior=False)
        return outputs["posterior_mu"], outputs["posterior_logvar"], outputs["hidden"]

    def _masked_mean(
        self,
        values: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(-1)
        weights = mask.to(dtype=values.dtype)
        denom = torch.clamp(weights.sum(), min=torch.finfo(values.dtype).eps)
        return torch.sum(values * weights) / denom

    def compute_loss(
        self,
        x: "torch.Tensor",
        mask: Optional["torch.Tensor"] = None,
        kl_beta: float = 1.0,
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        x_flat, mask = self._split_sequence(x, mask=mask)
        outputs = self.forward(x_flat, mask=mask, sample_posterior=True)
        recon_flat = outputs["recon"].reshape(x_flat.shape[0], x_flat.shape[1], -1)
        precision = self._precision_tensor()
        log_precision = torch.log(precision)
        log_two_pi = math.log(2 * math.pi)

        l2 = torch.sum((x_flat - recon_flat) ** 2, dim=-1)
        recon_nll_per_step = 0.5 * (
            x_flat.shape[-1] * (log_two_pi - log_precision) + precision * l2
        )

        posterior_var = torch.exp(outputs["posterior_logvar"])
        prior_var = torch.exp(outputs["prior_logvar"])
        diff = outputs["posterior_mu"] - outputs["prior_mu"]
        kl_per_step = 0.5 * torch.sum(
            outputs["prior_logvar"]
            - outputs["posterior_logvar"]
            + (posterior_var + diff * diff) / torch.clamp(prior_var, min=1e-8)
            - 1.0,
            dim=-1,
        )

        mask_float = mask.to(dtype=x_flat.dtype)
        valid_steps = torch.clamp(mask_float.sum(), min=1.0)
        recon_nll = torch.sum(recon_nll_per_step * mask_float) / valid_steps
        kl = torch.sum(kl_per_step * mask_float) / valid_steps
        kl_weight = x_flat.new_tensor(float(kl_beta))
        weighted_kl = kl_weight * kl
        loss = recon_nll + weighted_kl

        recon_mse = self._masked_mean((x_flat - recon_flat) ** 2, mask)
        latent_mean_abs = self._masked_mean(outputs["posterior_mu"].abs(), mask)
        latent_var_mean = self._masked_mean(posterior_var, mask)
        sequence_length_mean = mask_float.sum(dim=1).mean()

        stats = {
            "recon_mse": recon_mse,
            "recon_nll": recon_nll,
            "recon_nll_per_dim": recon_nll / x_flat.shape[-1],
            "kl": kl,
            "kl_per_dim": kl / self.z_dim,
            "kl_weight": kl_weight,
            "weighted_kl": weighted_kl,
            "latent_mean_abs": latent_mean_abs,
            "latent_var_mean": latent_var_mean,
            "sequence_length_mean": sequence_length_mean,
            "log_precision": log_precision,
            "model_precision": precision,
        }
        return loss, stats

    def sample(
        self,
        batch_size: int,
        steps: int,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        if device is None:
            device = self.device
        hidden = torch.zeros((int(batch_size), self.hidden_dim), device=device)
        recon_steps = []
        for _ in range(int(steps)):
            prior_features = self.prior_hidden(hidden)
            prior_mu = self.prior_mu(prior_features)
            prior_logvar = self._apply_logvar_bounds(self.prior_logvar(prior_features))
            z_t = self._reparameterize(prior_mu, prior_logvar)
            z_features = self.z_encoder(z_t)
            decoder_features = self.decoder_hidden(torch.cat([z_features, hidden], dim=-1))
            recon_t = torch.sigmoid(self.decoder_out(decoder_features))
            x_features = self.x_encoder(recon_t)
            hidden = self.rnn(torch.cat([x_features, z_features], dim=-1), hidden)
            recon_steps.append(recon_t)
        recon = torch.stack(recon_steps, dim=1)
        return recon.view(int(batch_size), int(steps), *self.input_shape)

    def save_state(self, filename: str) -> None:
        self._ensure_optimizer()
        state = {
            "model_type": "vrnn",
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": self.loss,
            "epoch": self.epoch,
            "lr": self.lr,
            "save_dir": self.save_dir,
            "z_dim": self.z_dim,
            "hidden_dim": self.hidden_dim,
            "x_feature_dim": self.x_feature_dim,
            "z_feature_dim": self.z_feature_dim,
            "input_shape": self.input_shape,
            "model_precision": self.model_precision,
            "log_precision": float(self.log_precision.detach().cpu()),
            "log_precision_min": self.log_precision_min,
            "log_precision_max": self.log_precision_max,
            "posterior_logvar_min": self.posterior_logvar_min,
            "posterior_logvar_max": self.posterior_logvar_max,
            "learn_observation_scale": self.learn_observation_scale,
        }
        filename = os.path.join(self.save_dir, filename)
        torch.save(state, filename)

    def load_state(self, filename: str, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        if checkpoint.get("model_type") not in (None, "vrnn"):
            raise ValueError("Checkpoint is not a VRNN checkpoint.")
        expected = {
            "z_dim": self.z_dim,
            "hidden_dim": self.hidden_dim,
            "x_feature_dim": self.x_feature_dim,
            "z_feature_dim": self.z_feature_dim,
            "input_shape": self.input_shape,
        }
        for key, value in expected.items():
            checkpoint_value = checkpoint.get(key, value)
            if key == "input_shape":
                checkpoint_value = tuple(checkpoint_value)
            if checkpoint_value != value:
                raise ValueError(
                    f"Checkpoint {key} {checkpoint_value!r} does not match model {value!r}."
                )
        self.load_state_dict(checkpoint["model_state"])
        self.loss = checkpoint.get("loss", {"train": {}, "test": {}})
        self.epoch = int(checkpoint.get("epoch", 0))
        if "log_precision" in checkpoint:
            with torch.no_grad():
                self.log_precision.copy_(
                    self.log_precision.new_tensor(float(checkpoint["log_precision"]))
                )
        if load_optimizer and checkpoint.get("optimizer_state") is not None:
            self._ensure_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])


__all__ = ["VRNN"]
