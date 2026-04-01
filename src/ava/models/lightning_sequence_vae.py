"""
PyTorch Lightning training utilities for the recurrent sequence VAE.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import torch

from ava.models.torch_onnx_compat import patch_torch_onnx_exporter

patch_torch_onnx_exporter()

try:
    import pytorch_lightning as pl
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "PyTorch Lightning is required for ava.models.lightning_sequence_vae. "
        "Install with `pip install pytorch-lightning`."
    ) from exc
from pytorch_lightning.loggers import TensorBoardLogger

from ava.models.run_metadata import write_run_metadata
from ava.models.training_dashboard import TrainingDashboardCallback
from ava.models.vrnn import VRNN


def _extract_sequence_batch(batch):
    if torch.is_tensor(batch):
        return batch, None
    if isinstance(batch, dict):
        x = batch.get("x")
        mask = batch.get("mask")
        if not torch.is_tensor(x):
            raise ValueError("Sequence batches must provide a tensor under 'x'.")
        if mask is not None and not torch.is_tensor(mask):
            raise ValueError("Sequence batch mask must be a tensor when provided.")
        return x, mask
    if isinstance(batch, (list, tuple)) and batch:
        x = batch[0]
        mask = batch[1] if len(batch) > 1 else None
        if not torch.is_tensor(x):
            raise ValueError("Sequence batches must start with a tensor.")
        if mask is not None and not torch.is_tensor(mask):
            mask = None
        return x, mask
    raise ValueError("Unsupported batch format for sequence VAE training.")


def _infer_input_shape(loaders: dict) -> tuple[int, int]:
    loader = loaders.get("train")
    if loader is None:
        raise ValueError("Cannot infer input_shape without a train dataloader.")
    batch = next(iter(loader))
    x, _ = _extract_sequence_batch(batch)
    if x.dim() != 4:
        raise ValueError("Sequence batches must have shape [batch, steps, height, width].")
    return tuple(x.shape[2:])


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _precision_requests_amp(precision: object) -> bool:
    if precision is None:
        return False
    if isinstance(precision, (int, float)):
        return int(precision) == 16
    if isinstance(precision, str):
        return precision.strip().lower() in ("16", "16-mixed", "bf16", "bf16-mixed")
    return False


class SequenceVAELightningModule(pl.LightningModule):
    """Lightning wrapper around :class:`ava.models.vrnn.VRNN`."""

    def __init__(
        self,
        vrnn: Optional[VRNN] = None,
        save_dir: str = "",
        lr: float = 1e-3,
        z_dim: int = 32,
        hidden_dim: int = 256,
        x_feature_dim: int = 256,
        z_feature_dim: int = 128,
        model_precision: float = 10.0,
        learn_observation_scale: bool = False,
        log_precision_min: Optional[float] = None,
        log_precision_max: Optional[float] = None,
        posterior_logvar_min: Optional[float] = None,
        posterior_logvar_max: Optional[float] = None,
        input_shape: Optional[tuple[int, int]] = None,
        kl_beta: float = 1.0,
        kl_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        if vrnn is None:
            if input_shape is None:
                raise ValueError("input_shape is required when constructing a VRNN.")
            vrnn = VRNN(
                save_dir=save_dir,
                lr=lr,
                z_dim=z_dim,
                hidden_dim=hidden_dim,
                x_feature_dim=x_feature_dim,
                z_feature_dim=z_feature_dim,
                model_precision=model_precision,
                learn_observation_scale=learn_observation_scale,
                device_name="cpu",
                input_shape=input_shape,
                log_precision_min=log_precision_min,
                log_precision_max=log_precision_max,
                posterior_logvar_min=posterior_logvar_min,
                posterior_logvar_max=posterior_logvar_max,
            )
        elif save_dir:
            vrnn.save_dir = save_dir
        self.vrnn = vrnn
        self.save_dir = self.vrnn.save_dir
        self.save_hyperparameters(ignore=["vrnn"])
        self.kl_beta = float(kl_beta)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def _kl_weight(self) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.kl_beta
        progress = min(1.0, float(self.current_epoch + 1) / float(self.kl_warmup_epochs))
        return self.kl_beta * progress

    def _compute_loss_and_stats(self, batch):
        x, mask = _extract_sequence_batch(batch)
        return self.vrnn.compute_loss(x, mask=mask, kl_beta=self._kl_weight())

    def _log_stats(self, stats: dict, prefix: str, batch_size: int) -> None:
        for name, value in stats.items():
            self.log(
                f"{prefix}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"recon_nll", "kl", "sequence_length_mean"},
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        loss, stats = self._compute_loss_and_stats(batch)
        x, _ = _extract_sequence_batch(batch)
        batch_size = int(x.shape[0])
        self._train_loss_sum += float(loss.detach().item()) * batch_size
        self._train_loss_count += batch_size
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self._log_stats(stats, "train", batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, stats = self._compute_loss_and_stats(batch)
        x, _ = _extract_sequence_batch(batch)
        batch_size = int(x.shape[0])
        self._val_loss_sum += float(loss.detach().item()) * batch_size
        self._val_loss_count += batch_size
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self._log_stats(stats, "val", batch_size)
        return loss

    def on_train_epoch_end(self) -> None:
        epoch = int(self.current_epoch)
        if self._train_loss_count:
            self.vrnn.loss["train"][epoch] = self._train_loss_sum / self._train_loss_count
        self._train_loss_sum = 0.0
        self._train_loss_count = 0
        self.vrnn.epoch = epoch + 1

    def on_validation_epoch_end(self) -> None:
        epoch = int(self.current_epoch)
        if self._val_loss_count:
            self.vrnn.loss["test"][epoch] = self._val_loss_sum / self._val_loss_count
        self._val_loss_sum = 0.0
        self._val_loss_count = 0

    def configure_optimizers(self):
        return self.vrnn.optimizer


class SequenceCheckpointCallback(pl.Callback):
    """Save VRNN checkpoints on a fixed epoch cadence."""

    def __init__(self, save_freq: Optional[int] = 10) -> None:
        self.save_freq = save_freq

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        if self.save_freq is None:
            return
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch % self.save_freq != 0:
            return
        pl_module.vrnn.epoch = completed_epoch
        pl_module.vrnn.save_state(f"checkpoint_{completed_epoch:03d}.tar")


def build_sequence_trainer(
    save_dir: str = "",
    epochs: int = 100,
    test_freq: Optional[int] = 2,
    save_freq: Optional[int] = 10,
    trainer_kwargs: Optional[dict] = None,
    extra_callbacks: Optional[list] = None,
) -> "pl.Trainer":
    callbacks = [TrainingDashboardCallback(save_dir=save_dir)]
    if save_freq is not None:
        callbacks.insert(0, SequenceCheckpointCallback(save_freq=save_freq))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    trainer_kwargs = dict(trainer_kwargs or {})
    trainer_kwargs.setdefault("accelerator", "auto")
    trainer_kwargs.setdefault("devices", 1)
    accelerator = trainer_kwargs.get("accelerator")
    accelerator_key = accelerator.lower() if isinstance(accelerator, str) else accelerator
    precision = trainer_kwargs.get("precision")
    if _precision_requests_amp(precision):
        auto_mps = (
            accelerator_key in (None, "auto")
            and not torch.cuda.is_available()
            and _mps_available()
        )
        if accelerator_key == "mps" or auto_mps:
            warnings.warn(
                f"AMP/autocast is not supported on MPS; forcing precision={precision!r} -> 32.",
                UserWarning,
            )
            trainer_kwargs["precision"] = 32
    if "precision" not in trainer_kwargs:
        use_amp = accelerator in (None, "auto", "gpu", "cuda") and torch.cuda.is_available()
        trainer_kwargs["precision"] = 16 if use_amp else 32
    trainer_kwargs.setdefault("enable_checkpointing", False)
    if "logger" not in trainer_kwargs:
        log_root = save_dir if save_dir else os.getcwd()
        trainer_kwargs["logger"] = TensorBoardLogger(
            save_dir=log_root,
            name="lightning_logs",
        )
    if test_freq is not None:
        trainer_kwargs.setdefault("check_val_every_n_epoch", test_freq)
    default_root_dir = save_dir if save_dir else None
    return pl.Trainer(
        max_epochs=epochs,
        default_root_dir=default_root_dir,
        callbacks=callbacks,
        **trainer_kwargs,
    )


def train_sequence_vae(
    loaders: dict,
    save_dir: str = "",
    lr: float = 1e-3,
    z_dim: int = 32,
    hidden_dim: int = 256,
    x_feature_dim: int = 256,
    z_feature_dim: int = 128,
    model_precision: float = 10.0,
    learn_observation_scale: bool = False,
    epochs: int = 100,
    log_precision_min: Optional[float] = None,
    log_precision_max: Optional[float] = None,
    posterior_logvar_min: Optional[float] = None,
    posterior_logvar_max: Optional[float] = None,
    test_freq: Optional[int] = 2,
    save_freq: Optional[int] = 10,
    trainer_kwargs: Optional[dict] = None,
    vrnn: Optional[VRNN] = None,
    extra_callbacks: Optional[list] = None,
    input_shape: Optional[tuple[int, int]] = None,
    kl_beta: float = 1.0,
    kl_warmup_epochs: int = 0,
    config_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
):
    """Train a sequence VAE with Lightning."""
    if "train" not in loaders or loaders["train"] is None:
        raise ValueError("loaders must include a non-empty 'train' dataloader.")
    if input_shape is None:
        input_shape = _infer_input_shape(loaders)
    module = SequenceVAELightningModule(
        vrnn=vrnn,
        save_dir=save_dir,
        lr=lr,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        x_feature_dim=x_feature_dim,
        z_feature_dim=z_feature_dim,
        model_precision=model_precision,
        learn_observation_scale=learn_observation_scale,
        log_precision_min=log_precision_min,
        log_precision_max=log_precision_max,
        posterior_logvar_min=posterior_logvar_min,
        posterior_logvar_max=posterior_logvar_max,
        input_shape=input_shape,
        kl_beta=kl_beta,
        kl_warmup_epochs=kl_warmup_epochs,
    )
    trainer = build_sequence_trainer(
        save_dir=module.save_dir,
        epochs=epochs,
        test_freq=test_freq,
        save_freq=save_freq,
        trainer_kwargs=trainer_kwargs,
        extra_callbacks=extra_callbacks,
    )
    if getattr(trainer, "is_global_zero", True):
        write_run_metadata(
            save_dir=module.save_dir,
            config_path=config_path,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
        )
    val_loader = loaders.get("test") if test_freq is not None else None
    trainer.fit(
        module,
        train_dataloaders=loaders["train"],
        val_dataloaders=val_loader,
    )
    return module, trainer


__all__ = [
    "SequenceVAELightningModule",
    "build_sequence_trainer",
    "train_sequence_vae",
]
