#!/usr/bin/env python3
"""
Lightning training CLI for the AVA VAE (shotgun window training).

Usage:
  python scripts/train_vae.py --config ava/conf/train.yaml \
         --audio-dirs path1[,path2,...] --roi-dirs segs1[,segs2,...]

Writes:
  - checkpoints under outputs/checkpoints/<run_id>/
  - logs/training.jsonl with structured events via loguru
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from loguru import logger

from ava.models.litvae import LitVAE
from ava.models.window_vae_dataset import get_fixed_window_data_loaders, get_window_partition
from ava.preprocessing.utils import get_spec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAE with Lightning")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--audio-dirs", type=str, required=True)
    p.add_argument("--roi-dirs", type=str, required=True)
    return p.parse_args()


def _slurp_config(path: Path) -> dict:
    text = path.read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise RuntimeError("Config must parse to a mapping")
    return cfg


def _mk_dirs(run_id: str) -> dict[str, Path]:
    out = Path("outputs")
    ckpt = out / "checkpoints" / run_id
    figs = out / "figs" / run_id
    logs = Path("logs")
    ckpt.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {"checkpoints": ckpt, "figs": figs, "logs": logs}


def _build_params(num_time_bins: int, num_freq_bins: int, fs: int) -> dict:
    return {
        "fs": fs,
        "get_spec": get_spec,
        "num_freq_bins": num_freq_bins,
        "num_time_bins": num_time_bins,
        "nperseg": 512,
        "noverlap": 256,
        "max_dur": int(1e9),
        "window_length": 0.12,
        "min_freq": 400,
        "max_freq": 10000,
        "spec_min_val": 2.0,
        "spec_max_val": 6.5,
        "mel": True,
        "time_stretch": False,
        "within_syll_normalize": False,
        "real_preprocess_params": ("min_freq", "max_freq", "spec_min_val", "spec_max_val"),
        "int_preprocess_params": tuple([]),
        "binary_preprocess_params": ("mel", "within_syll_normalize"),
    }


def main() -> None:
    args = _parse_args()
    cfg = _slurp_config(args.config)

    # Configure logging
    run_id = f"run_{cfg['model'].get('seed', 0)}"
    paths = _mk_dirs(run_id)
    logger.configure(handlers=[{"sink": paths["logs"] / "training.jsonl", "serialize": True}])
    logger.info("config_loaded", cfg=cfg)

    # Parse data dirs
    audio_dirs: list[str] = [s for s in args.audio_dirs.split(",") if s]
    roi_dirs: list[str] = [s for s in args.roi_dirs.split(",") if s]
    if len(audio_dirs) != len(roi_dirs):
        raise RuntimeError("audio-dirs and roi-dirs must have equal lengths")

    # Build datasets/loaders
    val_frac = float(cfg["data"].get("val_fraction", 0.2))
    split = max(0.0, min(1.0, 1.0 - val_frac))
    partition = get_window_partition(audio_dirs, roi_dirs, split=split)
    # Training params shape must match model's expected X_SHAPE
    from ava.models.vae import X_SHAPE

    fs = int(cfg["data"].get("fs", 32000))
    p = _build_params(num_time_bins=X_SHAPE[1], num_freq_bins=X_SHAPE[0], fs=fs)
    loaders = get_fixed_window_data_loaders(
        partition,
        p,
        batch_size=int(cfg["data"].get("batch_size", 128)),
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    # Build model
    model = LitVAE(
        latent_dim=int(cfg["model"].get("latent_dim", 32)),
        model_precision=float(cfg["model"].get("model_precision", 10.0)),
        lr=float(cfg["trainer"].get("lr", 1e-3)),
        beta_kl=float(cfg["model"].get("beta_kl", 1.0)),
    )

    # Seed
    try:
        from lightning import seed_everything  # type: ignore
    except Exception as _e:  # pragma: no cover
        raise RuntimeError("Lightning is required") from _e
    seed_everything(int(cfg["model"].get("seed", 13)), workers=True)

    # Trainer + callbacks
    from lightning import Trainer  # type: ignore
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

    ckpt_cb = ModelCheckpoint(
        dirpath=str(paths["checkpoints"]),
        monitor=cfg["callbacks"]["checkpoint"].get("monitor", "val_loss"),
        mode=cfg["callbacks"]["checkpoint"].get("mode", "min"),
        save_top_k=int(cfg["callbacks"]["checkpoint"].get("save_top_k", 1)),
        filename="best",
    )
    callbacks = [ckpt_cb]
    es_cfg = cfg["callbacks"].get("early_stopping", {})
    if bool(es_cfg.get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                mode=es_cfg.get("mode", "min"),
                patience=int(es_cfg.get("patience", 10)),
            )
        )

    tr = Trainer(
        max_epochs=int(cfg["trainer"].get("max_epochs", 100)),
        accelerator=str(cfg["trainer"].get("accelerator", "auto")),
        devices=cfg["trainer"].get("devices", 1),
        deterministic=bool(cfg["trainer"].get("deterministic", True)),
        enable_progress_bar=bool(cfg["trainer"].get("enable_progress_bar", True)),
        callbacks=callbacks,
        default_root_dir=str(paths["checkpoints"]),
        log_every_n_steps=1,
    )
    logger.info("trainer_created", device=str(tr.accelerator), ckpt_dir=str(paths["checkpoints"]))
    tr.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["test"])
    logger.info("training_finished", best_ckpt=str(ckpt_cb.best_model_path))


if __name__ == "__main__":
    main()
