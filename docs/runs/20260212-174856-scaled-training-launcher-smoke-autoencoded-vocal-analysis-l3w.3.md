# Scaled Training Launcher + DDP-Safe Artifacts (Smoke Run)

## Bead
- ID: `autoencoded-vocal-analysis-l3w.3`
- Title: `Scaled training launcher`
- Date (UTC): `2026-02-12`

## Objective
Add a launcher suitable for scaled (multi-GPU) training with:
- manifest/config/save-dir inputs
- AMP/caching support via Lightning `trainer_kwargs` and dataset cache options
- safe checkpointing/logging behavior under multi-process training

## Implementation
1) Added a new launcher script:
- `scripts/launch_birdsong_training.py`
  - Supports `--trainer-kwargs-json` overrides for multi-GPU/DDP/AMP configuration.

2) Made legacy artifact callbacks DDP-safe:
- `src/ava/models/lightning_vae.py`
  - `VAECheckpointCallback` and `VAEReconstructionCallback` now only write on `trainer.is_global_zero`.
  - `train_vae(...)` now writes `run_metadata.json` only on `trainer.is_global_zero`.

These guards prevent multi-rank races when running with `devices>1` / DDP.

## Smoke Test
This is a small CPU run to validate end-to-end wiring (logging, metadata, legacy checkpoint emission).

Inputs:
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json`
- Config: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml`
- Epochs: `2` (to force checkpoint emission at epoch 1 with `save_freq=1`)

RUN_ID:
- `20260212-174856-scaled-training-launcher-smoke-autoencoded-vocal-analysis-l3w.3`

Command:
```bash
conda run --no-capture-output -n ava python scripts/launch_birdsong_training.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-l3w.3/20260212-174856-scaled-training-launcher-smoke-autoencoded-vocal-analysis-l3w.3/training_run \
  --epochs 2 --batch-size 32 --num-workers 0 \
  --cpu \
  --trainer-kwargs-json '{"log_every_n_steps": 1}' \
  > docs/runs/artifacts/autoencoded-vocal-analysis-l3w.3/20260212-174856-scaled-training-launcher-smoke-autoencoded-vocal-analysis-l3w.3/training_stdout.log 2>&1
```

Results (from stdout log):
- `Trainer.fit` stopped: `max_epochs=2` reached.
- `Training complete. Train files: 455 Test files: 0`
- Legacy checkpoint emitted: `checkpoint_001.tar` (local-only; ignored by git).

## Artifacts
- Stdout log:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.3/20260212-174856-scaled-training-launcher-smoke-autoencoded-vocal-analysis-l3w.3/training_stdout.log`
- Training run directory:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.3/20260212-174856-scaled-training-launcher-smoke-autoencoded-vocal-analysis-l3w.3/training_run/`
  - `run_metadata.json` (written once on global zero)
  - `lightning_logs/.../events.out.tfevents.*`
  - `checkpoint_001.tar` (ignored by git)

## Example Multi-GPU Usage
Example (CUDA, 8 GPUs) using overrides:
```bash
python scripts/launch_birdsong_training.py \
  --manifest data/manifests/birdsong_manifest.json \
  --config examples/configs/fixed_window_finch_30ms_44k.yaml \
  --save-dir /path/to/run_dir \
  --trainer-kwargs-json '{"accelerator":"gpu","devices":8,"strategy":"ddp","precision":"16-mixed"}'
```

