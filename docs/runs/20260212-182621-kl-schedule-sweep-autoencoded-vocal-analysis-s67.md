# Medium Pilot: KL Schedule Sweep (beta + warmup)

## Bead
- ID: `autoencoded-vocal-analysis-s67`
- Title: `Medium pilot: sweep KL beta/warmup`
- Date (UTC): `2026-02-12`

## Objective
Evaluate whether changing the KL schedule improves medium-manifest validation loss without obvious collapse, using the same medium cohort/manifest as `autoencoded-vocal-analysis-381`.

## Setup (Shared)
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json`
- Device: MPS (`accelerator: mps`, `precision: 32`)
- Epochs: `51` (val logged every 5 epochs; last logged at epoch 49)
- Fixed (vs baseline): preprocess params kept at the medium pilot baseline (`min_freq: 400`, `spec_min_val: 1.5`) to isolate KL effects.

## Variants
All variants below used `test_freq: 5`, `vis_freq: 5`, and disabled legacy checkpoints (`save_freq: null`).

### Variant 1: Lower KL beta
- `kl_beta: 0.5`
- `kl_warmup_epochs: 20`

RUN_ID:
- `20260212-182621-kl-beta05-warm20-autoencoded-vocal-analysis-s67`

Config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-s67/20260212-182621-kl-beta05-warm20-autoencoded-vocal-analysis-s67/fixed_window_medium_pilot_kl_beta05_warm20.yaml`

TensorBoard (val_loss):
- epoch 4: `8012.95`
- min: `4220.31` (epoch 29)
- epoch 49 (last): `4429.81`

### Variant 2: Longer warmup
- `kl_beta: 0.7`
- `kl_warmup_epochs: 30`

RUN_ID:
- `20260212-182621-kl-beta07-warm30-autoencoded-vocal-analysis-s67`

Config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-s67/20260212-182621-kl-beta07-warm30-autoencoded-vocal-analysis-s67/fixed_window_medium_pilot_kl_beta07_warm30.yaml`

TensorBoard (val_loss):
- epoch 4: `8761.60`
- min: `4241.23` (epoch 44)
- epoch 49 (last): `4594.48`

### Variant 3: Higher KL beta
- `kl_beta: 1.0`
- `kl_warmup_epochs: 20`

RUN_ID:
- `20260212-182621-kl-beta10-warm20-autoencoded-vocal-analysis-s67`

Config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-s67/20260212-182621-kl-beta10-warm20-autoencoded-vocal-analysis-s67/fixed_window_medium_pilot_kl_beta10_warm20.yaml`

TensorBoard (val_loss):
- epoch 4: `8640.29`
- min: `4253.79` (epoch 49)
- epoch 49 (last): `4253.79`

## Baseline Reference (Prior Run)
From `autoencoded-vocal-analysis-381` (CPU run; same manifest):
- `kl_beta: 0.7`, `kl_warmup_epochs: 20`
- val_loss epoch 4: `8511.32`
- val_loss min: `4210.62` (epoch 39)
- val_loss epoch 49 (last): `4325.47`

## Recommendation
For the next medium/full training run, prefer:
- `kl_beta: 1.0`
- `kl_warmup_epochs: 20`

Rationale: among the tested variants it had the best final validation loss at epoch 49 (`4253.79`) and was still improving at the end of the run (min at epoch 49), while the baseline and other variants either plateaued or degraded by epoch 49.

## Commands
Each run followed this pattern:
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --config <one of the configs above> \
  --save-dir <RUN_DIR>/training_run \
  --batch-size 128 --num-workers 4 \
  > <RUN_DIR>/training_stdout.log 2>&1
```

## Artifacts
Each RUN_DIR contains:
- run config YAML
- `training_stdout.log`
- `training_run/run_metadata.json`
- `training_run/reconstruction_medium_pilot.pdf`
- `training_run/lightning_logs/version_0/events.out.tfevents.*`

