# Medium Pilot Training Run (Fixed-Window Shotgun VAE)

## Bead
- ID: `autoencoded-vocal-analysis-381`
- Title: `Medium-scale pilot (10–20 birds) ROI + training run`
- Date (UTC): `2026-02-12`

## Objective
Execute Phase 2 of `docs/birdsong_training_plan.md` on a medium cohort: train for ~50 epochs, evaluate for collapse/overfit, and capture a reproducible run report with config + artifacts + follow-ups.

## Cohort + Manifest
This run reuses the tuned medium cohort/manifest from `autoencoded-vocal-analysis-381.2`:
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json`
- Cohort: `12 birds`, `12 directories`, `11,653 wav files` (see `docs/runs/20260212-141438-medium-pilot-roi-calibration-autoencoded-vocal-analysis-381.2.md`)

## RUN_ID + Save Dir
- RUN_ID: `20260212-163833-medium-pilot-autoencoded-vocal-analysis-381`
- Artifacts root: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/`
- Training save dir: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/`

## ROI Duration Preflight (window_length=30ms)
From `training_stdout.log`:
- Window length: `0.03s`
- Compatibility: `0.9581` (`193,139 / 201,579` ROI segments compatible)
- ROI duration summary (sec):
  - min: `0.01161`
  - p05: `0.03483`
  - median: `0.09288`
  - p95: `0.30767`
  - max: `0.49923`
- Empty ROI files skipped: `71` (known from the tuned ROI calibration)

## Config
- Bead config: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/fixed_window_medium_pilot_tuned.yaml`
- Baseline reference: `examples/configs/fixed_window_finch_30ms_44k.yaml`

Key deltas vs baseline:
- `preprocess.spec_min_val: 2.0 -> 1.5`
- `training.kl_beta: 1.0 -> 0.7`
- `training.kl_warmup_epochs: 10 -> 20`
- CPU training (`accelerator: cpu`, `precision: 32`) to avoid the known MPS AMP autocast issue observed previously.
- Medium-run logging cadence: `test_freq=5`, `save_freq=5`, `vis_freq=5`

## Command
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/fixed_window_medium_pilot_tuned.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run \
  --epochs 51 --batch-size 128 --num-workers 4 --cpu \
  > docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_stdout.log 2>&1
```

## Results
Training completed successfully:
- `Train files: 7908`
- `Test files: 3674`
- `Trainer.fit` stopped: `max_epochs=51` reached

Loss trends (TensorBoard scalars; see `training_checks.json`):
- `train_loss`: `15890.50` (epoch 0) -> `1859.40` (epoch 50)
- `val_loss` (logged every 5 epochs):
  - epoch 4: `8511.32`
  - epoch 39 (min): `4210.62`
  - epoch 49 (last): `4325.47`
- `val_loss / train_loss` at end: `~2.33`

Collapse/overfit check summary (heuristics; see `training_checks.json`):
- Immediate collapse heuristic (mean abs ~0 and variance ~1): `false`
- Last `train_latent_mean_abs`: `1.5391`
- Last `train_latent_var_mean`: `0.00745`
- Validation curve improved strongly through ~epoch 39, then plateaued/slightly increased by epoch 49 (`+114.85` from min), suggesting mild overfit or an optimization plateau at current hyperparams.

## Artifacts
- Full stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_stdout.log`
- Run metadata: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/run_metadata.json`
- Scalar extraction + heuristics: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/training_checks.json`
- Reconstructions PDF: `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/reconstruction_medium_pilot.pdf`
- Lightning logs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/lightning_logs/version_0/events.out.tfevents.1770914349.home-mini.local.48068.0`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/lightning_logs/version_0/hparams.yaml`
- Checkpoints (local-only; ignored by git): `docs/runs/artifacts/autoencoded-vocal-analysis-381/20260212-163833-medium-pilot-autoencoded-vocal-analysis-381/training_run/checkpoint_*.tar`

## Follow-ups
1) Tune KL schedule around the plateau (e.g., sweep `kl_beta` and `kl_warmup_epochs`) and compare against this baseline using the same manifest.
2) Consider aligning training spectrogram params with the tuned ROI config (e.g., `min_freq` and `spec_min_val`) and re-run a shorter A/B to see if validation improves.
3) Fix/avoid the MPS AMP autocast issue to enable accelerated medium/full runs without forcing CPU.

