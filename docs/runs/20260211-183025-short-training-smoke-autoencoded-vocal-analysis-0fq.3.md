# Short Training Smoke Run

## Bead
- ID: `autoencoded-vocal-analysis-0fq.3`
- Title: `Short training smoke run`
- Date (UTC): `2026-02-11`

## Objective
Run fixed-window training for a very short schedule on the tiny manifest subset, then verify training artifacts and basic numerical stability.

## Inputs
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json`
- ROI root: `/tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/`
- Birdsong data root: `/Volumes/samsung_ssd/data/birdsong`
- Training entrypoint: `scripts/run_birdsong_validation.py`
- Runtime env: `conda` env `ava` on CPU (`--cpu`)

## Config + Runtime Choices
- Bead config: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml`
- Epochs: `2`
- Batch size: `32`
- Workers: `0`
- Device: `cpu`
- Save frequency: `1` epoch (to force checkpoint output in short run)

## Commands Run
1. Initial training attempt (failed due oversized window length vs ROI durations).
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --epochs 2 --batch-size 32 --num-workers 0 --cpu \
  --spec-cache-dir /tmp/ava_spec_cache_autoencoded-vocal-analysis-0fq.3
```

2. ROI duration sanity check.
```bash
python - <<'PY'
# computed ROI duration distribution under /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332
PY
```
- Result: all ROI durations were ~`0.01161s`.

3. Adjusted config and reran training with `window_length: 0.01`.
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --epochs 2 --batch-size 32 --num-workers 0 --cpu \
  --spec-cache-dir /tmp/ava_spec_cache_autoencoded-vocal-analysis-0fq.3
```

## Output Verification
Training completed successfully:
- `Train files: 455`
- `Test files: 0`
- `Trainer.fit stopped: max_epochs=2 reached`

Verified artifacts:
- Checkpoint: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar`
- Run metadata: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/run_metadata.json`
- Lightning logs: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/lightning_logs/version_0/events.out.tfevents.1770834522.home-mini.local.40697.0`
- Full stdout log: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_stdout.log`

## Numeric Stability Checks
Computed in:
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_checks.json`

Key checks:
- `train_loss` scalar values (TensorBoard):
  - step 63: `39276.9453125`
  - step 127: `18967.92578125`
- All inspected loss/stat series were finite (`no NaN/Inf`):
  - `train_loss`, `train_recon_mse`, `train_recon_nll`, `train_kl`, `train_latent_mean_abs`, `train_latent_var_mean`
- Immediate collapse heuristic (mean abs near 0 and variance near 1):
  - Last epoch `train_latent_mean_abs`: `2.399563789367676`
  - Last epoch `train_latent_var_mean`: `0.13347893953323364`
  - `immediate_collapse`: `false`

## Notes
- The manifest is train-only, so no validation loop ran (`test split empty`), which is expected for this smoke.
- `run_metadata.json` correctly references the bead config, manifest, dataset root, and git commit.
