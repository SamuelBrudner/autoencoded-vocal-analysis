# Medium Pilot: Spectrogram Preprocess A/B (min_freq + spec_min_val)

## Bead
- ID: `autoencoded-vocal-analysis-7r5`
- Title: `Medium pilot: A/B training spectrogram params vs ROI tuning`
- Date (UTC): `2026-02-12`

## Setup
Shared across both variants:
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json`
- Epochs: `20`
- Device: MPS (`accelerator: mps`, `precision: 32`)
- Batch size / workers: `128` / `4`
- Validation cadence: every `5` epochs (logged at epochs 4, 9, 14, 19)
- Fixed training schedule: `kl_beta: 0.7`, `kl_warmup_epochs: 20`

Variant A (baseline training preprocess):
- `min_freq: 400`
- `spec_min_val: 1.5`

Variant B (aligned with tuned ROI preprocessing):
- `min_freq: 300`
- `spec_min_val: 1.0`

## Runs
### Variant A (Baseline)
RUN_ID:
- `20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5`

Config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/fixed_window_medium_pilot_specA.yaml`

Command:
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/fixed_window_medium_pilot_specA.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/training_run \
  --batch-size 128 --num-workers 4 \
  > docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/training_stdout.log 2>&1
```

TensorBoard scalars (val_loss):
- epoch 4: `8308.27`
- epoch 9: `6423.87`
- epoch 14: `5434.24`
- epoch 19: `4880.01` (min/last)

### Variant B (Aligned)
RUN_ID:
- `20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5`

Config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/fixed_window_medium_pilot_specB.yaml`

Command:
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/fixed_window_medium_pilot_specB.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/training_run \
  --batch-size 128 --num-workers 4 \
  > docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/training_stdout.log 2>&1
```

TensorBoard scalars (val_loss):
- epoch 4: `6990.79`
- epoch 9: `4996.10`
- epoch 14: `4339.95`
- epoch 19: `4113.27` (min/last)

## Recommendation
For upcoming medium/full runs, prefer the aligned training preprocess params:
- `min_freq: 300`
- `spec_min_val: 1.0`

In this 20-epoch A/B, Variant B improved validation loss at every logged point and ended ~16% lower than Variant A (`4113` vs `4880` at epoch 19).

## Artifacts
Variant A:
- Stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/training_stdout.log`
- Training run: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/training_run/`
- Reconstruction PDF: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-baseline-autoencoded-vocal-analysis-7r5/training_run/reconstruction_medium_pilot.pdf`

Variant B:
- Stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/training_stdout.log`
- Training run: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/training_run/`
- Reconstruction PDF: `docs/runs/artifacts/autoencoded-vocal-analysis-7r5/20260212-181626-spec-ab-aligned-autoencoded-vocal-analysis-7r5/training_run/reconstruction_medium_pilot.pdf`

