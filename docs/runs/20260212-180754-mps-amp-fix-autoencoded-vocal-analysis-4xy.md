# Enable MPS Training (Disable AMP Autocast)

## Bead
- ID: `autoencoded-vocal-analysis-4xy`
- Title: `Enable MPS training: fix autocast/AMP device_type error`
- Date (UTC): `2026-02-12`

## Repro
On Apple Silicon with `accelerator: mps` + `precision: 16-mixed`, Lightning enables AMP and crashes with:
- `User specified an unsupported autocast device_type 'mps'`

RUN_ID:
- `20260212-175851-mps-amp-repro-autoencoded-vocal-analysis-4xy`

Command:
```bash
conda run --no-capture-output -n ava python scripts/launch_birdsong_training.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-175851-mps-amp-repro-autoencoded-vocal-analysis-4xy/training_run \
  --epochs 1 --max-files 1 \
  --trainer-kwargs-json '{"accelerator":"mps","devices":1,"precision":"16-mixed"}' \
  > docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-175851-mps-amp-repro-autoencoded-vocal-analysis-4xy/training_stdout.log 2>&1
```

Artifacts:
- `docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-175851-mps-amp-repro-autoencoded-vocal-analysis-4xy/training_stdout.log`
- `docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-175851-mps-amp-repro-autoencoded-vocal-analysis-4xy/training_run/run_metadata.json`

## Fix
In `src/ava/models/lightning_vae.py`, `build_trainer(...)` now detects MPS and, when the requested `precision` would enable AMP/autocast (`16-mixed`, `bf16-mixed`, etc.), forces `precision=32` and emits a warning. This keeps MPS acceleration without hitting `torch.autocast(device_type='mps')` failures.

Added an Apple Silicon smoke test (skips when MPS is unavailable):
- `tests/models/test_lightning_training.py::test_mps_mixed_precision_is_overridden_to_fp32`

Docs update:
- `README.md` now documents the recommended `precision: 32` setting for MPS.

## Validation Smoke Run (MPS)
RUN_ID:
- `20260212-180754-mps-smoke-autoencoded-vocal-analysis-4xy`

Command (same as repro, now succeeds):
```bash
conda run --no-capture-output -n ava python scripts/launch_birdsong_training.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-180754-mps-smoke-autoencoded-vocal-analysis-4xy/training_run \
  --epochs 1 --max-files 1 \
  --trainer-kwargs-json '{"accelerator":"mps","devices":1,"precision":"16-mixed"}' \
  > docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-180754-mps-smoke-autoencoded-vocal-analysis-4xy/training_stdout.log 2>&1
```

Results (from stdout log):
- Warning: `AMP/autocast is not supported on MPS; forcing precision='16-mixed' -> 32.`
- `GPU available: True (mps), used: True`
- `Trainer.fit` stopped: `max_epochs=1` reached.
- `Training complete. Train files: 1 Test files: 0`

Artifacts:
- `docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-180754-mps-smoke-autoencoded-vocal-analysis-4xy/training_stdout.log`
- `docs/runs/artifacts/autoencoded-vocal-analysis-4xy/20260212-180754-mps-smoke-autoencoded-vocal-analysis-4xy/training_run/`

