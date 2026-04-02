# PK249 Single-Bird 33-90 DPH Redo Assets + Smoke Validation

## Bead
- ID: `autoencoded-vocal-analysis-8ot`
- Title: `Redo PK249 single-bird 33-90dph shotgun VAE workflow`
- Date (UTC): `2026-04-02`

## Objective
Stand up a reproducible single-bird redo path for the historical `33-90dph` shotgun VAE run using the current recommended training regime, AWS-ready parquet ROI preprocessing, and a deterministic within-bird holdout split.

## Bird Selection Audit
`PK249` was kept as the target bird after checking day-by-day file coverage, not just total files.

Coverage summary:
- Distinct DPH in `[33, 90]`: `58 / 58` (no missing days)
- Total wav files: `95,402`
- Minimum files on any day: `140`
- Days with fewer than `250` files: `1`
- Days with fewer than `500` files: `3`
- Days with at least `1000` files: `53`

Artifact:
- `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/pk249_33_90_coverage_summary.json`

## Full-Run Assets
Generated manifest:
- `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json`
- Split rule: sort by `(dph, audio_dir_rel)` and assign every 5th directory at offset 4 to `test`
- Result: `47 train dirs / 76,374 files`, `11 test dirs / 19,028 files`
- Test DPHs: `37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87`

Run config:
- `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/fixed_window_pk249_33_90.yaml`

Key deltas vs `examples/configs/fixed_window_finch_30ms_44k.yaml`:
- `preprocess.min_freq: 400 -> 300`
- `preprocess.spec_min_val: 2.0 -> 1.0`
- `training.kl_beta: 1.0` (kept at the sweep winner)
- `training.kl_warmup_epochs: 10 -> 20`
- `training.epochs: 51`
- `training.test_freq/save_freq/vis_freq: 5/5/5`
- `training.vis_filename: reconstruction_pk249_33_90.pdf`
- `training.trainer_kwargs: accelerator=gpu, devices=1, precision=16-mixed, log_every_n_steps=1`
- `data.batch_size/num_workers/pin_memory/persistent_workers: 128/4/true/true`
- `data.spec_cache_dir: /tmp/ava_spec_cache_autoencoded-vocal-analysis-8ot`

## Cloud Dry Runs
These dry runs use placeholder S3/job identifiers and should be rewritten with the real bucket, queue, and job definition before submission.

Upload subset dry run:
- Log: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/dry_runs/upload_manifest_audio_dry_run.log`
- Result: `58` planned sync tasks from `day43 Bells/pk249/33` through `day43 Bells/pk249/90`

ROI Batch submit payload:
- Payload: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/dry_runs/pk249_roi_batch_submit_payload.json`
- Stdout copy: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/dry_runs/pk249_roi_batch_submit_stdout.log`
- Result: array size `8`, parquet ROI output, `split=all`, `skip_existing=1`

## Local Smoke Validation
To exercise the parquet/streaming path locally, a smaller smoke manifest was generated for `PK249` `33-37dph`:
- `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_37_smoke.json`
- Result: `4 train dirs / 1,703 files`, `1 test dir / 592 files`

### ROI Parquet Smoke
Command used:
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi_medium_pilot.yaml \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_37_smoke.json \
  --split all \
  --roi-root /tmp/ava_roi_autoencoded-vocal-analysis-8ot_smoke \
  --roi-output-format parquet \
  --jobs 2 \
  --summary-out docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/roi_summary.json
```

Outputs:
- Stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/roi_stdout.log`
- Summary: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/roi_summary.json`

Result:
- `planned_total=5`, `ok=5`, `failed=0`
- Per-directory parquet bundles written under `/tmp/ava_roi_autoencoded-vocal-analysis-8ot_smoke/.../roi.parquet`

Coverage aggregation:
- Summary: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/coverage/summary.json`
- Per-directory CSV: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/coverage/per_directory.csv`

Coverage result:
- `missing_roi_dirs=0`
- `missing_roi_files=0`
- `empty_roi_files=20` out of `2,295` wav files (`0.87%`, below the `1%` investigation threshold)
- `segments_total=39,099`

### Streaming Training Smoke
The initial smoke run exposed a compatibility bug in the streaming dataset: the legacy reconstruction callback samples `loader.dataset[indices]` with a NumPy vector, but `ManifestFixedWindowDataset.__getitem__` only accepted scalars. This was fixed in `src/ava/models/manifest_window_dataset.py`, and a regression check was added in `tests/models/test_manifest_window_dataset.py`.

Command used:
```bash
conda run --no-capture-output -n ava python scripts/launch_birdsong_training.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_37_smoke.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-8ot/fixed_window_pk249_33_90.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run \
  --roi-root /tmp/ava_roi_autoencoded-vocal-analysis-8ot_smoke \
  --streaming --roi-format parquet \
  --epochs 2 --batch-size 32 --num-workers 0 --dataset-length 64 --cpu
```

Outputs:
- Stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_stdout.log`
- Run metadata: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/run_metadata.json`
- Dashboard JSON/HTML:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/training_dashboard.json`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/training_dashboard.html`
- Reconstruction PDF: `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/reconstruction_pk249_33_90.pdf`
- Lightning logs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/lightning_logs/version_0/hparams.yaml`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-8ot/smoke/training_run/lightning_logs/version_0/events.out.tfevents.1775162388.home-mini.local.12025.0`

Result:
- Parquet preflight succeeded with `compatible_fraction=0.8114`
- `Trainer.fit` stopped at `max_epochs=2`
- Final dashboard status: `completed`
- Final dashboard summary: `current_epoch=2`, `latest_step=4`, `device=cpu`

Note:
- Because the full-run config keeps `training.test_freq=5`, the 2-epoch smoke only exercised the validation loader through Lightning sanity checking, not a scheduled validation epoch. That is sufficient for path validation, but the full run will not emit its first scheduled validation metric until epoch 4.

## Targeted Validation Commands
Tests run:
```bash
pytest -q tests/models/test_manifest_window_dataset.py \
  tests/scripts/test_build_birdsong_single_bird_manifest.py
```

Result:
- `3 passed`

## Full Run Commands
Build / refresh the full single-bird manifest:
```bash
python scripts/build_birdsong_single_bird_manifest.py \
  --source-manifest data/manifests/birdsong_manifest.json \
  --bird-id PK249 \
  --min-dph 33 --max-dph 90 \
  --test-every-n 5 --test-offset 4 \
  --out docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json
```

Upload the PK249 subset to S3:
```bash
python scripts/cloud/aws/upload_manifest_audio_to_s3.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json \
  --split all \
  --s3-audio-root s3://<bucket>/autoencoded-vocal-analysis/pk249-33-90/audio
```

Emit the ROI Batch payload:
```bash
python scripts/cloud/aws/submit_birdsong_roi_batch_array_job.py \
  --job-name pk249-33-90-roi \
  --job-queue <queue> \
  --job-definition <job-definition> \
  --array-size 8 \
  --manifest-s3-uri s3://<bucket>/autoencoded-vocal-analysis/pk249-33-90/manifest_pk249_33_90.json \
  --segment-config-s3-uri s3://<bucket>/autoencoded-vocal-analysis/pk249-33-90/birdsong_roi_medium_pilot.yaml \
  --s3-audio-root s3://<bucket>/autoencoded-vocal-analysis/pk249-33-90/audio \
  --s3-roi-root s3://<bucket>/autoencoded-vocal-analysis/pk249-33-90/roi \
  --split all \
  --skip-existing \
  --submit
```

Validate parquet ROI coverage after the real ROI run:
```bash
conda run --no-capture-output -n ava python scripts/report_birdsong_roi_coverage.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json \
  --split all \
  --roi-format parquet \
  --roi-root <local-roi-root> \
  --out-dir <coverage-out-dir> \
  --fail-on-empty
```

Launch the full streaming training run:
```bash
conda run --no-capture-output -n ava python scripts/launch_birdsong_training.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-8ot/fixed_window_pk249_33_90.yaml \
  --save-dir <run-dir> \
  --audio-root <local-audio-root> \
  --roi-root <local-roi-root> \
  --streaming \
  --roi-format parquet \
  --dataset-length 2048
```
