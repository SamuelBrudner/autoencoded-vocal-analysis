# AWS Developmental Baseline Staging Plan

## Summary

- Cohort rows: 602 directories; wav files: 1138367.
- Birds: 11 (PK249, R426, R467, R404, R150, R493, R470, R203, R425, R229, R122).
- dph range: 33.0..90.0.
- S3 root: `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/developmental-baseline-ava-v1-smoke-20260515`.
- AWS preflight: ok.

## AVA Lineage

- Source metadata: `/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/latent_sequences/day43 Bells/pk249/33/9249_38453.434375_4_11_10_25_30.json`.
- Checkpoint path in export metadata: `/mnt/ava_cache/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/inputs/checkpoint_050.tar`.
- Config path in export metadata: `/mnt/ava_cache/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/inputs/config.yaml`.
- Window/hop: 0.03 sec / 0.005804988662131519 sec.
- Export energy: True.

## Dry-Run Artifacts

- `upload_audio_dry_run_stdout`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/upload_audio_dry_run_stdout.txt`
- `roi_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/roi_batch_payload.json`
- `latent_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/latent_batch_payload.json`
- `roi_smoke_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/roi_smoke_batch_payload.json`
- `latent_smoke_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/latent_smoke_batch_payload.json`
- `aws_staging_plan`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/aws_staging_plan.json`
- `aws_preflight`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-roi-smoke2/aws_preflight.json`

## Next Submit Gate

No AWS jobs were submitted and no audio was uploaded by this preparation command. Before submission, stage the cohort manifest, segment config, recovered AVA config, and recovered checkpoint under the S3 `inputs/` layout, then run the one-shard smoke payloads before the full ROI and latent array payloads.

## Execution Result

Small inputs and the two-directory smoke audio subset were uploaded under this
authorized smoke prefix. The corrected ROI smoke job succeeded as a two-child
Batch array after the submit wrapper was changed to pass explicit entrypoint
arguments, including `--max-dirs 2`.

Smoke validation:

- Batch parent status: `SUCCEEDED`.
- Shard 0: `max_dirs=2`, `planned_total=2`, `selected_for_shard=1`, `roi_ok=1`, `roi_failed=0`.
- Shard 1: `max_dirs=2`, `planned_total=2`, `selected_for_shard=1`, `roi_ok=1`, `roi_failed=0`.
- ROI outputs: `day43 Bells/pk249/33/roi.parquet` and `day43 Bells/pk249/34/roi.parquet`.
- Local parquet inspection: dph 33 has 140 rows; dph 34 has 328 rows; columns are `clip_stem`, `onsets_sec`, and `offsets_sec`.

Operational notes:

- The originally planned `ava/developmental-baseline-ava-v1` prefix failed in
  Batch with S3 `403 Forbidden`; the current Batch task role is scoped to
  existing project prefixes.
- A first authorized-prefix retry succeeded but used the older container
  entrypoint behavior without honoring `AVA_MAX_DIRS`, producing accidental
  ROI parquet outputs under `autoencoded-vocal-analysis/developmental-baseline-ava-v1/roi`.
  That prefix must be cleaned or abandoned before any `--skip-existing` full
  run.
- The full 602-directory audio sync was intentionally stopped during this
  interactive gate. Resume it idempotently, preferably sharded, before the full
  ROI array.
