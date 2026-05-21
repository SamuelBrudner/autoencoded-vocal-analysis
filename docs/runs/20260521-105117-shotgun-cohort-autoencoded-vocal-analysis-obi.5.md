# Shotgun VAE Developmental Branch Commitment Launch Package

## Summary

- Built the PK249 pilot and fixed 11-bird shotgun VAE manifests from the staged developmental cohort manifest.
- PK249 pilot: 58 dph directories, 95,402 wavs, 33-90 dph, 10 training epochs.
- Fixed cohort: 602 dph directories, 1,138,367 wavs, 33-90 dph, 100 training epochs.
- Wrote shotgun configs from `examples/configs/fixed_window_finch_30ms_44k.yaml` with `min_freq=300`, `spec_min_val=1.0`, `kl_beta=1.0`, and `kl_warmup_epochs=20`.
- Staged the four small shotgun input files to the existing baseline S3 run prefix.
- Wrote no-submit AWS Batch payloads for the PK249 pilot and fixed-cohort model.
- Did not submit a GPU job. Current AVA training Batch definition reserves 4 GPUs, so the next action should be an explicit PK249 pilot launch decision.

## Launch Package

- `shotgun_cohort_manifest_summary`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_cohort_manifest_summary.json`
- `pilot_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/pk249_33_90_manifest.json`
- `cohort_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/fixed_11bird_33_90_manifest.json`
- `pilot_config`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_config.yaml`
- `cohort_config`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed_11bird_config.yaml`
- `input_staging_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_input_staging_manifest.json`
- `pilot_training_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_training_payload.json`
- `cohort_training_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed_cohort_training_payload.json`
- `pilot_runner_dry_run`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_runner_dry_run.json`

## Next Action

Submit the PK249 pilot only after confirming the 4-GPU Batch allocation is acceptable for a 10-epoch pipeline validation run. If that is too wasteful, create or select a 1-GPU AVA training job definition before launching.
