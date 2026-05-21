# Shotgun VAE Developmental Branch Commitment Launch Package

## Summary

- Built the PK249 pilot and fixed 11-bird shotgun VAE manifests from the staged developmental cohort manifest.
- PK249 pilot: 58 dph directories, 95,402 wavs, 33-90 dph, 10 training epochs.
- Fixed cohort: 602 dph directories, 1,138,367 wavs, 33-90 dph, 100 training epochs.
- Wrote shotgun configs from `examples/configs/fixed_window_finch_30ms_44k.yaml` with `min_freq=300`, `spec_min_val=1.0`, `kl_beta=1.0`, and `kl_warmup_epochs=20`.
- Staged the four small shotgun input files to the existing baseline S3 run prefix.
- Wrote no-submit AWS Batch payloads for the PK249 pilot and fixed-cohort model.
- Did not submit a GPU job. Current AVA training Batch definition reserves 4 GPUs, so the next action should be an explicit PK249 pilot launch decision.
- Static utilization audit: the original PK249 pilot payload is wasteful as written because the Batch job reserves 4 GPUs while the shotgun config requests `devices: 1`.
- Added an explicit 4-GPU utilization assay payload that overrides Lightning to `devices=4, strategy=ddp` and records `nvidia-smi` utilization every 5 seconds during a one-epoch PK249 run.

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
- `4gpu_utilization_assay_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_4gpu_utilization_assay_payload.json`
- `4gpu_utilization_assay_runner_dry_run`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_4gpu_utilization_assay_runner_dry_run.json`

## Utilization Assay

Do not submit the original PK249 pilot payload to the 4-GPU job definition; it would run with one Lightning device and leave the other three GPUs idle.

The utilization assay payload is the appropriate next paid run if we want to evaluate the existing 4-GPU definition. It is intentionally short: PK249 only, one epoch, `dataset_length=8192`, `batch_size=128`, `devices=4`, `strategy=ddp`, and GPU telemetry enabled. Treat the 4-GPU definition as efficient only if all four GPUs are visible in `gpu_utilization.csv`, all four show sustained non-trivial utilization during the training window, and wall-clock throughput is materially better than a 1-GPU run with the same dataset length. If DDP fails, only one GPU is active, or utilization is dominated by download/preprocessing stalls, use or create a 1-GPU AVA training definition for the PK249 pilot.
