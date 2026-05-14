# AWS Staging: Developmental Baseline AVA Replication

This runbook prepares the fixed 11-bird AVA baseline replication inputs before
shotgun VAE work starts. It is intentionally dry-run first: the preparation
command writes payloads and reports, but does not upload audio or submit Batch
jobs.

## Inputs

- Cohort manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-obi.4.1/20260513-011500-developmental-input-inventory/developmental_cohort_manifest.json`
- Default segment config: `examples/configs/birdsong_roi_medium_pilot.yaml`
- PK249 lineage source: `/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/latent_sequences`
- Required runtime values:
  - `AVA_S3_ROOT=s3://<bucket>/ava/developmental-baseline-ava-v1`
  - `AVA_ROI_JOB_QUEUE`
  - `AVA_ROI_JOB_DEFINITION`
  - `AVA_LATENT_JOB_QUEUE`
  - `AVA_LATENT_JOB_DEFINITION`

## Prepare Dry-Run Artifacts

```bash
python scripts/prepare_developmental_baseline_aws.py \
  --s3-root "$AVA_S3_ROOT" \
  --roi-job-queue "$AVA_ROI_JOB_QUEUE" \
  --roi-job-definition "$AVA_ROI_JOB_DEFINITION" \
  --latent-job-queue "$AVA_LATENT_JOB_QUEUE" \
  --latent-job-definition "$AVA_LATENT_JOB_DEFINITION" \
  --run-aws-preflight
```

This writes:

- `aws_staging_plan.json`
- `upload_audio_dry_run_stdout.txt`
- `roi_smoke_batch_payload.json`
- `latent_smoke_batch_payload.json`
- `roi_batch_payload.json`
- `latent_batch_payload.json`
- a report under `docs/runs/`

## Execution Gate

After reviewing the dry-run artifacts and AWS preflight:

1. Write the input staging manifest and review file sizes/checksums:

   ```bash
   python scripts/stage_developmental_baseline_inputs.py \
     --s3-root "$AVA_S3_ROOT" \
     --ava-config-source "$AVA_CONFIG_PATH" \
     --ava-checkpoint-source "$AVA_CHECKPOINT_PATH"
   ```

   Then, after explicit approval, upload `developmental_cohort_manifest.json`,
   `segment_config.yaml`, the recovered AVA `config.yaml`, and
   `checkpoint_050.tar` to the `inputs/` S3 prefix:

   ```bash
   python scripts/stage_developmental_baseline_inputs.py \
     --s3-root "$AVA_S3_ROOT" \
     --ava-config-source "$AVA_CONFIG_PATH" \
     --ava-checkpoint-source "$AVA_CHECKPOINT_PATH" \
     --execute
   ```

2. Sync audio with `scripts/cloud/aws/upload_manifest_audio_to_s3.py`.
3. Submit the ROI smoke payload first; inspect shard summary output.
4. Submit the full ROI array payload.
5. Submit the latent smoke payload with `--export-energy` and
   `hop_length_sec=0.005804988662131519`; inspect exported schema.
6. Submit the full latent array payload.
7. Sync `roi/` and `latents/ava_latent/` locally, rerun the input inventory,
   then rerun `scripts/analyze_developmental_replication.py` in full-rebuild
   mode with all 11 requested birds.

Do not start shotgun VAE training until the 11-bird AVA baseline replication
report is complete.
