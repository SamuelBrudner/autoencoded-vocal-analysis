# PK249 33-90 DPH AWS Batch Launch

## Bead
- ID: `autoencoded-vocal-analysis-aa9`
- Title: `Execute full PK249 33-90dph ROI parquet + training run`
- Date (UTC): `2026-04-07`

## Objective
Launch the real AWS path for the `PK249` `33-90dph` single-bird redo:
- upload the tracked manifest/config assets and audio subset to S3
- generate parquet ROIs with AWS Batch
- launch 4-GPU streaming training after ROI completion

## AWS Resources
- Region: `us-east-1`
- Account: `108633434817`
- Bucket: `s3://ava-birdsong-us-east-1-a1859d31`
- PK249 prefix: `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90`
- ROI queue / job definition: `ava-roi-queue` / `ava-roi-jobdef`
- Training queue / job definition: `ava-gpu-queue-4x` / `ava-train-gpu-4x`

## Images
- Refreshed ROI image: `108633434817.dkr.ecr.us-east-1.amazonaws.com/ava-roi:latest`
  - digest: `sha256:172bd06d04a460321f947c6d74d60f8f021d5dfc8f0d5796297bdf26740c322f`
  - pushed: `2026-04-06T19:46:36-04:00`
- New training image: `108633434817.dkr.ecr.us-east-1.amazonaws.com/ava-train:latest`
  - digest: `sha256:8486b012995fec46877395f91a863ae732e87817c42e7965bc455e7b4664bc6e`
  - pushed: `2026-04-06T19:59:46-04:00`

## Registered Training Job Definition
- ARN: `arn:aws:batch:us-east-1:108633434817:job-definition/ava-train-gpu-4x:1`
- Platform: `EC2`
- Resources: `44 vCPU`, `180000 MiB`, `4 GPU`
- Host cache mount: `/mnt/ava_cache -> /mnt/ava_cache`

## Pipeline Launcher
- Script: `scripts/cloud/aws/launch_pk249_33_90_batch_pipeline.sh`
- Runner submit helper: `scripts/cloud/aws/submit_birdsong_training_job.py`
- Batch runner image entrypoint: `scripts/cloud/aws/run_birdsong_training_batch_job.py`

## Live Launch State
- Active run stamp: `20260407_001600`
- Work root: `/tmp/pk249_33_90_aws_pipeline_20260407_001600`
- Training run name: `pk249-33-90-4gpu-batch-20260407_001600`
- Expected training output root:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/training-runs/pk249-33-90-4gpu-batch-20260407_001600`
- Current stage at handoff:
  - manifest/config uploads completed
  - audio upload running with `24` concurrent directory syncs
  - ROI and training Batch submissions will happen automatically after upload completes

## Notes
- The local `PK249` audio tree is large (`~63 GiB`), so the audio upload is the long pole.
- The launcher uses the manifest-based uploader rather than a raw top-level sync so only `*.wav` files are pushed.
- The training job is submitted only after ROI submission, with an explicit Batch dependency on the ROI array parent job.
