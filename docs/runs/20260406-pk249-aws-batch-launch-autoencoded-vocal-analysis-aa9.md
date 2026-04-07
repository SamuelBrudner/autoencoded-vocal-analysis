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

## Upload Outcome
- The initial manifest-based uploader was too slow for the `PK249` subset.
- The audio transfer was relaunched with `rclone` against the same prefix and completed successfully.
- Final upload work root: `/tmp/pk249_33_90_rclone_pipeline_20260406_204000`
- Upload log: `/tmp/pk249_33_90_rclone_pipeline_20260406_204000/rclone_upload.log`
- Completion summary from `rclone`:
  - `50.173 GiB / 50.173 GiB`
  - `11840` existing-object checks
  - `83562` transferred objects
  - elapsed `4h20m21.9s`

## First Batch Attempt
- ROI parent job: `pk249-33-90-roi-20260406_204000`
  - job id: `bdc3a4f2-7174-4b06-a4b1-d6f44ea94da7`
  - final status: `FAILED`
- Training job: `pk249-33-90-train-20260406_204000`
  - job id: `4b33f605-0099-4306-849a-8613bc3f2871`
  - final status: `FAILED` (`Dependent Job failed`)
- Failure cause:
  - all `8/8` ROI array children failed immediately on `aws s3 cp` of the manifest
  - CloudWatch logs showed `403 Forbidden` on `HeadObject`
  - the Batch task role `arn:aws:iam::108633434817:role/avaRoiTaskRole` only allowed `ava/birdsong/*`, while this run used `autoencoded-vocal-analysis/pk249-33-90/*`

## IAM Fix
- Updated inline policy `AvaRoiS3Access` on `avaRoiTaskRole`
- Added bucket-list access for prefix `autoencoded-vocal-analysis/*`
- Added object read/write access for `arn:aws:s3:::ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/*`
- Verified with `aws iam simulate-principal-policy` that `s3:GetObject` and `s3:ListBucket` are now allowed for the PK249 manifest path

## Current Live Launch State
- Relaunch work root: `/tmp/pk249_33_90_relaunch_20260407_123338`
- ROI parent job: `pk249-33-90-roi-20260407_123338`
  - job id: `dca35f2a-507e-427d-934a-cf387373f7b2`
  - current status at note update: `PENDING` parent, with all `8/8` child shards launched and `RUNNING`
- Training job: `pk249-33-90-train-20260407_123338`
  - job id: `32fd6a9d-d6a4-4abe-86d7-245c3c74c1c9`
  - run name: `pk249-33-90-4gpu-batch-20260407_123338`
  - expected output root:
    - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/training-runs/pk249-33-90-4gpu-batch-20260407_123338`
  - current status at note update: `PENDING` behind ROI dependency

## Notes
- The local `PK249` audio tree is large (`~63 GiB`), so the audio upload is the long pole.
- The launcher uses the manifest-based uploader rather than a raw top-level sync so only `*.wav` files are pushed.
- The training job is submitted only after ROI submission, with an explicit Batch dependency on the ROI array parent job.
- No parquet ROI files had been written yet at the time this note was updated, but the relaunched ROI shards had moved past the original S3 permission failure and were actively running.
