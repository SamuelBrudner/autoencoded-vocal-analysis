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
- Telemetry training refresh: `108633434817.dkr.ecr.us-east-1.amazonaws.com/ava-train:20260408-disk-telemetry`
  - digest: `sha256:5d8289d2fb586c6951f819b016fbc4b34d9fc007fa780f27c36109411e2b6f62`
  - also tagged as `latest`
  - pushed: `2026-04-08T07:35:56-04:00`

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
  - final status: `SUCCEEDED`
  - child shard summary: `8/8 SUCCEEDED`
  - ROI outputs written: `58` parquet bundles
- Training job: `pk249-33-90-train-20260407_123338`
  - job id: `32fd6a9d-d6a4-4abe-86d7-245c3c74c1c9`
  - run name: `pk249-33-90-4gpu-batch-20260407_123338`
  - expected output root:
    - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/training-runs/pk249-33-90-4gpu-batch-20260407_123338`
  - final status: `FAILED`
  - failure cause: Lightning DDP aborted at epoch 0 because the module has parameters unused by `training_step` under plain `strategy="ddp"`

## Training Relaunch
- Patched run-specific launcher default in `scripts/cloud/aws/launch_pk249_33_90_batch_pipeline.sh`
  - changed strategy from `ddp` to `ddp_find_unused_parameters_true`
- Submitted training-only relaunch against the completed ROI parquet outputs:
  - work root: `/tmp/pk249_33_90_train_rerun_20260407_132439`
  - job name: `pk249-33-90-train-20260407_132439`
  - job id: `9f52d127-78dc-4fe5-b9a7-08cda28e9fab`
  - run name: `pk249-33-90-4gpu-batch-20260407_132439`
  - trainer kwargs:
    - `{"accelerator":"gpu","devices":4,"strategy":"ddp_find_unused_parameters_true","precision":"16-mixed","log_every_n_steps":10}`
  - final status: `FAILED`
  - failure cause: `OSError: [Errno 28] No space left on device` after training reached roughly epoch `37`

## Scratch-Space Fix
- Patched the AWS training submit path to stop using container `/tmp` for the main run directory:
  - `scripts/cloud/aws/submit_birdsong_training_job.py`
  - `scripts/cloud/aws/launch_pk249_33_90_batch_pipeline.sh`
  - `scripts/cloud/aws/run_birdsong_training_batch_job.py`
- New behavior:
  - training jobs now pass `AVA_WORKDIR=/mnt/ava_cache/<run_name>`
  - the Batch runner creates a per-run scratch tree under `/mnt/ava_cache`
  - the Batch runner also points `TMPDIR`, `TEMP`, and `TMP` into that scratch tree
- Updated the 4-GPU launch template `ava-gpu-batch-root600g`:
  - created version `2`
  - root volume increased from `600 GiB gp3` to `1200 GiB gp3`
  - set version `2` as the default launch template version
- The 4-GPU Batch compute environment `ava-gpu-ec2-4x` uses launch template version `$Latest`, so fresh nodes now inherit the larger root disk.

## Current Training Relaunch
- Submitted a new training-only job against the existing ROI parquet outputs:
  - work root: `/tmp/pk249_33_90_train_rerun_20260407_235058`
  - job name: `pk249-33-90-train-20260407_235058`
  - job id: `78f5232c-f843-4e68-bb75-5f74aabc2452`
  - run name: `pk249-33-90-4gpu-batch-20260407_235058`
  - workdir override:
    - `/mnt/ava_cache/pk249-33-90-4gpu-batch-20260407_235058`
  - current status at note update: `RUNNING`
  - current CloudWatch log stream:
    - `ava-train-gpu-4x/default/46f72812d9b6462dbbcb7bcb3011b3fa`
- Batch needed an explicit scale-out nudge after the relaunch:
  - temporarily set `desiredvCpus=48` on `ava-gpu-ec2-4x`
  - Batch then launched a fresh 4-GPU ECS container instance and moved the job from `RUNNABLE` to `RUNNING`
  - Batch rejected a manual scale-down back to `desiredvCpus=0` while the environment was active; it must scale down on its own after the job is no longer consuming capacity

## Disk Telemetry Patch + New Relaunch
- The mounted-scratch rerun still failed with `OSError: [Errno 28] No space left on device`:
  - job id: `78f5232c-f843-4e68-bb75-5f74aabc2452`
  - run name: `pk249-33-90-4gpu-batch-20260407_235058`
  - status: `FAILED`
  - last visible progress: around epoch `37`, step `996/2048`
- Added reusable disk telemetry instrumentation in the training code:
  - Batch runner snapshots before/after setup, downloads, coverage, and on exception/finalize
  - Lightning callback snapshots at fit start and every `5` train epochs
  - telemetry snapshots write under `logs/disk_telemetry/` and also print to stdout
- Submitted a telemetry-enabled training rerun:
  - work root: `/tmp/pk249_33_90_train_rerun_20260408_113628`
  - job name: `pk249-33-90-train-20260408_113628`
  - job id: `603d68ef-1077-4b2d-8c6a-90a8e3ff9682`
  - run name: `pk249-33-90-4gpu-batch-20260408_113628`
  - workdir override:
    - `/mnt/ava_cache/pk249-33-90-4gpu-batch-20260408_113628`
  - telemetry frequency:
    - every `5` train epochs
  - status at note update: `STARTING`
  - current CloudWatch log stream:
    - `ava-train-gpu-4x/default/d56266aaba1e4a248578ee410ce89cf0`
  - Batch scale-out nudge reapplied:
    - `desiredvCpus=48` on `ava-gpu-ec2-4x`
  - Batch has already launched a fresh 4-GPU ECS container instance for this job

## Notes
- The local `PK249` audio tree is large (`~63 GiB`), so the audio upload is the long pole.
- The launcher uses the manifest-based uploader rather than a raw top-level sync so only `*.wav` files are pushed.
- The training job is submitted only after ROI submission, with an explicit Batch dependency on the ROI array parent job.
- The first 4-GPU training attempt was useful because it proved the AWS GPU path, parquet preflight, and DDP initialization were all working before the unused-parameter error.
