# AWS Batch: Full-Scale ROI Extraction (Parquet Bundles)

This is a minimal, reproducible workflow for running full-manifest ROI
extraction on AWS using:
- **S3** for storage (audio + ROI outputs)
- **AWS Batch** array jobs for compute
- **Per-directory Parquet** outputs (one `roi.parquet` per `audio_dir_rel`)

The in-repo tooling lives under:
- `scripts/cloud/aws/upload_manifest_audio_to_s3.py`
- `scripts/cloud/aws/run_birdsong_roi_batch_shard.py`
- `scripts/cloud/aws/submit_birdsong_roi_batch_array_job.py`
- `docker/Dockerfile.roi`

## S3 Layout (Suggested)

Pick one bucket + prefix. Example layout:
- Audio root: `s3://<bucket>/ava/birdsong/audio/`
- ROI root: `s3://<bucket>/ava/birdsong/roi/`
- Manifest: `s3://<bucket>/ava/birdsong/inputs/birdsong_manifest.json`
- Segment config: `s3://<bucket>/ava/birdsong/inputs/segment_config.yaml`

ROI outputs will land at:
- `s3://<bucket>/ava/birdsong/roi/<audio_dir_rel>/roi.parquet`

## Step 1: Upload Inputs (Manifest + Segment Config)

Upload the manifest and your segment config:

```bash
aws s3 cp data/manifests/birdsong_manifest.json \
  s3://<bucket>/ava/birdsong/inputs/birdsong_manifest.json

aws s3 cp <local_segment_config.yaml> \
  s3://<bucket>/ava/birdsong/inputs/segment_config.yaml
```

## Step 2: Upload Audio Subset (Manifest Directories Only)

This uploads only the directories referenced by the manifest, preserving the
`audio_dir_rel` layout.

```bash
python scripts/cloud/aws/upload_manifest_audio_to_s3.py \
  --manifest data/manifests/birdsong_manifest.json \
  --split all \
  --audio-root <LOCAL_AUDIO_ROOT> \
  --s3-audio-root s3://<bucket>/ava/birdsong/audio \
  --jobs 8 \
  --summary-out /tmp/ava_upload_audio_summary.json
```

Notes:
- `aws s3 sync` is idempotent; reruns resume safely.
- For parallel uploads across multiple machines, shard the manifest entries:
  - machine 0: `--num-shards 4 --shard-index 0`
  - machine 1: `--num-shards 4 --shard-index 1`
  - machine 2: `--num-shards 4 --shard-index 2`
  - machine 3: `--num-shards 4 --shard-index 3`

## Step 3: Build + Push the ROI Container Image

Build:

```bash
docker build -f docker/Dockerfile.roi -t ava-roi:latest .
```

Push to ECR (example):

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag ava-roi:latest <account>.dkr.ecr.<region>.amazonaws.com/ava-roi:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/ava-roi:latest
```

## Step 4: Create AWS Batch Resources

You need:
- a Compute Environment (EC2 or Spot EC2)
- a Job Queue
- a Job Definition pointing at your ECR image

The job should allocate enough:
- **vCPU** for per-node parallelism
- **memory** for spectrogram/segmentation
- **ephemeral disk** to download audio for the shard (choose array sizing accordingly)

This repo provides the container + submission tooling, but does not create the
AWS resources automatically.

## Step 5: Submit an Array Job

Choose an array size:
- If you want ~1 directory per job, set `--array-size` to the number of manifest
  entries for your split.
- Larger array sizes increase scheduler overhead but reduce per-job disk needs.

Submit:

```bash
python scripts/cloud/aws/submit_birdsong_roi_batch_array_job.py \
  --job-name ava-birdsong-roi \
  --job-queue <BATCH_JOB_QUEUE> \
  --job-definition <BATCH_JOB_DEFINITION> \
  --array-size <NUM_SHARDS> \
  --manifest-s3-uri s3://<bucket>/ava/birdsong/inputs/birdsong_manifest.json \
  --segment-config-s3-uri s3://<bucket>/ava/birdsong/inputs/segment_config.yaml \
  --s3-audio-root s3://<bucket>/ava/birdsong/audio \
  --s3-roi-root s3://<bucket>/ava/birdsong/roi \
  --split all \
  --skip-existing \
  --download-jobs 8 \
  --jobs 8 \
  --s3-summary-root s3://<bucket>/ava/birdsong/roi/_summaries \
  --submit
```

Reruns are safe:
- `--skip-existing` checks S3 for `<audio_dir_rel>/roi.parquet` before doing any
  downloads/compute.

## Outputs

Per directory:
- `s3://<bucket>/ava/birdsong/roi/<audio_dir_rel>/roi.parquet`

Optional per-shard summaries (if `--s3-summary-root` is set):
- `.../job_summary_shard_<index>.json`
- `.../roi_summary_shard_<index>.json`

