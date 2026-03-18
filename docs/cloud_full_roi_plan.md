# Cloud Plan: Full Birdsong ROI Extraction

This document describes an actionable, cloud-friendly plan for running ROI
extraction at full (manifest) scale, plus a recommended output format that
avoids millions of tiny files.

## Current Scale (Local Facts)

As of `data/manifests/birdsong_manifest.json`:
- Directories in manifest: 3,145 (`train=2,624`, `test=521`)
- Wav files in manifest: 3,821,776 (`train=3,190,613`, `test=631,163`)
- Manifest audio root: `/Volumes/samsung_ssd/data/birdsong`
- Manifest ROI root: `/Volumes/samsung_ssd/data/ava_roi_birdsong_full`

As of `data/metadata/birdsong/birdsong_files.parquet`:
- Total metadata rows: 5,174,439
- Total audio size: ~2,929.9 GiB (all metadata rows)
- Estimated audio size for the manifest directories: ~1,856.4 GiB

## TL;DR Recommendation

Use AWS as the default target:
- Storage: S3 (raw audio + ROI outputs)
- Compute: AWS Batch (array jobs) on EC2 Spot (CPU instances)
- Output: write ROI as **one Parquet (or compressed tar) per audio directory**
  to keep object count reasonable (3,145 objects, not 3.8M+).

If you need POSIX semantics for existing code paths, add a short-lived shared
filesystem layer:
- FSx for Lustre linked to S3 (fast, POSIX, expensive but transient), or
- A single large EBS volume per worker that writes per-wav `.txt` locally and
  uploads a single bundle per directory.

## Key Blockers / Dependencies

1. **Data transfer**: audio is not currently in the cloud. Uploading ~1.8 TiB
   (manifest subset) is the gating step.
2. **Output format decision**:
   - Per-wav `.txt` matches current training code but produces millions of
     objects and is awkward on object stores.
   - Per-directory Parquet/tar dramatically reduces object count but requires
     downstream reader support (or an expansion step).
3. **Training at full scale is not currently feasible** with the existing
   dataset loader that loads ROI text into memory during init. Cloud ROI is a
   prerequisite, not the end state.

## Provider Options (Equivalent Building Blocks)

AWS (recommended first):
- Object storage: S3
- Batch compute: AWS Batch (EC2)
- Shared filesystem (optional): FSx for Lustre (S3 integration) or EFS

GCP:
- Object storage: GCS
- Batch compute: Batch / Cloud Run Jobs / GKE
- Shared filesystem (optional): Filestore (NFS) + GCS

Azure:
- Object storage: Blob Storage
- Batch compute: Azure Batch
- Shared filesystem (optional): Azure Files / NetApp Files

## Data Transfer Plan

### Option A (Simplest): Upload Only Manifest Directories

Goal: upload the audio required by `data/manifests/birdsong_manifest.json` only.

Suggested approach:
1. Generate an upload plan from the manifest (list of `audio_dir_rel`).
2. For each directory, `sync` that directory to the bucket/prefix.

Notes:
- 3,145 directory sync operations is a lot; implement an upload driver that
  batches work and runs in parallel (future implementation bead).
- Uploading entire dataset root is simpler operationally but larger (~2.9 TiB).

### Option B (Operationally Easy): Upload the Whole Dataset Root

Pros:
- One-time bulk upload; fewer edge cases.
Cons:
- Upload ~2.9 TiB instead of ~1.8 TiB.

### Rough Upload Time

Upload time is bandwidth-limited; real throughput depends heavily on your ISP:
- 100 Mbps upstream (12.5 MB/s): ~1.8 TiB takes ~41 hours (best case)
- 1 Gbps upstream (125 MB/s): ~1.8 TiB takes ~4 hours (best case)

Plan on multi-day wall clock if using consumer-grade upstream.

## Compute Plan (Sharding + Idempotency)

### Unit of Work

Shard by **manifest entry** (audio directory), not by wav file.
- Pros: aligns with manifest schema; natural output bundling; easier retries.
- Count: 3,145 independent units (fits well in an array job).

### Orchestration

Use an array job (AWS Batch job array) where:
- `ARRAY_INDEX` maps to a deterministic subset of entries.
- Each job runs:
  - `python scripts/run_birdsong_roi.py --manifest ... --num-shards N --shard-index i ...`
  - `--jobs` controls per-node parallelism (joblib).

Retries:
- Prefer retry at the directory granularity (rerun shard for failed dirs).
- Ensure each directory output is written atomically (write temp then rename).

Idempotency:
- Continue to rely on `--skip-existing` where applicable.
- For bundled outputs, treat “bundle exists” as complete and skip (recommended).

## Output Format Recommendation

### Recommended: One Parquet Per Audio Directory

Write a single file per `audio_dir_rel`, e.g.:
- `s3://<bucket>/roi/parquet/<audio_dir_rel>.parquet`

Schema suggestion:
- `clip_stem` (string)
- `onset_sec` (float32/float64)
- `offset_sec` (float32/float64)
- Optional metadata columns: `bird_id_norm`, `regime`, `dph`, `session_label`

Benefits:
- ~3,145 objects instead of ~3.8M objects
- Efficient to download/scan for coverage
- Natural fit for downstream scalable dataloaders

### Compatible Stopgap: Per-Directory tar.gz of ROI `.txt`

If you need to preserve the current `.txt` layout without changing training
immediately:
- On worker: write per-wav `.txt` locally to a temp directory.
- Create `roi_<audio_dir_rel_sanitized>.tar.gz`.
- Upload one tarball per directory.

This keeps S3 object count low while deferring downstream code changes.

## Cost Drivers (Rough, Provider-Specific Numbers Vary)

Storage (ballpark, S3 Standard in typical US regions):
- Audio: ~1.8 TiB stored is on the order of tens of USD per month.
- ROI outputs: likely far smaller than audio (but depends on format).

Compute:
- Dominated by CPU-hours required for segmentation.
- Use Spot where possible; expect interruptions, so make work retryable.

Requests/IO:
- Millions of tiny objects are painful operationally even if request-cost is low.
- Bundling outputs is recommended primarily for performance and tractability.

Egress:
- Downloading audio/ROI out of the cloud (e.g., to train locally) can add
  meaningful cost and time; prefer to keep compute close to storage (same
  region) and only egress derived artifacts when possible.

## Immediate Next Steps (What To Do Next)

1. Decide provider (default: AWS) and region.
2. Decide output format (default: per-directory Parquet).
3. Run a micro-benchmark locally:
   - Pick 1-3 representative audio directories.
   - Measure wall time per directory with your ROI config.
   - Use `scripts/run_birdsong_roi.py --summary-out ...` to capture a comparable
     `dirs_per_sec` metric for extrapolation.
   - Extrapolate total CPU-hours.
4. Implement an “upload driver” that stages the manifest subset to object storage.
5. Implement cloud job packaging:
   - container image with this repo + dependencies
   - batch job definition + array submission

## Notes on Manifest Root Overrides

Scripts now support `--audio-root` and `--roi-root` overrides even when the
manifest stores absolute paths (see bead `autoencoded-vocal-analysis-4cd`).
This is required for cloud runs where the mounted paths differ from local ones.
