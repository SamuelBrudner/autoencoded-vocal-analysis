# Parallel ROI Generation Runner (Shardable) + Smoke Test

## Bead
- ID: `autoencoded-vocal-analysis-l3w.1`
- Title: `Parallel ROI generation at scale`
- Date (UTC): `2026-02-12`

## Objective
Harden `scripts/run_birdsong_roi.py` for full-dataset ROI generation with:
- deterministic sharding (`--num-shards/--shard-index`) for job-array style execution
- resume support via `--skip-existing`
- retry support via `--max-retries`
- basic throughput reporting + optional JSON summary output

## Implementation Notes
Changes landed in `scripts/run_birdsong_roi.py`:
- Added sharding flags: `--num-shards`, `--shard-index`
  - Sharding is applied **before** `--skip-existing` filtering so shards remain stable.
- Added `--max-retries` and per-directory error capture.
- Added `--summary-out` to write a machine-readable run summary (including errors).
- Updated resume logic to ignore AppleDouble sidecars (`._*`) when counting `.wav` and `.txt` files (important on exFAT-like volumes that emit `._` entries).

## Inputs
- Manifest: `data/manifests/birdsong_manifest.json` (from `autoencoded-vocal-analysis-l3w.2`)
- Segment config: `examples/configs/birdsong_roi_medium_pilot.yaml`
- Audio root: `/Volumes/samsung_ssd/data/birdsong`
- ROI root (from manifest): `/Volumes/samsung_ssd/data/ava_roi_birdsong_full`

## RUN_ID
- RUN_ID: `20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1`
- Artifacts: `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/`

## Commands
1) Dry-run (plan two leaf dirs from the manifest).
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi_medium_pilot.yaml \
  --manifest data/manifests/birdsong_manifest.json \
  --split train --max-dirs 2 \
  --num-shards 1 --shard-index 0 \
  --skip-existing \
  --dry-run
```

2) Execute (2 dirs), write JSON summary.
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi_medium_pilot.yaml \
  --manifest data/manifests/birdsong_manifest.json \
  --split train --max-dirs 2 \
  --num-shards 1 --shard-index 0 \
  --skip-existing \
  --jobs 2 \
  --summary-out docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/roi_run_summary.json
```

3) Resume check (should skip completed leaf dirs).
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi_medium_pilot.yaml \
  --manifest data/manifests/birdsong_manifest.json \
  --split train --max-dirs 2 \
  --num-shards 1 --shard-index 0 \
  --skip-existing \
  --dry-run
```

## Results
Dry-run planned 2 directories:
- `/Volumes/samsung_ssd/data/birdsong/day43 Bells/pk244/36`
- `/Volumes/samsung_ssd/data/birdsong/day43 Bells/pk244/37`

Execution completed:
- `total=2 ok=2 failed=0`
- summary JSON written (see artifacts)

ROI file presence (excluding AppleDouble `._*`):
- `day43 Bells/pk244/36`: `212 wav` -> `212 ROI .txt`
- `day43 Bells/pk244/37`: `508 wav` -> `508 ROI .txt`

Resume dry-run after completion:
- `Planned ROI directories: 0 (skipped_existing=2 ...)`

## Artifacts
- Dry-run stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/dry_run_stdout.log`
- Run stdout: `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/roi_run_stdout.log`
- Run summary: `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/roi_run_summary.json`
- Resume dry-run (post AppleDouble-safe counting): `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.1/20260212-172741-roi-runner-smoke-autoencoded-vocal-analysis-l3w.1/resume_dry_run_stdout_after_fix.log`

## Example Job-Array Invocation
For 8 shards over the full manifest:
```bash
python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi_medium_pilot.yaml \
  --manifest data/manifests/birdsong_manifest.json \
  --split all \
  --num-shards 8 --shard-index 0 \
  --skip-existing --jobs 4 \
  --max-retries 1 \
  --summary-out /tmp/roi_shard0_summary.json
```

