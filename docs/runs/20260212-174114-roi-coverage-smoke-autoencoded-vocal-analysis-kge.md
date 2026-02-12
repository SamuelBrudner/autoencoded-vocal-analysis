# ROI Coverage Report (Per Dir + Aggregates) + Smoke Test

## Bead
- ID: `autoencoded-vocal-analysis-kge`
- Title: `ROI coverage report (per bird/regime)`
- Date (UTC): `2026-02-12`

## Objective
Add a post-ROI-generation reporting step that summarizes ROI coverage:
- per leaf audio directory: missing/empty ROI files, segment counts, segment duration stats
- aggregated by `bird_id_norm` / `regime` / `dph` / `session_label` (plus split)

Outputs include machine-readable artifacts (CSV/Parquet/JSON) plus this short markdown summary.

## Implementation
New script:
- `scripts/report_birdsong_roi_coverage.py`

Key features:
- Manifest-driven (`docs/birdsong_manifest.md`)
- Deterministic sharding: `--num-shards/--shard-index`
- Optional filtering: `--audio-dir-rel-prefix` (useful for targeted validation)
- Parallel directory processing: `--jobs`
- Outputs to `--out-dir`:
  - `per_directory.{json,csv,parquet}`
  - `by_bird.{csv,parquet}`
  - `summary.json`
- Exit codes:
  - exits non-zero when missing ROI dirs/files or ROI parse errors are detected
  - `--fail-on-empty` optionally treats empty ROI files as an error

## Smoke Test Inputs
- Manifest: `data/manifests/birdsong_manifest.json`
- Split: `train`
- Filter: `--audio-dir-rel-prefix "day43 Bells/pk244/"`
- Max dirs: `3` (two dirs have ROIs from prior smoke; one does not)
- Max files per dir: `20` (to keep the smoke run fast)

## Command
```bash
conda run --no-capture-output -n ava python scripts/report_birdsong_roi_coverage.py \
  --manifest data/manifests/birdsong_manifest.json \
  --split train \
  --audio-dir-rel-prefix "day43 Bells/pk244/" \
  --max-dirs 3 \
  --max-files-per-dir 20 \
  --jobs 2 \
  --out-dir docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge
```

## Results (Smoke)
From `summary.json`:
- `directories=3`
- `missing_roi_dirs=1`
- `missing_roi_files=20` (bounded by `--max-files-per-dir 20`)
- `empty_roi_files=0`
- `roi_parse_errors=0`
- `segments_total=461`
- `segment_duration_total_sec=47.28173`

Note: this smoke run intentionally includes one directory without ROI output, so the script exits non-zero (expected) while still producing artifacts.

## Artifacts
- Directory: `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/`
- Stdout log: `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/roi_coverage_stdout.log`
- Summary: `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/summary.json`
- Per-directory outputs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/per_directory.json`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/per_directory.csv`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/per_directory.parquet`
- Aggregates:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/by_bird.csv`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-kge/20260212-174114-roi-coverage-smoke-autoencoded-vocal-analysis-kge/by_bird.parquet`

