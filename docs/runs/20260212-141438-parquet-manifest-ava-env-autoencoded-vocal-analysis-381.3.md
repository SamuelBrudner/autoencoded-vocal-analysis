# Parquet Manifest Workflow Validation in `ava` Environment

## Bead
- ID: `autoencoded-vocal-analysis-381.3`
- Title: `Enable parquet support in ava env for manifest workflows`
- Date (UTC): `2026-02-12`

## Objective
Ensure manifest workflows can run end-to-end inside the `ava` conda environment without relying on system Python for parquet support.

## Changes
- Added `pyarrow` to:
  - `environment.dev.yml`
  - `environment.train.yml`
- Updated dependency notes in:
  - `docs/birdsong_manifest.md`
  - `docs/source/install.rst`

## Verification
1) Confirm parquet dependency in `ava`.
```bash
conda run --no-capture-output -n ava python -c "import pyarrow,sys;print(pyarrow.__version__)"
```
Observed: `21.0.0`

2) Run manifest builder in `ava` against project metadata parquet.
```bash
conda run --no-capture-output -n ava python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_medium_autoencoded-vocal-analysis-381.2 \
  --out docs/runs/artifacts/autoencoded-vocal-analysis-381.3/manifest_pyarrow_check.json \
  --seed 3813 --birds-per-regime 1 --max-dirs-per-bird 1 \
  --regimes bells isolates samba simple
```

Result:
- Command succeeded in `ava`.
- Output manifest created at:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-381.3/manifest_pyarrow_check.json`

## Summary
Manifest generation from parquet now has explicit environment support and verified in-env execution path.
