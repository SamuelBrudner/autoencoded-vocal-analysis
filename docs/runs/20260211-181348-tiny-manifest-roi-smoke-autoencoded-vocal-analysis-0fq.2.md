# Tiny Manifest + ROI Smoke Run

## Bead
- ID: `autoencoded-vocal-analysis-0fq.2`
- Title: `Tiny manifest + ROI smoke`
- Date (UTC): `2026-02-11`

## Objective
Build a tiny stratified birdsong manifest, run ROI generation, and validate ROI completeness on the selected subset.

## Environment + Inputs
- Requested audio root: `/Volumes/Extreme_SSD/birdsong_data` (not present on this machine).
- Actual audio root used: `/Volumes/samsung_ssd/data/birdsong` (matches metadata provenance).
- Metadata parquet: `data/metadata/birdsong/birdsong_files.parquet`
- Conda env for ROI/validation: `ava`

## Commands Run
1. Build manifest (deterministic tiny stratified sample).
```bash
python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332 \
  --out docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --seed 332 \
  --birds-per-regime 1 \
  --max-dirs-per-bird 1 \
  --regimes bells isolates samba simple
```

2. Run ROI generation on manifest subset.
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/segment_config_lenient.yaml \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --jobs 1
```

3. Validate ROIs.
```bash
conda run --no-capture-output -n ava python scripts/validate_birdsong_rois.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --output docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/roi_validation_summary.json
```

## Result Summary
- Manifest split: `train-only` (4 birds, 4 directories, 455 WAV files)
- Regime coverage: `bells`, `isolates`, `samba`, `simple`
- ROI validation outcome:
  - `directories_checked`: `4`
  - `wav_files_checked`: `455`
  - `missing_roi_dirs`: `0`
  - `missing_roi_files`: `0`
  - `empty_roi_files`: `0`

## Artifacts
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json`
- ROI validator summary: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/roi_validation_summary.json`
- ROI segment config used: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/segment_config_lenient.yaml`

## Notes
- `ava` could not read parquet (`pyarrow`/`fastparquet` unavailable in read-only env), so manifest creation was executed with system Python (which has `pyarrow`), while ROI generation and ROI validation were run in `ava`.
- Default ROI config produced many empty files on this tiny sample; a lenient smoke config was used to satisfy the no-missing/no-empty validation gate.
