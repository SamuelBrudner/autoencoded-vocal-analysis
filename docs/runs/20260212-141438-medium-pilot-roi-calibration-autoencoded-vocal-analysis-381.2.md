# Medium Pilot ROI Threshold Calibration

## Bead
- ID: `autoencoded-vocal-analysis-381.2`
- Title: `Calibrate ROI segmentation thresholds for medium pilot`
- Date (UTC): `2026-02-12`

## Objective
Compare default versus tuned segmentation thresholds on a representative medium cohort (10-20 birds) and select a config suitable for medium pilot ROI generation.

## Cohort + Manifest
- Metadata parquet: `data/metadata/birdsong/birdsong_files.parquet`
- Audio root: `/Volumes/samsung_ssd/data/birdsong`
- Regimes: `bells`, `isolates`, `samba`, `simple`
- Selection: `--birds-per-regime 3 --max-dirs-per-bird 1 --seed 3812`
- Resulting cohort size: `12 birds`, `12 directories`, `11,653 wav files`

Manifest artifacts:
- Default-root manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium.json`
- Tuned-root manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json`

## Commands
1) Build manifests (same cohort/seed, different ROI roots).
```bash
conda run --no-capture-output -n ava python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_medium_autoencoded-vocal-analysis-381.2-default \
  --out docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium.json \
  --seed 3812 --birds-per-regime 3 --max-dirs-per-bird 1 \
  --regimes bells isolates samba simple

conda run --no-capture-output -n ava python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_medium_autoencoded-vocal-analysis-381.2-tuned \
  --out docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --seed 3812 --birds-per-regime 3 --max-dirs-per-bird 1 \
  --regimes bells isolates samba simple
```

2) Run ROI generation + validation (default config).
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config examples/configs/birdsong_roi.yaml \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium.json \
  --split all --jobs 4

conda run --no-capture-output -n ava python scripts/validate_birdsong_rois.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium.json \
  --split all \
  --output docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_default.json
```

3) Run ROI generation + validation (tuned candidate).
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config docs/runs/artifacts/autoencoded-vocal-analysis-381.2/segment_config_candidate.yaml \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --split all --jobs 4

conda run --no-capture-output -n ava python scripts/validate_birdsong_rois.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json \
  --split all \
  --output docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_candidate.json
```

## Results
Validation summary:
- Default (`examples/configs/birdsong_roi.yaml`):
  - `missing_roi_files=0`
  - `empty_roi_files=11580` / `11653` (`99.37%` empty)
  - Source: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_default.json`
- Tuned candidate:
  - `missing_roi_files=0`
  - `empty_roi_files=71` / `11653` (`0.61%` empty)
  - Source: `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_candidate.json`

## Selected Medium-Pilot Config
Committed config:
- `examples/configs/birdsong_roi_medium_pilot.yaml`

Parameter deltas versus default:
- Lower frequency floor: `min_freq 400 -> 300`
- Broader normalized spectrogram window: `spec_min_val 2.0 -> 1.0` (same `spec_max_val 6.5`)
- Lower amplitude thresholds: `th_1/th_2/th_3 = 0.05/0.10/0.15`
- Shorter minimum duration: `min_dur 0.005`
- Less smoothing: `smoothing_timescale 0.004`

## Tradeoffs
- The tuned config dramatically improves ROI coverage and satisfies the medium-pilot gate (`missing_roi_files=0`, `empty_roi_files` near zero).
- It also increases segment density substantially, so downstream quality checks should monitor for false positives and consider per-regime refinements if needed.

## Artifacts
- `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/manifest_medium_tuned.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_default.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/roi_validation_candidate.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-381.2/segment_config_candidate.yaml`
