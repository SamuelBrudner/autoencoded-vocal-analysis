# Full Manifest + Split Exports

## Bead
- ID: `autoencoded-vocal-analysis-l3w.2`
- Title: `Full manifest + split exports`
- Date (UTC): `2026-02-12`

## Objective
Generate a full birdsong manifest (train/test split stratified by bird within each regime) and export the split lists for downstream ROI generation, training, and latent export workflows.

## Inputs
- Metadata parquet: `data/metadata/birdsong/birdsong_files.parquet`
- Audio root: `/Volumes/samsung_ssd/data/birdsong`
- ROI root (planned for full runs): `/Volumes/samsung_ssd/data/ava_roi_birdsong_full`
- Seed: `0`
- Train fraction: `0.8`

## RUN_ID
- RUN_ID: `20260212-171952-full-manifest-split-exports-autoencoded-vocal-analysis-l3w.2`
- Artifacts: `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.2/20260212-171952-full-manifest-split-exports-autoencoded-vocal-analysis-l3w.2/`

## Commands
1) Build full manifest.
```bash
conda run --no-capture-output -n ava python scripts/build_birdsong_manifest.py \
  --metadata data/metadata/birdsong/birdsong_files.parquet \
  --root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /Volumes/samsung_ssd/data/ava_roi_birdsong_full \
  --out data/manifests/birdsong_manifest.json \
  --seed 0 --train-fraction 0.8 \
  > docs/runs/artifacts/autoencoded-vocal-analysis-l3w.2/20260212-171952-full-manifest-split-exports-autoencoded-vocal-analysis-l3w.2/build_manifest_stdout.log 2>&1
```

2) Validate manifest invariants and export split lists (birds + audio_dir_rel).
```bash
# See: docs/runs/artifacts/.../manifest_validation.json
```

## Results
Full manifest written:
- `data/manifests/birdsong_manifest.json`
- sha256: `b8cb42d8dc7ea50bea2035ebbd0a615e0e40fee8d9c2956e89fb475f4e6cc40d`

Builder summary (from `build_manifest_stdout.log`):
- Train:
  - birds: `43`
  - directories: `2624`
  - files: `3190613`
  - by regime:
    - `bells`: `15 birds`, `1082 dirs`, `1221165 files`
    - `isolates`: `10 birds`, `395 dirs`, `585769 files`
    - `samba`: `13 birds`, `895 dirs`, `1017857 files`
    - `simple`: `5 birds`, `252 dirs`, `365822 files`
- Test:
  - birds: `10`
  - directories: `521`
  - files: `631163`
  - by regime:
    - `bells`: `4 birds`, `261 dirs`, `240684 files`
    - `isolates`: `2 birds`, `55 dirs`, `67684 files`
    - `samba`: `3 birds`, `153 dirs`, `241583 files`
    - `simple`: `1 bird`, `52 dirs`, `81212 files`

Validation (see `manifest_validation.json`):
- Required top-level keys present: `PASS`
- Required per-entry fields present: `PASS`
- Split label consistency (`split` field matches list): `PASS`
- No bird overlap across splits: `PASS`
- No audio_dir_rel overlap across splits: `PASS`

## Split Exports
These are intended as convenient inputs for downstream parallelization/sharding:
- Train birds: `data/manifests/birdsong_manifest_train_birds.txt`
- Test birds: `data/manifests/birdsong_manifest_test_birds.txt`
- Train audio_dir_rel list: `data/manifests/birdsong_manifest_train_audio_dir_rels.txt`
- Test audio_dir_rel list: `data/manifests/birdsong_manifest_test_audio_dir_rels.txt`

## Artifacts
- Build stdout + printed summary:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.2/20260212-171952-full-manifest-split-exports-autoencoded-vocal-analysis-l3w.2/build_manifest_stdout.log`
- Validation output:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-l3w.2/20260212-171952-full-manifest-split-exports-autoencoded-vocal-analysis-l3w.2/manifest_validation.json`

## Follow-ups
- Proceed to `autoencoded-vocal-analysis-l3w.1` (parallel ROI generation at scale) using:
  - manifest: `data/manifests/birdsong_manifest.json`
  - segment config: `examples/configs/birdsong_roi_medium_pilot.yaml` (or a full-dataset tuned successor)

