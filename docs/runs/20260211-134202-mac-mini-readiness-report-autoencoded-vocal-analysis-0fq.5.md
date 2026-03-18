# Mac mini Readiness Report

## Bead
- ID: `autoencoded-vocal-analysis-0fq.5`
- Title: `Mac mini readiness report`
- Date (UTC): `2026-02-11`
- RUN_ID: `20260211-134202-mac-mini-readiness-report-autoencoded-vocal-analysis-0fq.5`

## Scope
This report consolidates evidence from beads `autoencoded-vocal-analysis-0fq.1` through `autoencoded-vocal-analysis-0fq.4` and evaluates readiness to proceed to medium pilot bead `autoencoded-vocal-analysis-381`.

Source run reports:
- `docs/runs/20260211-125951-runtime-sanity-autoencoded-vocal-analysis-0fq.1.md`
- `docs/runs/20260211-181348-tiny-manifest-roi-smoke-autoencoded-vocal-analysis-0fq.2.md`
- `docs/runs/20260211-183025-short-training-smoke-autoencoded-vocal-analysis-0fq.3.md`
- `docs/runs/20260211-183822-latent-export-smoke-autoencoded-vocal-analysis-0fq.4.md`

## Executive Recommendation
- **Recommendation:** `NO-GO` for immediate medium pilot execution as-is.
- **Gate to `GO`:** close the prerequisite follow-up beads created in this session:
  - `autoencoded-vocal-analysis-381.1`
  - `autoencoded-vocal-analysis-381.2`
  - `autoencoded-vocal-analysis-381.3`
- **Rationale:** end-to-end smoke path is functional on CPU and export schema checks pass, but reproducibility and robustness gaps remain (ROI threshold sensitivity, environment parity for parquet, and missing preflight for window/ROI compatibility).

## Stage-by-Stage Pass/Fail
| Stage | Bead | Checks | Status |
|---|---|---|---|
| Runtime sanity | `0fq.1` | Torch/NumPy roundtrip, device probe, AVA imports, script `--help` probes | `PASS` |
| Tiny manifest + ROI | `0fq.2` | Manifest creation, ROI generation, ROI completeness validator | `PASS (with mitigation)` |
| Short training smoke | `0fq.3` | 2-epoch fixed-window train, artifact emission, finite metrics/collapse checks | `PASS (after config fix)` |
| Latent export smoke | `0fq.4` | Dry-run planning, no-energy export + skip/resume, energy export, NPZ+JSON schema/finite/invariant checks | `PASS (after code fixes)` |

Detailed stage notes:
- `0fq.1` runtime compatibility checks all passed with `torch==2.2.2`, `numpy==1.26.4`; MPS was built but unavailable, so runtime device was CPU.
- `0fq.2` default segmentation thresholds produced empty ROI files on tiny subset; smoke used a lenient config and then passed validator (`missing_roi_files=0`, `empty_roi_files=0`).
- `0fq.3` first train attempt failed due `window_length=0.03` exceeding observed ROI durations (~`0.01161s`); after setting `window_length=0.01`, training completed for 2 epochs with decreasing finite loss (`39276.9453125 -> 18967.92578125`) and no immediate collapse.
- `0fq.4` latent export passed after fixing latent grid shape handling and adding optimizer-skip loading for inference checkpoint load; validation reported `ok=true` for no-energy and energy exports.

## Executed Commands Across Epic (0fq.1 -> 0fq.4)

### `autoencoded-vocal-analysis-0fq.1` (runtime sanity)
The run report captures executed checks but does not include a literal shell command transcript. Documented probes executed:
- Torch/NumPy compatibility roundtrip probe.
- Device availability probe (`mps` built/available, `cuda` available).
- Core AVA import sweep.
- Script entrypoint `--help` checks for:
  - `scripts/build_birdsong_manifest.py`
  - `scripts/build_birdsong_metadata.py`
  - `scripts/evaluate_latent_metrics.py`
  - `scripts/export_latent_sequences.py`
  - `scripts/inspect_birdsong.py`
  - `scripts/run_birdsong_roi.py`
  - `scripts/run_birdsong_validation.py`
  - `scripts/validate_birdsong_rois.py`

### `autoencoded-vocal-analysis-0fq.2` (tiny manifest + ROI smoke)
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

```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_roi.py \
  --segment-config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/segment_config_lenient.yaml \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --jobs 1
```

```bash
conda run --no-capture-output -n ava python scripts/validate_birdsong_rois.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --output docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/roi_validation_summary.json
```

### `autoencoded-vocal-analysis-0fq.3` (short training smoke)
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --epochs 2 --batch-size 32 --num-workers 0 --cpu \
  --spec-cache-dir /tmp/ava_spec_cache_autoencoded-vocal-analysis-0fq.3
```

```bash
python - <<'PY'
# computed ROI duration distribution under /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332
PY
```

```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --epochs 2 --batch-size 32 --num-workers 0 --cpu \
  --spec-cache-dir /tmp/ava_spec_cache_autoencoded-vocal-analysis-0fq.3
```

### `autoencoded-vocal-analysis-0fq.4` (latent export smoke)
```bash
conda run --no-capture-output -n ava python scripts/export_latent_sequences.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --checkpoint docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar \
  --out-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --max-dirs 1 --max-files-per-dir 1 --max-clips 1 \
  --device cpu --dry-run
```

```bash
MPLCONFIGDIR=/tmp/mpl_ava conda run --no-capture-output -n ava python scripts/export_latent_sequences.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --checkpoint docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar \
  --out-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --max-dirs 1 --max-files-per-dir 1 --max-clips 1 \
  --device cpu --report-every 1 \
  --summary-out docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_first.json
```

```bash
MPLCONFIGDIR=/tmp/mpl_ava conda run --no-capture-output -n ava python scripts/export_latent_sequences.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --checkpoint docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar \
  --out-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --max-dirs 1 --max-files-per-dir 1 --max-clips 1 \
  --device cpu --report-every 1 \
  --summary-out docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_second.json
```

```bash
MPLCONFIGDIR=/tmp/mpl_ava conda run --no-capture-output -n ava python scripts/export_latent_sequences.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --split train \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --checkpoint docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar \
  --out-dir docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_with_energy \
  --audio-root /Volumes/samsung_ssd/data/birdsong \
  --roi-root /tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/ \
  --max-dirs 1 --max-files-per-dir 1 --max-clips 1 \
  --device cpu --export-energy --report-every 1 \
  --summary-out docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_with_energy.json
```

```bash
conda run --no-capture-output -n ava pytest -q tests/models/test_latent_sequence_export.py
```

```bash
conda run --no-capture-output -n ava pytest -q tests/models/test_lightning_training.py::test_lightning_checkpoint_loads_legacy
```

## Artifacts Produced (Paths)

### Run reports
- `docs/runs/20260211-125951-runtime-sanity-autoencoded-vocal-analysis-0fq.1.md`
- `docs/runs/20260211-181348-tiny-manifest-roi-smoke-autoencoded-vocal-analysis-0fq.2.md`
- `docs/runs/20260211-183025-short-training-smoke-autoencoded-vocal-analysis-0fq.3.md`
- `docs/runs/20260211-183822-latent-export-smoke-autoencoded-vocal-analysis-0fq.4.md`

### `autoencoded-vocal-analysis-0fq.2` artifacts
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/roi_validation_summary.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/segment_config_lenient.yaml`
- `/tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/` (generated ROI tree root)

### `autoencoded-vocal-analysis-0fq.3` artifacts
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_checks.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/lightning_logs/version_0/events.out.tfevents.1770834522.home-mini.local.40697.0`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/lightning_logs/version_0/hparams.yaml`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/run_metadata.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_stdout.log`
- `/tmp/ava_spec_cache_autoencoded-vocal-analysis-0fq.3` (spectrogram cache directory)

### `autoencoded-vocal-analysis-0fq.4` artifacts
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/export_validation_summary.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.npz`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_with_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_with_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.npz`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_no_energy_first.log`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_no_energy_second.log`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_with_energy.log`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_first.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_second.json`
- `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_with_energy.json`

## Key Findings Incorporated
1. Runtime environment:
- `torch==2.2.2`, `numpy==1.26.4`.
- CPU-only execution on this machine (`mps` built but unavailable).
- Numba cache compatibility fix applied in `src/ava/__init__.py` via writable temp fallback for `NUMBA_CACHE_DIR`.

2. ROI smoke behavior:
- Default segmentation thresholds produced empty ROIs on tiny subset.
- Lenient segmentation config was required for smoke gate pass.

3. Training smoke behavior:
- 2 epochs completed.
- Loss decreased (`39276.9453125 -> 18967.92578125`) and remained finite.
- Collapse heuristic did not trigger.
- `window_length` adjusted from `0.03` to `0.01` due ROI duration mismatch.

4. Latent export smoke behavior:
- Fixed latent grid shape bug in `src/ava/models/latent_sequence.py`.
- Added optimizer-restore skip path in `src/ava/models/vae.py` and used it in `src/ava/models/latent_metrics.py`.
- Exported NPZ+JSON outputs validated (`schema_version=ava_latent_sequence_v1`, finite arrays, shape invariants satisfied).

5. Code fixes across epic included:
- `src/ava/__init__.py`
- `src/ava/models/latent_sequence.py`
- `src/ava/models/vae.py`
- `src/ava/models/latent_metrics.py`
- `scripts/build_birdsong_metadata.py`
- `scripts/evaluate_latent_metrics.py`
- `scripts/run_birdsong_roi.py`
- `scripts/run_birdsong_validation.py`

## Known Limitations and Issues Found
- Medium pilot not yet proven under default ROI segmentation settings.
- Manifest creation currently requires tooling outside `ava` env when parquet engine is missing, reducing reproducibility.
- Training pipeline lacked an upfront guard for incompatible `window_length` versus ROI duration distributions.
- Runtime check report (`0fq.1`) does not provide full literal command transcript, limiting strict replay auditability.
- Machine remains CPU-only for this stack (`mps` unavailable), so pilot runtime/performance may be slower than expected.
- Tiny smoke runs were train-only (`test split` empty), so no validation-loop behavior was exercised in this epic.

## Follow-up Beads/Issues (Created)
These were created as concrete prerequisites under `autoencoded-vocal-analysis-381`:
- `autoencoded-vocal-analysis-381.1`: Add preflight check for `window_length` vs ROI durations.
- `autoencoded-vocal-analysis-381.2`: Calibrate ROI segmentation thresholds for medium pilot.
- `autoencoded-vocal-analysis-381.3`: Enable parquet support in `ava` env for manifest workflows.

Related existing issue worth including in pilot readiness tracking:
- `autoencoded-vocal-analysis-kge` (open): ROI coverage report (per bird/regime).

## Final Go/No-Go Statement for Bead `381`
- `NO-GO` to launch `autoencoded-vocal-analysis-381` immediately without prerequisites.
- `GO` once `381.1`, `381.2`, and `381.3` are closed and their acceptance criteria are met, with `kge` integrated into pilot monitoring for ROI coverage visibility.
