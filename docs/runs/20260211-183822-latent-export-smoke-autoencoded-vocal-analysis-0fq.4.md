# Latent Export Smoke on Trained Checkpoint

## Bead
- ID: `autoencoded-vocal-analysis-0fq.4`
- Title: `Latent export smoke on trained checkpoint`
- Date (UTC): `2026-02-11`

## Objective
Run a manifest-driven latent export on a tiny deterministic split using the trained smoke checkpoint, validate NPZ+JSON schema and finite values, verify skip/resume behavior, and check optional energy export.

## Inputs + Runtime
- Manifest: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json`
- Checkpoint: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/training_run/checkpoint_001.tar`
- Training config: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml`
- ROI root: `/tmp/ava_roi_smoke_autoencoded-vocal-analysis-0fq.2-s332/`
- Audio root: `/Volumes/samsung_ssd/data/birdsong`
- Runtime env: `conda run --no-capture-output -n ava`
- Device: `cpu`
- Training window length from config: `0.01` sec

## Deterministic Tiny Split
Command:
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
Dry-run result:
- `Planned clips: 1 (split=train shard=0/1)`
- Clip: `day 75 bells/R521/49/521_26113_on_July_30_13_32_35`

## Export Commands Run
1. No-energy export (first run)
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
Result: `total=1 exported=1 skipped=0 failed=0`

2. No-energy export (second run, skip/resume check)
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
Result: `total=1 exported=0 skipped=1 failed=0`

3. Energy export path
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
Result: `total=1 exported=1 skipped=0 failed=0`

## Output Validation
Validation output: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/export_validation_summary.json`

Checks performed:
- NPZ+JSON pair exists per clip in both output roots.
- Required NPZ arrays present: `start_times_sec`, `window_length_sec`, `hop_length_sec`, `mu`, `logvar`.
- Required JSON metadata fields present: `schema_version`, `created_utc`, `clip_id`, `audio_path`, `audio_sha256`, `sample_rate_hz`.
- `schema_version == ava_latent_sequence_v1`.
- Arrays finite (`no NaN/Inf`).
- Invariants: `mu.shape == logvar.shape == (T, z_dim)`, `start_times_sec.shape == (T,)`, `T >= 1`, strictly increasing timestamps.
- Energy run contains `energy` array with shape `(T,)`.

Observed for exported clip:
- `T=99`, `z_dim=32`
- No-energy NPZ keys: `start_times_sec`, `window_length_sec`, `hop_length_sec`, `mu`, `logvar`
- Energy NPZ keys: previous keys + `energy`

Validation verdict: `ok=true` for both runs.

## Implementation Notes
Two code issues surfaced during this smoke and were fixed:
- `src/ava/models/latent_sequence.py`: fixed frequency/time grid construction for interpolation by broadcasting arrays before stacking (previous shape mismatch caused export failure).
- `src/ava/models/vae.py`: added optional `load_optimizer` flag to `load_state()` (default `True`) so inference-only loading can skip optimizer restoration.
- `src/ava/models/latent_metrics.py`: `load_vae_from_checkpoint()` now calls `model.load_state(..., load_optimizer=False)`.

Verification for code changes:
- `conda run --no-capture-output -n ava pytest -q tests/models/test_latent_sequence_export.py` -> `4 passed`
- `conda run --no-capture-output -n ava pytest -q tests/models/test_lightning_training.py::test_lightning_checkpoint_loads_legacy` -> `1 passed`

## Artifacts
- Output root: `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/`
- No-energy outputs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.npz`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_no_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.json`
- Energy outputs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_with_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.npz`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/latent_with_energy/day 75 bells/R521/49/521_26113_on_July_30_13_32_35.json`
- Run summaries:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_first.json`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_no_energy_second.json`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/summary_with_energy.json`
- Validation summary:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/export_validation_summary.json`
- Logs:
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_no_energy_first.log`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_no_energy_second.log`
  - `docs/runs/artifacts/autoencoded-vocal-analysis-0fq.4/logs/export_with_energy.log`
