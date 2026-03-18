# Window-Length vs ROI Duration Preflight

## Bead
- ID: `autoencoded-vocal-analysis-381.1`
- Title: `Add preflight check for window_length vs ROI durations`
- Date (UTC): `2026-02-12`

## Objective
Add a pre-training guard that computes ROI duration stats and fails fast when configured `window_length` is incompatible with available ROI segments.

## Implementation
- Added module: `src/ava/models/roi_preflight.py`
  - Computes ROI duration statistics from ROI files.
  - Validates `window_length` compatibility.
  - Raises actionable error when no ROI segment is long enough.
- Integrated in CLI: `scripts/run_birdsong_validation.py`
  - Runs preflight before data loader/training setup.
  - Prints JSON summary of ROI duration stats.

## Automated Tests
```bash
pytest -q tests/models/test_roi_preflight.py tests/data/test_validate_birdsong_rois.py
```
Result: `5 passed`.

## Runtime Verification
Failure path (expected incompatible window):
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config examples/configs/fixed_window_finch_30ms_44k.yaml \
  --save-dir /tmp/ava_preflight_check_3811 \
  --max-files 8 --num-workers 0 --cpu
```
Observed error:
- `window_length=0.03000s`, `max_duration=0.01161s`, `roi_segments=917`

Pass path (compatible window):
```bash
conda run --no-capture-output -n ava python scripts/run_birdsong_validation.py \
  --manifest docs/runs/artifacts/autoencoded-vocal-analysis-0fq.2/manifest.json \
  --config docs/runs/artifacts/autoencoded-vocal-analysis-0fq.3/fixed_window_smoke_config.yaml \
  --save-dir /tmp/ava_preflight_check_3811_pass \
  --max-files 8 --num-workers 0 --batch-size 4 --epochs 1 --cpu --disable-spec-cache
```
Observed:
- Preflight summary emitted with `compatible_fraction=1.0`.
- Training completed successfully.

## Documentation Updates
- Added recommended defaults and preflight guidance to:
  - `docs/birdsong_training_plan.md`
