# Runtime Sanity Check (Mac mini)

## Bead
- ID: `autoencoded-vocal-analysis-0fq.1`
- Title: `Runtime sanity check on Mac mini`
- Date: `2026-02-11`

## Environment
- Conda env: `ava`
- Python executable: `/opt/anaconda3/envs/ava/bin/python`
- Python version: `3.10.15`
- Platform: `macOS-26.0.1-arm64-arm-64bit`
- Machine: `arm64`

## Package Versions (key runtime deps)
- `autoencoded-vocal-analysis`: `0.3.1`
- `torch`: `2.2.2`
- `numpy`: `1.26.4`
- `pytorch-lightning`: `2.6.0`
- `scipy`: `1.15.3`
- `numba`: `0.63.1`
- `umap-learn`: `0.5.11`
- `scikit-learn`: `1.7.2`
- `h5py`: `3.15.1`
- `matplotlib`: `3.10.8`
- `affinewarp`: `0.2.0`
- `pyarrow`: not installed (optional for metadata parquet script)

## Check 1: Torch <-> NumPy Compatibility
- Status: `PASS`
- Probe: `numpy -> torch.from_numpy -> torch ops -> tensor.numpy -> torch.tensor`
- Result:
  - roundtrip numerics: `True`
  - dtype/shape stability: `float32`, `[3, 4]`
  - warnings during probe: none

## Check 2: Device Availability
- Status: `PASS`
- `torch.backends.mps.is_built()`: `True`
- `torch.backends.mps.is_available()`: `False`
- `torch.cuda.is_available()`: `False`
- Selected runtime device for probe: `cpu`
- Smoke op (`8x8` matmul on selected device): `PASS`

## Check 3: Core AVA Imports
- Status: `PASS`
- Imported modules:
  - `ava`
  - `ava.models`
  - `ava.preprocessing`
  - `ava.preprocessing.preprocess`
  - `ava.preprocessing.utils`
  - `ava.data`
  - `ava.data.data_container`
  - `ava.segmenting`
  - `ava.segmenting.segment`
  - `ava.plotting`
  - `ava.plotting.grid_plot`
  - `ava.timescale_analysis`

## Check 4: Script Entrypoints (`--help`)
- Status: `PASS`
- Verified scripts:
  - `scripts/build_birdsong_manifest.py`
  - `scripts/build_birdsong_metadata.py`
  - `scripts/evaluate_latent_metrics.py`
  - `scripts/export_latent_sequences.py`
  - `scripts/inspect_birdsong.py`
  - `scripts/run_birdsong_roi.py`
  - `scripts/run_birdsong_validation.py`
  - `scripts/validate_birdsong_rois.py`

## Compatibility Fixes Applied
1. UMAP/numba import compatibility in readonly-style environments.
- Symptom: importing modules that transitively import `umap` failed with:
  - `RuntimeError: cannot cache function 'rdist': no locator available ...`
- Fix: added package-level fallback in `src/ava/__init__.py` to set `NUMBA_CACHE_DIR` to a writable temp directory when unset.

2. Script `src`-layout import compatibility.
- Symptom: some scripts could not resolve `ava` package when run as direct file entrypoints.
- Fix: added `ROOT/SRC_ROOT` `sys.path` bootstrap in:
  - `scripts/evaluate_latent_metrics.py`
  - `scripts/run_birdsong_roi.py`
  - `scripts/run_birdsong_validation.py`

3. Optional dependency import guard for `--help` path.
- Symptom: `scripts/build_birdsong_metadata.py --help` failed when `pyarrow` not installed.
- Fix: made `pyarrow` import lazy/guarded and raise a clear error only when metadata build is executed.

## Notes
- Matplotlib emitted cache warnings in this sandbox because `~/.matplotlib` is not writable here. AVA imports still completed successfully using temporary cache directories.
