# Latent Sequence Export Format (AVA)

This document defines the **canonical output schema** for exporting time-indexed
latent sequences from AVA models (e.g., the fixed-window VAE) for downstream
temporal pipelines (e.g., distance-image + persistence workflows in
`juvenile_learning_tda`).

The goal is to represent a long recording as a **sequence of posterior
statistics** over time, not as a bag of windows.

See also:
- `docs/temporal_topology_handoff.md` for an end-to-end export workflow and the
  handoff into `juvenile_learning_tda` distance-image/persistence code.

## Versioning

Schema version string: `ava_latent_sequence_v1`

Any breaking change must bump the version.

## On-Disk Representation (Canonical)

One clip export is represented by **two files**:

1. `<clip_id>.npz` (numeric arrays; fast load)
2. `<clip_id>.json` (metadata/provenance; human-readable)

Rationale:
- `.npz` is a simple, dependency-light container for dense tensors.
- JSON is better for strings, nested config/provenance, and forward-compat.

## Required Arrays (`.npz`)

All arrays are **time-major** (`T` first).

- `start_times_sec`: float64, shape `[T]`
  - Window start times in seconds relative to the start of the audio clip
    (t=0).
  - Must be strictly increasing.

- `window_length_sec`: float64, shape `[]` (scalar)
  - Window duration used when encoding.

- `hop_length_sec`: float64, shape `[]` (scalar)
  - Step between consecutive window starts.

- `mu`: float32, shape `[T, z_dim]`
  - Posterior mean for each window.

- `logvar`: float32, shape `[T, z_dim]`
  - Posterior log-variance for each window, i.e. `log(sigma**2)`.
  - Convert to `sigma` via `sigma = exp(0.5 * logvar)`.

## Optional Arrays (`.npz`)

These are recommended when available.

- `energy`: float32, shape `[T]`
  - Per-window energy proxy (e.g., RMS of raw audio, or summed magnitude of the
    input spectrogram window).
  - Intended for silence gating downstream.

- `gating_weight`: float32, shape `[T]`
  - A precomputed `[0, 1]` weight suitable for silence gating.
  - If present, downstream code should prefer this over recomputing a weight
    from `energy`.

## Required Metadata (`.json`)

The JSON file must include:

- `schema_version`: string (must be `ava_latent_sequence_v1`)
- `created_utc`: string (ISO-8601 UTC timestamp)
- `clip_id`: string
- `audio_path`: string (as provided to the exporter; can be relative)
- `audio_sha256`: string or null
- `sample_rate_hz`: integer or null

## Recommended Metadata (`.json`)

These fields make exports easier to reproduce and audit:

- `roi_path`: string or null
  - ROI file used to restrict which parts of the audio were encoded (if any).

- `model_checkpoint_path`: string or null
- `git_commit`: string or null
- `config_path`: string or null
- `manifest_path`: string or null

- `model`: object
  - Suggested keys:
    - `z_dim`
    - `input_shape` (`[freq_bins, time_bins]`)
    - `posterior_type` (`diag`/`lowrank`)
    - `conv_arch` (`residual`/`plain`)
    - `decoder_type` (`upsample`/`convtranspose`)
    - `learn_observation_scale` (bool)
    - `log_precision` (float; if present in checkpoint)

- `preprocess`: object
  - A serialized copy of the preprocessing parameters used to construct the
    model input windows (e.g., mel vs linear, min/max freq, clipping ranges).

## Invariants / Validation Rules

An export is valid if:
- `mu.shape == logvar.shape == (T, z_dim)`
- `start_times_sec.shape == (T,)`
- `T >= 1`
- All numeric arrays are finite (`no NaN/Inf`)
- `start_times_sec` is strictly increasing

## Example Loader

```python
import json
import numpy as np

npz = np.load("clip_0001.npz")
with open("clip_0001.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

mu = npz["mu"]          # [T, z_dim]
logvar = npz["logvar"]  # [T, z_dim]
t = npz["start_times_sec"]  # [T]
sigma = np.exp(0.5 * logvar)
```
