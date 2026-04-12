# PK249 33-90 DPH Latent Access Guide

This note is for downstream analysts who want to load the completed
`PK249` `33-90dph` latent export without having to reconstruct the training or
AWS export workflow.

For the canonical latent file schema, see
`docs/latent_sequence_export.md`.

## Dataset Summary

- Bird: `PK249`
- Regime: `bells`
- DPH window: `33-90` inclusive
- Manifest split:
  - `47` train directories / `76,374` wav files
  - `11` test directories / `19,028` wav files
- Export outcome:
  - `95,402` total manifest clips
  - `95,084` exported latent clips
  - `318` skipped clips
  - `0` failed clips
- Latent window length: `0.03` seconds
- Latent stride / hop length: `0.005804988662131519` seconds
  - this is about `5.805 ms`

## Canonical Locations

All canonical artifacts live in `us-east-1` under:

- bucket: `s3://ava-birdsong-us-east-1-a1859d31`
- PK249 prefix:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90`

Core inputs:

- manifest:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/manifest_pk249_33_90.json`
- ROI parquet bundles:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/roi/`
- audio subset:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/audio/`
- training config:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/fixed_window_pk249_33_90.yaml`
- model checkpoint:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/training-runs/pk249-33-90-4gpu-batch-20260409_010209/training_run/checkpoint_050.tar`

Completed latent export:

- export root:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/`
- latent files:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/latent_sequences/`
- export summary:
  - `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/export_summary.json`

## Layout

The export writes one `.npz` and one `.json` per clip under
`latent_sequences/`.

Example:

- `latent_sequences/day43 Bells/pk249/90/9249_38510.4255787037_6_7_10_12_50.npz`
- `latent_sequences/day43 Bells/pk249/90/9249_38510.4255787037_6_7_10_12_50.json`

The relative directory under `latent_sequences/` mirrors the manifest
`audio_dir_rel`.

Practical consequence:

- if you want all adult-day clips, use the `day43 Bells/pk249/90/` subtree
- if you want split or DPH metadata, use either:
  - the `.json` sidecars, which include `entry.split` and `entry.dph`
  - the manifest, keyed by the parent directory such as `day43 Bells/pk249/90`

## What Is In Each File

Each `.npz` contains time-major arrays:

- `mu`: posterior means, shape `[T, 32]`
- `logvar`: posterior log-variances, shape `[T, 32]`
- `start_times_sec`: window start times, shape `[T]`
- `window_length_sec`: scalar
- `hop_length_sec`: scalar
- `energy`: optional per-window energy, shape `[T]`

Each `.json` includes provenance and per-clip metadata, including:

- `clip_id`
- `audio_path`
- `manifest_path`
- `roi_path`
- `num_windows`
- `z_dim`
- `entry.audio_dir_rel`
- `entry.dph`
- `entry.split`

## Expected Counts

The latent export is not exactly one-for-one with the manifest.

- manifest clips: `95,402`
- latent exports present: `95,084`
- skipped: `318`

Those skipped clips were not export failures. They were clips with no usable
windows after ROI/time filtering, so analysts should not treat their absence as
data corruption.

## AWS CLI Examples

Download the export summary:

```bash
aws s3 cp \
  s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/export_summary.json \
  -
```

Download only latent arrays for the full corpus:

```bash
aws s3 sync \
  s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/latent_sequences/ \
  /path/to/pk249_latents \
  --exclude "*" \
  --include "*.npz"
```

Download arrays plus metadata for one DPH:

```bash
aws s3 cp --recursive \
  "s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/latent-exports/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/latent_sequences/day43 Bells/pk249/90/" \
  /path/to/pk249_90dph_latents \
  --exclude "*" \
  --include "*.npz" \
  --include "*.json"
```

Download the manifest:

```bash
aws s3 cp \
  s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/pk249-33-90/manifest_pk249_33_90.json \
  manifest_pk249_33_90.json
```

## Minimal Python Loader

```python
import json
import numpy as np

npz = np.load("9249_38510.4255787037_6_7_10_12_50.npz")
with open("9249_38510.4255787037_6_7_10_12_50.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

mu = npz["mu"]                      # (T, 32)
logvar = npz["logvar"]              # (T, 32)
start_times = npz["start_times_sec"]
hop_sec = float(npz["hop_length_sec"])
window_sec = float(npz["window_length_sec"])
energy = npz["energy"] if "energy" in npz else None

dph = meta["entry"]["dph"]
split = meta["entry"]["split"]
audio_dir_rel = meta["entry"]["audio_dir_rel"]
```

If you only need one vector per clip, a common reduction is:

```python
clip_mu = mu.mean(axis=0)
```

That is the representation used for the full-corpus PCA inspection.

## Notes For Analysts

- The latent stride is `5.805 ms`; do not confuse it with the finer internal
  spectrogram-bin spacing inside each 30 ms window.
- Paths contain spaces, for example `day43 Bells`, so quote S3 URIs in shell
  commands.
- If you need raw audio aligned to the latent sequence, use the matching path
  under the `audio/` prefix and the `start_times_sec` array.
- If you need split assignments but do not want to download `.json` sidecars,
  join by the parent directory using the manifest top-level `train` and `test`
  lists.
