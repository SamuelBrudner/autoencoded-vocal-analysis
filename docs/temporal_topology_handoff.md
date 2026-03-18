# Temporal Export + Topology Handoff (AVA → juvenile_learning_tda)

This note documents how to export **time-indexed latent sequences** from AVA and
plug them into the **distance-image → persistence** pipeline prototyped in
`juvenile_learning_tda`.

For the canonical on-disk schema (NPZ arrays + JSON metadata), see:
`docs/latent_sequence_export.md`.

## Prereqs

- A birdsong **manifest JSON** (schema: `docs/birdsong_manifest.md`).
- ROI `.txt` files laid out per `docs/birdsong_roi_layout.md`.

Optional preflight:

```bash
python scripts/validate_birdsong_rois.py --manifest path/to/manifest.json
```

## Export Latent Sequences (CLI)

Export one `.npz` + one `.json` per wav file:

```bash
python scripts/export_latent_sequences.py \
  --manifest path/to/birdsong_manifest.json \
  --split train \
  --config examples/configs/fixed_window_finch_30ms_44k.yaml \
  --checkpoint path/to/checkpoint_100.tar \
  --out-dir path/to/latent_exports \
  --device cuda \
  --batch-size 64 \
  --export-energy
```

Notes:
- **Resume/skip** is on by default. Use `--no-skip-existing` or `--force` to
  recompute.
- For distributed export, use deterministic **sharding**:

```bash
python scripts/export_latent_sequences.py ... --num-shards 16 --shard-index 0
```

## Output Layout

For each wav, the exporter writes:

- `<out_dir>/<clip_id>.npz`
- `<out_dir>/<clip_id>.json`

Where `clip_id` is the wav stem, optionally prefixed by the manifest
`audio_dir_rel` (so subdirectories may be created under `out_dir`).

The `.npz` includes (time-major):
- `start_times_sec`: `[T]`
- `window_length_sec`: `[]`
- `hop_length_sec`: `[]`
- `mu`: `[T, z_dim]`
- `logvar`: `[T, z_dim]`
- optional `energy`: `[T]` (when `--export-energy` is set)

## Handoff: Build Distance Images + Persistence (juvenile_learning_tda)

The minimal topology pipeline in `juvenile_learning_tda` expects a **distance
image** between two time series, optional **silence gating**, then cubical
persistence on an upsampled image.

### 1) Load two latent sequences

```python
import numpy as np


def load_latent_npz(path: str):
    npz = np.load(path)
    mu = npz["mu"]              # (T, z_dim)
    logvar = npz["logvar"]      # (T, z_dim)
    sigma = np.exp(0.5 * logvar)
    energy = npz["energy"] if "energy" in npz else None  # (T,)
    return mu, sigma, energy


mu1, sigma1, energy1 = load_latent_npz("latent_exports/clip_a.npz")
mu2, sigma2, energy2 = load_latent_npz("latent_exports/clip_b.npz")
```

### 2) Distance image between Gaussian posteriors

This matches the student-side distance used in `juvenile_learning_tda`:

```python
diff = mu1[:, None, :] - mu2[None, :, :]
dist_sq = (diff ** 2).sum(-1)
dist_sq = dist_sq + (sigma1 ** 2).sum(-1)[:, None]
dist_sq = dist_sq + (sigma2 ** 2).sum(-1)[None, :]
D = np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)  # (T1, T2)
```

### 3) Optional silence gating (recommended only if needed)

If you exported `energy`, you can compute gating weights and apply the same
`alpha * (1 - w_i * w_j)` penalty described in `juvenile_learning_tda`:

```python
from birdsong_topo_minimal.distance import (
    apply_silence_gating,
    compute_gating_weights,
    minmax_scale,
    upsample_bilinear,
)

w1 = compute_gating_weights(energy1, percentile=70, tau=0.05)
w2 = compute_gating_weights(energy2, percentile=70, tau=0.05)

I = apply_silence_gating(D, w1, w2, alpha=0.0)  # set alpha>0 to enable gating
I = minmax_scale(I)                             # required by the Betti loss setup
I_up = upsample_bilinear(I, scale=2)            # helps diagonal connectivity
```

If you did not export `energy`, you can skip gating:

```python
from birdsong_topo_minimal.distance import minmax_scale, upsample_bilinear

I = minmax_scale(D)
I_up = upsample_bilinear(I, scale=2)
```

### 4) Persistence

```python
from birdsong_topo_minimal.persistence import cubical_persistence, persistence_diagram

pers = cubical_persistence(I_up, superlevel=False)
diag_h0 = persistence_diagram(pers, homology_dim=0)  # (n_bars, 2)
```

## Practical Notes

- The exported sequences are **window-indexed** (fixed hop). Comparing two
  clips yields a rectangular `(T1, T2)` distance image; this is expected.
- `start_times_sec` is included for alignment/debugging; most topology steps
  only need `(mu, logvar)` in time order.
- `energy` is RMS over the **resampled audio** inside each window. It is not
  globally normalized; percentile-based thresholds are usually more robust than
  absolute thresholds.

