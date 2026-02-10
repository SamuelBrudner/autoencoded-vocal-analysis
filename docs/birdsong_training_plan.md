# Birdsong Training Plan (Fixed-Window Shotgun VAE)

This plan captures the staged approach from local validation to full-scale preprocessing and training.

## Phase 0 — Prep (1–2 hours)
- Confirm the sub-syllable config (30 ms, 44.1 kHz) and GPU defaults.
- Select a small validation cohort (e.g., 1–2 birds per regime: bells/simple/samba/isolates).

## Phase 1 — Local Validation (1–2 days)
1) Cohort selection
   - Use the metadata parquet to choose birds across all regimes.
2) ROI generation (local)
   - Run segmentation on a handful of leaf dirs.
   - Inspect a few ROI files for sanity.
3) Mini training run
   - 5–10 epochs on the subset (GPU or CPU).
   - Verify reconstructions and latent plots.
4) Iterate parameters
   - Tune min_freq/max_freq and spec_min_val/spec_max_val.
   - Adjust window_length if needed.

## Phase 2 — Medium-Scale Pilot (1–2 weeks)
1) Expand to 10–20 birds, stratified by regime.
2) Generate ROIs for all leaf dirs in the cohort.
3) Train longer (50–100 epochs).
4) Check for collapse/overfit; tune kl_warmup/kl_beta.

## Phase 3 — Full Dataset Preprocessing
1) Build a manifest of all leaf audio directories.
   - Use `scripts/build_birdsong_manifest.py` (schema: `docs/birdsong_manifest.md`).
2) Run ROI generation in parallel (local or cloud).
3) Produce a ROI coverage report (files/segments per bird/regime).

## Phase 4 — Full Dataset Training
1) Train with multi-GPU or multi-node data parallel.
2) Enable caching, AMP, and multi-worker DataLoaders.
3) Monitor throughput + metrics; adjust batch size and loader params.

## Notes
- Train/test split should include all regimes and be stratified by bird to avoid leakage.
- The fixed-window dataset uses ROI files (onset/offset in seconds) for sampling windows.
- Use `examples/configs/fixed_window_finch_30ms_44k.yaml` as the baseline config.
