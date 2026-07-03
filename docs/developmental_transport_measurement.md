# Developmental Transport Measurement Plan

This document records the measurement-first path for developmental vocal
learning analyses in AVA latent space. It supersedes fully joint
encoder/dynamics/topology training as the first confirmatory target.

## Core Decision

The primary estimand is the developmental movement of frozen latent measures:

```text
Q = T[p_{bird, time}(z), p_{bird, time + delta}(z)]
```

Here `p_{bird, time}(z)` is the empirical distribution of framewise VAE posterior
states for one bird at one developmental time. The encoder is treated as a
measurement instrument. Developmental labels such as age, tutor regime, tutor
start day, and isolate/tutored status must not back-propagate into the encoder
for confirmatory analyses.

## Confirmatory Spine

1. Train one shared, age-blind encoder across birds and regimes.
2. Freeze the encoder.
3. Export latent sequences with `mu`, `logvar`, timestamps, and metadata.
4. Build bird-by-developmental-time latent measures without averaging away
   posterior uncertainty.
5. Model development as regularized transport between consecutive measures.
6. Evaluate with bird-level splits, tutor-onset contrasts, isolate-vs-tutored
   contrasts, and rig/era controls where metadata permit.

TDA and path signatures remain important, but they are read-only summaries at
this stage. TDA measures component/loop/recurrence structure of latent measures.
Path signatures summarize ordered traversal through local latent paths. Neither
should train the encoder in the first confirmatory analysis.

## Why Transport First

Transport makes merge, split, abandonment, condensation, and drift well-posed
without committing to a fixed atlas. A fixed atlas can still be useful later as
a visualization layer or local coordinate chart, but it should not be the
load-bearing developmental ontology if the scientific question is whether
discrete token-like structure emerges.

The transport framing also gives a direct falsification target: a forward model
or interpolation rule should predict held-out future latent measures better
than simple baselines such as age-only smoothing, adjacent-age interpolation, or
within-bird shuffled developmental order.

## Immediate Gates

Run the cohort readiness audit:

```bash
python scripts/audit_developmental_transport_readiness.py \
  --manifest data/manifests/birdsong_manifest.json \
  --out-json artifacts/developmental_transport/cohort_readiness.json \
  --out-md artifacts/developmental_transport/cohort_readiness.md
```

This checks bird-level split integrity, longitudinal coverage, tutor-start-day
variation, DPH availability, and the isolate/tutored cohort balance needed to
separate growth from copying-related learning.

Train the first shared encoder with the transport-specific config:

```bash
python scripts/launch_birdsong_training.py \
  --manifest data/manifests/birdsong_manifest.json \
  --config examples/configs/fixed_window_transport_shared_30ms.yaml \
  --save-dir artifacts/transport_encoder/shared_30ms \
  --streaming \
  --roi-format parquet \
  --train-dataset-length 131072 \
  --test-dataset-length 16384 \
  --dry-run
```

Remove `--dry-run` only after the planned directory counts and ROI preflight
look sane. The config uses `entry_weight_mode: regime_bird_uniform` so the
streaming sampler gives each regime equal mass, then each bird within a regime
equal mass, instead of letting raw file counts dominate the shared encoder.

After frozen latent sequence export, build a latent-measure index:

```bash
python scripts/build_latent_measure_index.py \
  --latent-dir artifacts/latent_sequences \
  --out-json artifacts/developmental_transport/latent_measure_index.json \
  --clips-csv artifacts/developmental_transport/latent_measure_clips.csv \
  --measures-csv artifacts/developmental_transport/latent_measures.csv
```

The measure index groups per-clip latent sequence files into
`bird_id_norm x regime x dph` measures while retaining per-clip `logvar`
availability and window counts. It is an index, not a destructive aggregation.

## Hop Sensitivity Gate

Before using path signatures or fast latent dynamics as biological readouts,
export the same frozen encoder at multiple latent-sequence hops. Start by
holding the acoustic window fixed and sweeping only `--hop-length-sec`:

```bash
python scripts/export_latent_sequences.py \
  --manifest data/manifests/birdsong_manifest.json \
  --split all \
  --config path/to/fixed_window.yaml \
  --checkpoint path/to/checkpoint.tar \
  --out-dir artifacts/latent_hop_sweep/hop030 \
  --hop-length-sec 0.030 \
  --export-energy
```

Repeat with candidate hops such as `0.015`, `0.010`, and `0.0058`, then
summarize the exports:

```bash
python scripts/analyze_latent_hop_sensitivity.py \
  --latent-dir hop030=artifacts/latent_hop_sweep/hop030 \
  --latent-dir hop015=artifacts/latent_hop_sweep/hop015 \
  --latent-dir hop010=artifacts/latent_hop_sweep/hop010 \
  --latent-dir hop0058=artifacts/latent_hop_sweep/hop0058 \
  --out-json artifacts/developmental_transport/hop_sensitivity.json \
  --summary-csv artifacts/developmental_transport/hop_sensitivity_summary.csv \
  --clips-csv artifacts/developmental_transport/hop_sensitivity_clips.csv
```

The key warning sign is a hop that creates large apparent path length or
near-tautological one-step prediction while the AR(1)-style effective sample
fraction collapses. Transport summaries that are stable across hop are more
credible than signature or fast-dynamics summaries that depend strongly on
overlap.

## Hyperbolic Geometry Gate

Hyperbolic geometry remains a post-encoder readout, not the measurement
substrate. Use it only after the frozen-measure transport signal is credible:
build a developmental graph from measures, local components, or transport
links; compare hyperbolic distortion against Euclidean baselines; and require
branch/radius structure to survive bird-level and age-shuffled nulls. A
hyperbolic VAE should remain gated behind that post-hoc evidence.

## Non-Goals For The First Paper

- No developmental loss terms in the encoder.
- No age/tutor leakage into the latent representation.
- No fixed atlas as the primary ontology.
- No claim that a sparse slow law is a biological law until it survives
  leave-bird-out validation and simple developmental baselines.
- No requirement for waveform synthesis; spectrogram-level decode checks and
  latent-measure prediction are sufficient first falsification tests.

## Later Mechanistic Layer

Once the frozen-measure analysis establishes a stable transport signal, staged
models can ask which sparse terms explain transport summaries. SINDy-style fast
and slow dynamics, TDA summaries, and path signatures should enter here as
post-encoder explanatory models with cross-bird validation.
