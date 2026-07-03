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
