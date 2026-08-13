# VDP longitudinal transport panel correction

Date: `2026-08-13`

Program Bead: `vdp-transport`  
Producer Bead: `autoencoded-vocal-analysis-x43`

## Trigger

The complete 5,106-clip AVA export passed producer contract validation, but
TDA's pre-inference grouping gate found 2,568 measures rather than the frozen
2,553 bird-by-DPH identities. No full-panel transport statistic had been
computed.

All 15 excess groups came from `R71`. Its selected clips span two source
cohorts with conflicting explicit `tutor_start_dph` values (`43` and `60`) at
the same bird and DPH. The panel selector had checked regime and source split
at bird level but had not rejected conflicting tutor-start metadata.

## Correction

Exclude `R71` from this analysis as metadata-ambiguous. Do not infer a tutor
start, merge contradictory values, select one source cohort after export, or
treat the two values as independent birds. Preserve the rejected complete
export for audit and create a byte-identical subset containing all other
members.

The corrected exploratory panel therefore contains 46 birds, 2,528 measures,
and 5,056 clips. Its source-test partition contains seven tutored birds and no
isolate bird, making the already descriptive test-only condition contrast
entirely inestimable. The pooled exploratory association remains observational
and uses bird-level resampling.

This correction is based solely on metadata identity and occurred before any
full-panel transport outcome. The shared encoder, posterior sampling,
Sinkhorn estimator, temporal-order endpoint, thresholds, and sensitivity
settings remain unchanged.
