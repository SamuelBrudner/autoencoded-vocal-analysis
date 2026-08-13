# VDP longitudinal transport panel preregistration

Date: `2026-08-13`

Program Bead: `vdp-transport`  
Consumer Beads: `juvenile_learning_tda-huc2`, `juvenile_learning_tda-a5h4`

## Purpose

Produce the AVA-owned longitudinal latent export needed for exploratory,
bird-grouped posterior-transport validation in TDA. This panel uses the
accepted shared encoder and its exact registered 53-bird cohort; it does not
train or select a new representation.

## Frozen inputs

- Registered source dataset: `vdp-ds-002`.
- Source manifest SHA-256:
  `4c28dc6d36d9a045cd6bc5dba13f8601b0230cebfc447cbcdbf5e635a07f8921`.
- Source member manifest SHA-256:
  `f80f7284d46264224caafc4d76328528a3a318a865255e29ff7a0e91bcebd47c`.
- Registered encoder: `vdp-enc-002`, epoch 100 checkpoint SHA-256
  `881ecbd0b0013c4ba9c69662790e45785050d17bb93d23d4443a008bf4ebee08`.
- Selection seed: `20260813`.

## Frozen selection

1. Join exact cohort members to their registered manifest entries.
2. Require explicit finite `bird_id`, `dph`, and `regime`; missing values are
   excluded and never inferred from paths.
3. Preserve the existing bird-disjoint train/test assignment and reject any
   bird appearing in both splits or more than one regime.
4. Group members by bird × DPH. Require at least two source clips per measure.
5. Retain exactly two stable SHA-256-ranked clips per eligible measure.
6. Require at least 20 eligible DPH measures per bird. Report every excluded
   bird and every source measure below the clip minimum.
7. Preserve explicit `tutor_start_dph`; isolates remain null. Recording IDs
   remain null where unavailable.

## Pre-outcome operational clarification

Before any longitudinal latent export or transport result was produced, panel
construction was made to enforce an exporter prerequisite that had previously
been implicit: a selected clip must have at least one finite,
positive-duration ROI before stable hash ranking. The manifest reports the
excluded member and group counts. Exact stem matching found all selected clips
exportable; an earlier diagnostic mismatch was caused by treating decimal
points within extensionless clip stems as filename suffixes. This availability
criterion is independent of latent values and scientific outcomes; the seed,
grouping, minimum-measure gate, and all downstream estimators remain unchanged.

A preliminary read-only coverage audit predicted 5,174 clips, 2,587 measures, and 47 birds:
39 source-train and 8 source-test birds; 19 bells, 14 samba, 9 isolate, and 5
simple birds. These are selection-integrity expectations, not scientific
outcomes.

## Export and downstream status

The export retains `ava_latent_sequence_v1`, the frozen training-cohort
normalization, a 30 ms acoustic window, and a 10 ms hop. It is exploratory and
uses `bird_disjoint` split semantics. TDA may use the source-test birds only as
a grouped validation set; no clip or measure may cross bird partitions.

No tutor/isolate or developmental claim is authorized by panel construction.
Cloud export requires a separate costed preflight and explicit launch approval.
