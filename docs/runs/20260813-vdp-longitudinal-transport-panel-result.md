# VDP longitudinal transport panel result

Date: `2026-08-13`

Program Bead: `vdp-transport`  
Producer Bead: `autoencoded-vocal-analysis-x43`

## Verdict

Pass for exploratory latent export. The panel preserves the registered
bird-disjoint source split, contains no bird overlap between partitions, and
has complete audio and ROI availability. It is not a scientific result.

## Frozen panel

- Members: 5,106 clips forming 2,553 bird-by-DPH measures.
- Sampling: exactly two stable hash-ranked clips per measure.
- Birds: 47 total; 39 source-train and 8 source-test.
- Regimes: 19 bells, 14 samba, 9 isolates, and 5 simple.
- Excluded by the preregistered 20-measure bird gate: `R14`, `R20`, `R205`,
  `R369`, `R428`, and `R557`.
- Audio availability: 5,106 of 5,106.
- Exportable ROI availability: 5,106 of 5,106, with at least one finite,
  positive-duration interval per selected clip.
- Panel manifest SHA-256:
  `04eb288807a4604159316f823b83b255f01c5f190db1ab983f36cfa957e0dd76`.
- Member manifest SHA-256:
  `0ea455e8798bd435baed3f19dd2fc2494630e21749dacf1dc7a966bccb881b0e`.

The preliminary audit predicted 5,174 clips and 2,587 measures. The exact
selection is smaller by 68 clips and 34 measures because that preliminary
count included otherwise eligible measures from six birds that subsequently
failed the preregistered bird-level minimum. No selection rule changed in
response to this difference.

## Authorization boundary

This result authorizes an exploratory AVA latent export with the registered
shared encoder. It does not register the panel, authorize confirmatory use, or
support a developmental or tutor/isolate claim.
