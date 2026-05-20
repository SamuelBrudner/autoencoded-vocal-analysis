# AWS Baseline Latent Export

## Summary

- Bead: `autoencoded-vocal-analysis-obi.4.1.2`.
- Input gate: ROI staging complete for 602/602 cohort directories and 11/11 birds.
- AVA lineage: staged `checkpoint_050.tar`, staged `config.yaml`, `window_length_sec=0.03`, `hop_length_sec=0.005804988662131519`, export energy enabled.
- Latent root: `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/developmental-baseline-ava-v1-full-20260515/latents/ava_latent`.

## Smoke

The two-shard capped smoke export succeeded:

- Batch children: 2 succeeded, 0 failed.
- Export summaries: 2/2 present.
- Output objects after smoke: 8 objects, 4 clip exports.
- Smoke artifacts:
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_smoke_20260518_submit_stdout.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_smoke_20260518_status.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_smoke_20260518_summaries/`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_smoke_20260518_output_inventory.json`

## CPU Benchmark

The two-shard full-directory CPU benchmark succeeded:

- Batch children: 2 succeeded, 0 failed.
- Shard 0: 140 clips total, 133 exported, 5 skipped without ROI, 2 skipped without windows, 0 failed.
- Shard 1: 328 clips total, 328 exported, 0 skipped, 0 failed.
- Benchmark artifacts:
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_benchmark_20260518_payload.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_benchmark_20260518_submit_stdout.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_benchmark_20260518_status.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_benchmark_20260518_summaries/`

## Full Export

The full 602-shard CPU latent export was submitted after the smoke and benchmark passed.

- Batch job id: `5956c373-8e36-4fbf-aad1-da43cabb5dd8`.
- Initial status at 2026-05-18 14:51:05 UTC: 594 runnable, 8 starting, 0 failed.
- Follow-up status at 2026-05-18 14:55:57 UTC: 593 runnable, 8 running, 1 succeeded, 0 failed.
- Latest saved status before commit at 2026-05-18 14:57:12 UTC: 592 runnable, 1 starting, 7 running, 2 succeeded, 0 failed.
- The queue currently permits 16 vCPU total for this Fargate path, so only 8 two-vCPU children run concurrently. This is expected to complete slowly but cheaply; if wall time becomes the bottleneck, move to the GPU latent-export scaling bead rather than changing the scientific analysis.
- Full-run artifacts:
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_full_20260518_payload.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_full_20260518_submit_stdout.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_full_20260518_status.json`

## Retry

The full array reached terminal state on 2026-05-19 with 601 succeeded and 1 failed. The single failed child was shard 265, which maps to `day 60 samba/R150/66`; it failed before container start due to an image-pull timeout, not an export error.

A targeted retry was launched on 2026-05-20 using a one-directory manifest and a two-child array:

- Retry job id: `92c3b944-93a4-41a6-9755-0129cd9e430b`.
- Retry child 0 processed `day 60 samba/R150/66`; retry child 1 was the expected no-op shard.
- Retry status: 2 succeeded, 0 failed.
- Retry export: 1,464 clips seen, 1,462 exported, 2 skipped without ROI, 0 failed.
- Combined full-plus-retry coverage: 602/602 shard summaries, 1,138,367 clips seen, 914,788 exported, 223,579 skipped, 0 failed.
- Combined skip counts: 204,681 no-ROI skips and 18,898 no-window skips.
- Retry artifacts:
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_manifest.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_index_map.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_payload.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_submit_stdout.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_status.json`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_retry_shard265_20260520_summaries/`
  - `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_full_plus_retry_20260520_summary.json`

## Next Gate

Sync latent outputs locally, rerun `scripts/inventory_developmental_replication_inputs.py`, and run the multi-bird developmental branch-commitment replication with no bird substitutions.
