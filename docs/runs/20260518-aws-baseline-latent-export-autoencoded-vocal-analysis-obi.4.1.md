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

## Next Gate

Continue monitoring the full export until all 602 children are terminal. Then sync latent outputs locally, rerun `scripts/inventory_developmental_replication_inputs.py`, and run the multi-bird developmental branch-commitment replication with no bird substitutions.
