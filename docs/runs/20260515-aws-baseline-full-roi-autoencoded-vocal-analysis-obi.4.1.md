# AWS Developmental Baseline Staging Plan

## Summary

- Cohort rows: 602 directories; wav files: 1138367.
- Birds: 11 (PK249, R426, R467, R404, R150, R493, R470, R203, R425, R229, R122).
- dph range: 33.0..90.0.
- S3 root: `s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/developmental-baseline-ava-v1-full-20260515`.
- AWS preflight: ok.

## AVA Lineage

- Source metadata: `/Volumes/samsung_ssd/data/ava_hyperbolic_pk249_inputs/latent_sequences/day43 Bells/pk249/33/9249_38453.434375_4_11_10_25_30.json`.
- Checkpoint path in export metadata: `/mnt/ava_cache/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/inputs/checkpoint_050.tar`.
- Config path in export metadata: `/mnt/ava_cache/pk249-33-90-latent-hop005805-4shard-roifix-20260410_144941/inputs/config.yaml`.
- Window/hop: 0.03 sec / 0.005804988662131519 sec.
- Export energy: True.

## Artifacts

- `upload_audio_dry_run_stdout`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/upload_audio_dry_run_stdout.txt`
- `roi_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_batch_payload.json`
- `latent_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_batch_payload.json`
- `roi_smoke_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_smoke_batch_payload.json`
- `latent_smoke_batch_payload`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/latent_smoke_batch_payload.json`
- `audio_coverage_initial`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_coverage.json`
- `audio_coverage_after_repair`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_coverage_after_repair.json`
- `audio_coverage_final`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_coverage_final.json`
- `upload_audio_summary_shards`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/upload_audio_summary_shard_*.json`
- `upload_audio_repair_summary`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/upload_audio_repair_summary.json`
- `audio_coverage_repair_summary`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_coverage_repair_summary.json`
- `audio_hidden_sidecar_removal`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_hidden_sidecar_removal.json`
- `roi_full_submit_stdout`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_full_submit_stdout.json`
- `roi_full_describe_jobs`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_full_describe_jobs.json`
- `roi_full_child_status_initial`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_full_child_status_initial.json`
- `roi_full_child_status_followup`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_full_child_status_followup.json`
- `roi_full_child_status_latest`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_full_child_status_latest.json`
- `roi_retry_manifest`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_failed_20260518_manifest.json`
- `roi_retry_index_map`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_failed_20260518_index_map.json`
- `roi_retry_submit_stdout`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_20260518_submit_stdout.json`
- `roi_retry_status`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_20260518_status.json`
- `roi_retry_r470_bad_wav_inventory`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_r470_bad_wav_inventory.json`
- `roi_retry_r470_local_debug_summary`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_r470_local_debug_summary.json`
- `roi_retry_r470_recovered_summary`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_r470_recovered_summary.json`
- `roi_retry_r470_recovered_uploads`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/roi_retry_r470_recovered_uploads.json`
- `aws_staging_plan`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/aws_staging_plan.json`
- `aws_preflight`: `artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/aws_preflight.json`

## Post-Upload Coverage Gate

Run this after all audio upload shard summaries report zero failures and before ROI submission:

```bash
/opt/anaconda3/bin/python /Users/samuelbrudner/.codex/worktrees/cc22/autoencoded-vocal-analysis/scripts/cloud/aws/check_manifest_audio_s3_coverage.py --manifest /Users/samuelbrudner/.codex/worktrees/cc22/autoencoded-vocal-analysis/docs/runs/artifacts/autoencoded-vocal-analysis-obi.4.1/20260513-011500-developmental-input-inventory/developmental_cohort_manifest.json --split all --s3-audio-root s3://ava-birdsong-us-east-1-a1859d31/autoencoded-vocal-analysis/developmental-baseline-ava-v1-full-20260515/audio --out docs/runs/artifacts/autoencoded-vocal-analysis-obi.4.1/20260515-aws-baseline-full-roi/audio_coverage.json --fail-on-missing --fail-on-count-mismatch
```

## Execution Update

Audio upload was run as four local shard controllers over the 602-entry cohort manifest. The first pass wrote all directories but reported 25 transient S3 errors across shard summaries, caused by request timeouts or TLS EOFs. A targeted low-concurrency repair over the 25 failed directories completed with zero repair failures.

The initial post-upload coverage gate found 11 count mismatches: ten S3 undercounts and one overcount. The ten undercount directories were repaired with zero failures. The single overcount was traced to one hidden AppleDouble sidecar file under `isolates/R 203/82`; local scanning found exactly one such `._*.wav` sidecar across the cohort manifest. That non-audio sidecar object was removed from S3 and recorded in `audio_hidden_sidecar_removal.json`.

The final strict audio coverage gate passed:

- Manifest entries: 602.
- Expected WAV files: 1138367.
- Observed S3 directories: 602.
- Observed S3 WAV files: 1138367.
- Missing directories: 0.
- Count mismatch directories: 0.
- Unmatched WAV keys: 0.

The full ROI Batch array was submitted after the coverage gate passed. Initial child status showed 590 runnable, 4 starting, 3 running, 5 succeeded, and 0 failed. A follow-up snapshot showed 585 runnable, 1 starting, 6 running, 10 succeeded, and 0 failed. The latest snapshot before the first full-run commit showed 563 runnable, 1 starting, 7 running, 31 succeeded, and 0 failed.

The full array reached terminal state with 592 succeeded and 10 failed. Seven failures were infrastructure/no-start failures and succeeded in a targeted 10-shard retry. The remaining three failures were R470 dph 38, 42, and 47; local reproduction showed unreadable WAV headers in those folders, so parquet ROI generation was patched to skip only clips that fail at `wavfile.read` and record read-skip counts in the ROI summary.

The R470 local recovery generated ROI parquet bundles for all three remaining directories and uploaded them to the staged ROI prefix:

- R470 dph 38: 1,361 readable clips, 3,662 ROI segments, 222 unreadable WAV headers skipped.
- R470 dph 42: 869 readable clips, 4,461 ROI segments, 365 unreadable WAV headers skipped.
- R470 dph 47: 1,754 readable clips, 11,622 ROI segments, 130 unreadable WAV headers skipped.

Final ROI staging status is complete: 602/602 cohort directories now have `roi.parquet` in S3 and in the synced local ROI root. The post-recovery local inventory reports ROI parquet artifacts for all 11 requested birds; the only remaining input gap is latent export for the ten non-PK249 birds.

## Next Gate

Move to AVA latent export staging under `autoencoded-vocal-analysis-obi.4.1.2`. Start with the existing two-shard smoke payload, then scale to the full cohort export once the smoke summaries show exported latents and expected no-ROI skips for unreadable R470 clips.
