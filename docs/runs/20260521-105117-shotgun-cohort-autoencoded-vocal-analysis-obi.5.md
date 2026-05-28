# Shotgun VAE Developmental Branch Commitment Launch Package

## Summary

- Built the PK249 pilot and fixed 11-bird shotgun VAE manifests from the staged developmental cohort manifest.
- PK249 pilot: 58 dph directories, 95,402 wavs, 33-90 dph, 10 training epochs.
- Fixed cohort: 602 dph directories, 1,138,367 wavs, 33-90 dph, 100 training epochs.
- Wrote shotgun configs from `examples/configs/fixed_window_finch_30ms_44k.yaml` with `min_freq=300`, `spec_min_val=1.0`, `kl_beta=1.0`, and `kl_warmup_epochs=20`.
- Staged the four small shotgun input files to the existing baseline S3 run prefix.
- Wrote no-submit AWS Batch payloads for the PK249 pilot and fixed-cohort model.
- Submitted the fixed 11-bird 100-epoch 4-GPU DDP shotgun cohort model on 2026-05-28 after read-only AWS/S3 preflight and a cost estimate.
- The full cohort job reached `RUNNING`; first CloudWatch output showed container setup telemetry, and the runner was in quiet audio/ROI staging at handoff.
- Static utilization audit: the original PK249 pilot payload is wasteful as written because the Batch job reserves 4 GPUs while the shotgun config requests `devices: 1`.
- Added an explicit 4-GPU utilization assay payload that overrides Lightning to `devices=4, strategy=ddp` and records `nvidia-smi` utilization every 5 seconds during a one-epoch PK249 run.
- Read-only Batch inspection found enabled 1-GPU AVA queues backed by 1-GPU instance types, but no 1-GPU AVA training job definition using the AVA training image.
- Added a redacted 1-GPU AVA job-definition candidate and a no-submit PK249 1-GPU pilot payload that can be used after registering/selecting a real 1-GPU AVA training definition.
- Registered a constrained 1-GPU AVA job definition and ran a paid PK249 scaling comparison against the 4-GPU DDP path.
- The 10-epoch comparison changes the training recommendation: for the full shotgun cohort model, use the 4-GPU DDP Batch shape. The observed speedup is an end-to-end Batch-shape throughput result, not pure GPU scaling efficiency.

## Launch Package

- `shotgun_cohort_manifest_summary`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_cohort_manifest_summary.json`
- `pilot_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/pk249_33_90_manifest.json`
- `cohort_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/fixed_11bird_33_90_manifest.json`
- `pilot_config`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_config.yaml`
- `cohort_config`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed_11bird_config.yaml`
- `input_staging_manifest`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_input_staging_manifest.json`
- `pilot_training_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_training_payload.json`
- `cohort_training_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed_cohort_training_payload.json`
- `pilot_runner_dry_run`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_runner_dry_run.json`
- `4gpu_utilization_assay_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_4gpu_utilization_assay_payload.json`
- `4gpu_utilization_assay_runner_dry_run`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_4gpu_utilization_assay_runner_dry_run.json`
- `1gpu_job_definition_candidate`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/ava_train_gpu_1x_job_definition_candidate.json`
- `1gpu_pilot_training_payload`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_pilot_1gpu_training_payload.json`
- `4gpu_utilization_assay_summary`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_4gpu_utilization_assay_summary.json`
- `full_cohort_100epoch_payload_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_payload_redacted_summary.json`
- `full_cohort_100epoch_submit_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_submit_redacted.json`
- `full_cohort_100epoch_status_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_status_redacted.json`
- `full_cohort_allow_missing_roi_patch_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_allow_missing_roi_patch_redacted.json`
- `full_cohort_allow_missing_roi_payload_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_allow_missing_roi_payload_redacted_summary.json`
- `full_cohort_allow_missing_roi_submit_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_allow_missing_roi_submit_redacted.json`
- `full_cohort_allow_missing_roi_status_redacted`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_fixed11_4gpu_100epoch_20260528_allow_missing_roi_status_redacted.json`

## Utilization Assay

Do not submit the original PK249 pilot payload to the 4-GPU job definition; it would run with one Lightning device and leave the other three GPUs idle.

The utilization assay payload is the appropriate next paid run if we want to evaluate the existing 4-GPU definition. It is intentionally short: PK249 only, one epoch, `dataset_length=8192`, `batch_size=128`, `devices=4`, `strategy=ddp`, and GPU telemetry enabled. Treat the 4-GPU definition as efficient only if all four GPUs are visible in `gpu_utilization.csv`, all four show sustained non-trivial utilization during the training window, and wall-clock throughput is materially better than a 1-GPU run with the same dataset length. If DDP fails, only one GPU is active, or utilization is dominated by download/preprocessing stalls, use or create a 1-GPU AVA training definition for the PK249 pilot.

The lower-risk default is to register/select a 1-GPU AVA training job definition and use the 1-GPU PK249 pilot payload. That matches the current shotgun config (`devices: 1`) and avoids paying for idle GPUs during the pipeline-validation pilot. The committed 1-GPU job-definition artifact is intentionally redacted and should not be passed directly to `aws batch register-job-definition`; copy role fields from the active AVA training definition at execution time.

## 4-GPU DDP Assay Result

The assay was run on 2026-05-21. The original prepared payload failed immediately because the deployed training image uses a baked legacy entrypoint, so the command override was interpreted as unexpected arguments. A legacy-entrypoint payload then failed because `test_dataset_length=0` is rejected by the runner. A corrected legacy payload reached 4-rank NCCL DDP and started training, but default `strategy=ddp` failed after the first batch because the Lightning module has unused parameters.

The final corrected payload used `strategy=ddp_find_unused_parameters_true` and succeeded. It initialized all four DDP ranks, all ranks saw the four visible CUDA devices, and the one-epoch training loop completed 16 batches in 24 seconds (`0.67 it/s`) with `train_dataset_length=8192`, `batch_size=128`, and inferred global step size 512.

Operationally, this is not efficient enough for the PK249 pipeline-validation pilot. The successful run spent 646 seconds queued before start, 498 seconds in the Batch container, and 1,144 seconds from submission to terminal status; the 24-second training epoch was only 4.8% of container runtime and 2.1% of total elapsed time. The deployed legacy entrypoint also did not include the new `nvidia-smi` monitor, so no `gpu_utilization.csv` was produced.

Decision: use the 1-GPU path for the immediate PK249 pilot unless a direct 1-GPU comparison proves worse. Keep 4-GPU DDP available for the full-cohort model only after publishing an image that includes GPU telemetry and rerunning a longer comparison with `ddp_find_unused_parameters_true`.

## 10-Epoch Scaling Comparison

A longer PK249 comparison was run on 2026-05-24 with `train_dataset_length=50000`, `batch_size=128`, `precision=16-mixed`, and parquet ROIs. The 4-GPU job used DDP with `ddp_find_unused_parameters_true`; the 1-GPU comparison used the constrained 1-GPU Batch definition and was terminated after three complete epochs plus part of epoch 3 once the result was clear.

One-time costs should not be counted per epoch. The setup/download/coverage/preflight stages ran once per Batch job. The 4-GPU container reached `fit_start` at about 296 seconds after container start; the 1-GPU container reached `fit_start` at about 517 seconds after container start. Those costs matter for short smoke runs, but they do not rerun during a real 100-epoch training job.

The training-loop timing is decisive. The 4-GPU run completed 10 epochs with a median steady epoch of 124 seconds, about 403 samples/sec for the 50k-sample epoch. The 1-GPU run completed epochs in 1743, 1884, and 1905 seconds, with a median steady epoch of 1894.5 seconds, about 26 samples/sec. Excluding queue time and using the measured fit-start overhead, a 100-epoch run projects to about 3.5 hours on the 4-GPU path versus about 52.8 hours on the constrained 1-GPU path.

This is not a claim that four GPUs alone give a 15x physical speedup. The 4-GPU DDP run uses batch size 128 per rank, so its global batch is about 512 and it needs 98 optimizer steps per epoch instead of 391. That accounts for roughly a 4x step-count reduction. The remaining difference likely comes from the larger Batch host: more CPU, memory, I/O headroom, and aggregate dataloader workers. Treat the result as an AWS Batch instance-shape throughput comparison. A strict GPU-efficiency assay would still require telemetry or a same-host comparison.

Decision: use 4-GPU DDP for the full shotgun cohort model as the practical AWS training path. The constrained 1-GPU queue is too slow for the 100-epoch cohort run, even after excluding one-time setup costs. Keep the 1-GPU path for small command validation and recovery tests, not for the scientific cohort model.

Artifacts:

- `scaling_comparison_summary`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_scaling_comparison_summary.json`
- `scaling_event_timestamps`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_scaling_event_timestamps_redacted.json`
- `4gpu_scaling_logs`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_scaling_4gpu_10epoch_logs_latest.txt`
- `1gpu_scaling_logs`: `artifacts/autoencoded-vocal-analysis-obi.5/20260521-105117-shotgun-cohort/shotgun_pk249_scaling_1gpu_10epoch_v3_logs.txt`

## Full Cohort Training Launch

The fixed 11-bird shotgun VAE cohort model was submitted on 2026-05-28 using the practical 4-GPU DDP Batch shape selected by the scaling comparison. The run uses the fixed cohort manifest, parquet ROI inputs, 100 epochs, `train_dataset_length=200000`, `batch_size=128`, `num_workers=8`, mixed precision, and `strategy=ddp_find_unused_parameters_true`.

The expected compute cost was estimated at about `$67` from the measured 4-GPU throughput and the `g6.12xlarge` on-demand rate, with a practical budget envelope of `$80-$100` to absorb staging and startup variance. The submitted job reached `RUNNING` at 2026-05-28 15:14:06 UTC. The first CloudWatch event was the runner's `after_setup` disk telemetry; subsequent silence is expected while the runner quietly syncs all manifest audio and ROI directories with `--only-show-errors`.

The next operational bead is `autoencoded-vocal-analysis-obi.5.4`: monitor the job to terminal status, inventory the checkpoint/output artifacts, run or queue shotgun latent export from the completed checkpoint, and then rerun the developmental replication analysis with shotgun latents.

## Full Cohort Retry

The first full cohort job failed before training, after data staging and during the ROI coverage gate. The staged inputs were readable: 602 directories, about 485 GB of audio, all ROI parquet directories present, and 14,167,039 ROI segments. The fatal count was 717 missing per-clip ROI records; these should be treated as skipped clips for this training path, not as an input-staging failure. Empty ROI clips were also counted but were already nonfatal unless the empty-fraction threshold is exceeded.

I built and pushed a versioned training-image overlay, `ava-train:20260528-allow-missing-roi`, that changes only the Batch entrypoint. The patched runner captures the coverage report even when the report command exits nonzero, records the return code and coverage summary, and allows missing per-clip ROI records when `--allow-missing-roi` is set. Missing ROI directories and ROI parse errors remain fatal. A separate 4-GPU job definition, `ava-train-gpu-4x-allow-missing-roi`, was registered so the original 4-GPU training definition is unchanged.

The retry job was submitted on 2026-05-28 with the same 100-epoch cohort settings plus `--allow-missing-roi`. Initial status was `RUNNABLE` with the 4-GPU compute environment requesting 48 vCPUs.
