#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

BUCKET="${BUCKET:-ava-birdsong-us-east-1-a1859d31}"
PREFIX="${PREFIX:-autoencoded-vocal-analysis/pk249-33-90}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pk249-33-90-4gpu-batch-${RUN_STAMP}}"
ROI_JOB_NAME="${ROI_JOB_NAME:-pk249-33-90-roi-${RUN_STAMP}}"
TRAIN_JOB_NAME="${TRAIN_JOB_NAME:-pk249-33-90-train-${RUN_STAMP}}"

AWS_REGION="${AWS_REGION:-us-east-1}"
ROI_JOB_QUEUE="${ROI_JOB_QUEUE:-ava-roi-queue}"
ROI_JOB_DEFINITION="${ROI_JOB_DEFINITION:-ava-roi-jobdef}"
ROI_ARRAY_SIZE="${ROI_ARRAY_SIZE:-8}"
TRAIN_JOB_QUEUE="${TRAIN_JOB_QUEUE:-ava-gpu-queue-4x}"
TRAIN_JOB_DEFINITION="${TRAIN_JOB_DEFINITION:-ava-train-gpu-4x}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
TRAIN_DATASET_LENGTH="${TRAIN_DATASET_LENGTH:-262144}"
TEST_DATASET_LENGTH="${TEST_DATASET_LENGTH:-16384}"
TRAINER_KWARGS_JSON="${TRAINER_KWARGS_JSON:-{\"accelerator\":\"gpu\",\"devices\":4,\"strategy\":\"ddp_find_unused_parameters_true\",\"precision\":\"16-mixed\",\"log_every_n_steps\":10}}"
TRAIN_WORKDIR="${TRAIN_WORKDIR:-/mnt/ava_cache/${RUN_NAME}}"
DISK_TELEMETRY_EVERY_N_EPOCHS="${DISK_TELEMETRY_EVERY_N_EPOCHS:-5}"
DISABLE_SPEC_CACHE="${DISABLE_SPEC_CACHE:-1}"

MANIFEST_PATH="${MANIFEST_PATH:-${ROOT}/docs/runs/artifacts/autoencoded-vocal-analysis-8ot/manifest_pk249_33_90.json}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${ROOT}/docs/runs/artifacts/autoencoded-vocal-analysis-8ot/fixed_window_pk249_33_90.yaml}"
ROI_CONFIG_PATH="${ROI_CONFIG_PATH:-${ROOT}/examples/configs/birdsong_roi_medium_pilot.yaml}"
LOCAL_AUDIO_ROOT="${LOCAL_AUDIO_ROOT:-/Volumes/samsung_ssd/data/birdsong/day43 Bells/pk249}"
UPLOAD_JOBS="${UPLOAD_JOBS:-16}"

S3_MANIFEST_URI="s3://${BUCKET}/${PREFIX}/manifest_pk249_33_90.json"
S3_TRAIN_CONFIG_URI="s3://${BUCKET}/${PREFIX}/fixed_window_pk249_33_90.yaml"
S3_ROI_CONFIG_URI="s3://${BUCKET}/${PREFIX}/birdsong_roi_medium_pilot.yaml"
S3_AUDIO_ROOT="s3://${BUCKET}/${PREFIX}/audio"
S3_ROI_ROOT="s3://${BUCKET}/${PREFIX}/roi"
S3_RUN_ROOT="s3://${BUCKET}/${PREFIX}/training-runs"

WORK_ROOT="${WORK_ROOT:-/tmp/pk249_33_90_aws_pipeline_${RUN_STAMP}}"
mkdir -p "${WORK_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

json_field() {
  local path="$1"
  local expr="$2"
  python - "$path" "$expr" <<'PY'
import json
import sys

path = sys.argv[1]
expr = sys.argv[2].strip()
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
value = payload
for part in expr.split("."):
    if not part:
        continue
    value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

log "Using work root ${WORK_ROOT}"
log "Uploading manifest/config inputs to s3://${BUCKET}/${PREFIX}"
aws s3 cp "${MANIFEST_PATH}" "${S3_MANIFEST_URI}" --only-show-errors
aws s3 cp "${TRAIN_CONFIG_PATH}" "${S3_TRAIN_CONFIG_URI}" --only-show-errors
aws s3 cp "${ROI_CONFIG_PATH}" "${S3_ROI_CONFIG_URI}" --only-show-errors

log "Uploading PK249 manifest audio subset to ${S3_AUDIO_ROOT} with ${UPLOAD_JOBS} concurrent directory syncs"
python "${ROOT}/scripts/cloud/aws/upload_manifest_audio_to_s3.py" \
  --manifest "${MANIFEST_PATH}" \
  --split all \
  --s3-audio-root "${S3_AUDIO_ROOT}" \
  --jobs "${UPLOAD_JOBS}" \
  --summary-out "${WORK_ROOT}/upload_summary.json"

ROI_SUBMIT_JSON="${WORK_ROOT}/roi_submit_response.json"
ROI_PAYLOAD_JSON="${WORK_ROOT}/roi_submit_payload.json"

log "Submitting ROI Batch array job ${ROI_JOB_NAME}"
python "${ROOT}/scripts/cloud/aws/submit_birdsong_roi_batch_array_job.py" \
  --job-name "${ROI_JOB_NAME}" \
  --job-queue "${ROI_JOB_QUEUE}" \
  --job-definition "${ROI_JOB_DEFINITION}" \
  --array-size "${ROI_ARRAY_SIZE}" \
  --manifest-s3-uri "${S3_MANIFEST_URI}" \
  --segment-config-s3-uri "${S3_ROI_CONFIG_URI}" \
  --s3-audio-root "${S3_AUDIO_ROOT}" \
  --s3-roi-root "${S3_ROI_ROOT}" \
  --split all \
  --skip-existing \
  --emit-json "${ROI_PAYLOAD_JSON}" \
  --submit > "${ROI_SUBMIT_JSON}"

ROI_JOB_ID="$(json_field "${ROI_SUBMIT_JSON}" jobId)"
log "Submitted ROI job id ${ROI_JOB_ID}"

TRAIN_SUBMIT_JSON="${WORK_ROOT}/train_submit_response.json"
TRAIN_PAYLOAD_JSON="${WORK_ROOT}/train_submit_payload.json"
TRAIN_EXTRA_ARGS=()
case "${DISABLE_SPEC_CACHE,,}" in
  1|true|yes|on)
    TRAIN_EXTRA_ARGS+=(--disable-spec-cache)
    ;;
esac

log "Submitting training Batch job ${TRAIN_JOB_NAME} depending on ROI completion"
python "${ROOT}/scripts/cloud/aws/submit_birdsong_training_job.py" \
  --job-name "${TRAIN_JOB_NAME}" \
  --job-queue "${TRAIN_JOB_QUEUE}" \
  --job-definition "${TRAIN_JOB_DEFINITION}" \
  --manifest-s3-uri "${S3_MANIFEST_URI}" \
  --config-s3-uri "${S3_TRAIN_CONFIG_URI}" \
  --s3-audio-root "${S3_AUDIO_ROOT}" \
  --s3-roi-root "${S3_ROI_ROOT}" \
  --s3-run-root "${S3_RUN_ROOT}" \
  --run-name "${RUN_NAME}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --num-workers "${TRAIN_NUM_WORKERS}" \
  --train-dataset-length "${TRAIN_DATASET_LENGTH}" \
  --test-dataset-length "${TEST_DATASET_LENGTH}" \
  --trainer-kwargs-json "${TRAINER_KWARGS_JSON}" \
  --disk-telemetry-every-n-epochs "${DISK_TELEMETRY_EVERY_N_EPOCHS}" \
  --workdir "${TRAIN_WORKDIR}" \
  --depends-on-job-id "${ROI_JOB_ID}" \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --emit-json "${TRAIN_PAYLOAD_JSON}" \
  --submit > "${TRAIN_SUBMIT_JSON}"

TRAIN_JOB_ID="$(json_field "${TRAIN_SUBMIT_JSON}" jobId)"
log "Submitted training job id ${TRAIN_JOB_ID}"
log "Training outputs will land under ${S3_RUN_ROOT}/${RUN_NAME}"
