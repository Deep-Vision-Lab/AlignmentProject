#!/usr/bin/env bash
# Submit direct synthetic training and queue calibrated holdout evaluation after it.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

JOB_ID="${JOB_ID:-cnn_connected_subword_direct_synthetic}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
EPOCHS="${EPOCHS:-35}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-128}"
THRESHOLDS="${THRESHOLDS:-0.60,0.70,0.80,0.85,0.90}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-100}"
HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES:-100}"
CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-1}"
HOLDOUT_START_INDEX="${HOLDOUT_START_INDEX:-101}"

unset PRETRAINED_WEIGHTS SYNTHETIC_WEIGHTS

TRAIN_OUTPUT="$(
  JOB_ID="${JOB_ID}" \
  DATA_DIR="${DATA_DIR}" \
  HF_HOME="${HF_HOME}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  EPOCHS="${EPOCHS}" \
  NUM_GPUS="${NUM_GPUS}" \
  EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE}" \
  WINDOW_SIZE="${WINDOW_SIZE:-32}" \
  STRIDE_RATIO="${STRIDE_RATIO:-0.25}" \
  LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
  TRAIN_SEED="${TRAIN_SEED:-42}" \
  DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}" \
  bash scripts/train/run_connected_subword_direct_synthetic.sh
)"
printf '%s\n' "${TRAIN_OUTPUT}"

TRAIN_JOB_ID="$(awk '/Submitted batch job/ {print $4}' <<< "${TRAIN_OUTPUT}" | tail -1)"
[[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not parse training Slurm job ID." >&2
  exit 1
}

WEIGHTS="${PROJECT_DIR}/Weights/${JOB_ID}/model_latest.pth"
EVAL_OUTPUT="$(
  WEIGHTS="${WEIGHTS}" \
  DATA_DIR="${DATA_DIR}" \
  RUN_TAG="${JOB_ID}_calibrated_holdout" \
  THRESHOLDS="${THRESHOLDS}" \
  CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX}" \
  CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES}" \
  HOLDOUT_START_INDEX="${HOLDOUT_START_INDEX}" \
  HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES}" \
  DEPENDENCY_JOB_ID="${TRAIN_JOB_ID}" \
  EVAL_JOB_NAME="eval_${JOB_ID}" \
  bash Evaluation/evaluate_connected_subword_direct_synthetic.sh
)"
printf '%s\n' "${EVAL_OUTPUT}"

EVAL_JOB_ID="$(awk '/^[0-9]+$/ {print $1}' <<< "${EVAL_OUTPUT}" | tail -1)"

cat <<EOF
Submitted direct connected-subword pipeline:
  training job   = ${TRAIN_JOB_ID}
  evaluation job = ${EVAL_JOB_ID:-unknown} (afterok:${TRAIN_JOB_ID})
  checkpoint     = ${WEIGHTS}
  results        = ${PROJECT_DIR}/Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/${JOB_ID}_calibrated_holdout
EOF
