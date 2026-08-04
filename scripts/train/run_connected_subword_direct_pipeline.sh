#!/usr/bin/env bash
# Submit balanced direct synthetic training fully offline.
# Optionally queue evaluation only on a separate, explicitly supplied dataset.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

JOB_ID="${JOB_ID:-cnn_connected_subword_direct_multisource}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DataSet}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
SYNTHETIC_DATA_DIRS="${SYNTHETIC_DATA_DIRS:-${DATA_ROOT}/Synthetic_Arabic_1,${DATA_ROOT}/Synthetic_Arabic_2,${DATA_ROOT}/Synthetic_Arabic_3,${DATA_ROOT}/Synthetic_Arabic_4}"
SYNTHETIC_SAMPLES_PER_DIR="${SYNTHETIC_SAMPLES_PER_DIR:-3000}"
SYNTHETIC_REQUIRE_FULL_PER_DIR="${SYNTHETIC_REQUIRE_FULL_PER_DIR:-1}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
EPOCHS="${EPOCHS:-35}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-}"
THRESHOLDS="${THRESHOLDS:-0.60,0.70,0.80,0.85,0.90}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-100}"
HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES:-100}"
CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-1}"
HOLDOUT_START_INDEX="${HOLDOUT_START_INDEX:-101}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"

[[ "${SYNTHETIC_SAMPLES_PER_DIR}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: SYNTHETIC_SAMPLES_PER_DIR must be a positive integer." >&2
  exit 2
}
IFS=',' read -r -a SYNTHETIC_ROOTS <<< "${SYNTHETIC_DATA_DIRS}"
SOURCE_COUNT="${#SYNTHETIC_ROOTS[@]}"
(( SOURCE_COUNT > 0 )) || { echo "ERROR: SYNTHETIC_DATA_DIRS is empty." >&2; exit 2; }
for root in "${SYNTHETIC_ROOTS[@]}"; do
  root="${root//[[:space:]]/}"
  [[ -d "${root}/images" && -d "${root}/texts" ]] || {
    echo "ERROR: synthetic source must contain images/ and texts/: ${root}" >&2
    exit 2
  }
done
NUM_SAMPLES=$((SOURCE_COUNT * SYNTHETIC_SAMPLES_PER_DIR))

if [[ -n "${EVAL_DATA_DIR}" ]]; then
  [[ -d "${EVAL_DATA_DIR}/images" && -d "${EVAL_DATA_DIR}/texts" ]] || {
    echo "ERROR: EVAL_DATA_DIR must contain images/ and texts/: ${EVAL_DATA_DIR}" >&2
    exit 2
  }
  for root in "${SYNTHETIC_ROOTS[@]}"; do
    root="$(readlink -f "${root//[[:space:]]/}")"
    if [[ "$(readlink -f "${EVAL_DATA_DIR}")" == "${root}" ]]; then
      echo "ERROR: EVAL_DATA_DIR is one of the training sources: ${root}" >&2
      echo "Use a separate synthetic evaluation dataset to avoid leakage." >&2
      exit 2
    fi
  done
fi

[[ -f "${PROJECT_DIR}/scripts/train/run_connected_subword_direct_job.sh" ]] || {
  echo "ERROR: missing Slurm-side training wrapper." >&2
  exit 2
}

unset PRETRAINED_WEIGHTS SYNTHETIC_WEIGHTS

export PROJECT_DIR JOB_ID DATA_ROOT DATA_DIR SYNTHETIC_DATA_DIRS
export SYNTHETIC_SAMPLES_PER_DIR SYNTHETIC_REQUIRE_FULL_PER_DIR
export HF_HOME NUM_SAMPLES EPOCHS NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE
export LEARNING_RATE WINDOW_SIZE STRIDE_RATIO TRAIN_SEED DATASET_SPLIT_SEED
export CONDA_ENV THRESHOLDS CALIBRATION_SAMPLES HOLDOUT_SAMPLES
export CALIBRATION_START_INDEX HOLDOUT_START_INDEX
export DIRECT_SUBWORD_BOX_SAFE_AUGMENT="${DIRECT_SUBWORD_BOX_SAFE_AUGMENT:-1}"
export DIRECT_SUBWORD_AUGMENT_PROBABILITY="${DIRECT_SUBWORD_AUGMENT_PROBABILITY:-0.85}"
export DIRECT_SUBWORD_CLEAN_PROBABILITY="${DIRECT_SUBWORD_CLEAN_PROBABILITY:-0.15}"
export DIRECT_SUBWORD_NOISE_STD_MAX="${DIRECT_SUBWORD_NOISE_STD_MAX:-10.0}"

TRAIN_JOB_RAW="$(
  sbatch --parsable \
    --job-name="${SLURM_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --tasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL \
    "${PROJECT_DIR}/scripts/train/run_connected_subword_direct_job.sh"
)"
TRAIN_JOB_ID="${TRAIN_JOB_RAW%%;*}"
[[ "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not parse training Slurm job ID from: ${TRAIN_JOB_RAW}" >&2
  exit 1
}

WEIGHTS="${PROJECT_DIR}/Weights/${JOB_ID}/model_latest.pth"
EVAL_JOB_ID=""
RESULTS_PATH="not scheduled"
if [[ -n "${EVAL_DATA_DIR}" ]]; then
  EVAL_OUTPUT="$(
    WEIGHTS="${WEIGHTS}" \
    DATA_DIR="${EVAL_DATA_DIR}" \
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
  EVAL_JOB_ID="$(awk '/^[0-9]+(;[^[:space:]]+)?$/ {sub(/;.*/, "", $1); print $1}' <<< "${EVAL_OUTPUT}" | tail -1)"
  RESULTS_PATH="${PROJECT_DIR}/Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/${JOB_ID}_calibrated_holdout"
fi

cat <<EOF
Submitted fully offline direct connected-subword training:
  training job       = ${TRAIN_JOB_ID}
  sources            = ${SYNTHETIC_DATA_DIRS}
  samples per source = ${SYNTHETIC_SAMPLES_PER_DIR}
  total samples      = ${NUM_SAMPLES}
  augmentation       = box-safe appearance/stroke
  checkpoint         = ${WEIGHTS}
  training log       = ${PROJECT_DIR}/out/${SLURM_JOB_NAME}_${TRAIN_JOB_ID}.out
  evaluation job     = ${EVAL_JOB_ID:-not scheduled}
  evaluation source  = ${EVAL_DATA_DIR:-not supplied}
  results            = ${RESULTS_PATH}
EOF
