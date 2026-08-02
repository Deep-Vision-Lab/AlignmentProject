#!/usr/bin/env bash
# Submit direct synthetic training fully offline, then queue evaluation after it.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

JOB_ID="${JOB_ID:-cnn_connected_subword_direct_synthetic}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
NUM_SAMPLES="${NUM_SAMPLES:-8000}"
EPOCHS="${EPOCHS:-35}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
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

[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}
[[ -f "${PROJECT_DIR}/scripts/train/run_connected_subword_direct_job.sh" ]] || {
  echo "ERROR: missing Slurm-side training wrapper." >&2
  exit 2
}

unset PRETRAINED_WEIGHTS SYNTHETIC_WEIGHTS

# Export the complete experiment configuration before sbatch. The Slurm-side
# wrapper builds/validates sidecars and then starts training in the same job.
export PROJECT_DIR JOB_ID DATA_DIR HF_HOME NUM_SAMPLES EPOCHS NUM_GPUS
export EFFECTIVE_GLOBAL_BATCH_SIZE LEARNING_RATE WINDOW_SIZE STRIDE_RATIO
export TRAIN_SEED DATASET_SPLIT_SEED CONDA_ENV
export THRESHOLDS CALIBRATION_SAMPLES HOLDOUT_SAMPLES
export CALIBRATION_START_INDEX HOLDOUT_START_INDEX

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

echo "Submitted offline training job: ${TRAIN_JOB_ID}"

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

EVAL_JOB_ID="$(awk '/^[0-9]+(;[^[:space:]]+)?$/ {sub(/;.*/, "", $1); print $1}' <<< "${EVAL_OUTPUT}" | tail -1)"

cat <<EOF
Submitted fully offline direct connected-subword pipeline:
  training job   = ${TRAIN_JOB_ID}
  evaluation job = ${EVAL_JOB_ID:-unknown} (afterok:${TRAIN_JOB_ID})
  checkpoint     = ${WEIGHTS}
  training log   = ${PROJECT_DIR}/out/${SLURM_JOB_NAME}_${TRAIN_JOB_ID}.out
  results        = ${PROJECT_DIR}/Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/${JOB_ID}_calibrated_holdout
EOF
