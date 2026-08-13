#!/usr/bin/env bash
# Held-out real Arabic image-to-text retrieval diagnostic. No training is performed.
set -euo pipefail

[[ "$#" -eq 0 ]] || { echo "Usage: WEIGHTS=<checkpoint> [RUN_TAG=<tag>] bash Evaluation/evaluate_real_image_text_retrieval.sh" >&2; exit 2; }
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to the checkpoint to diagnose.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }
REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}")"
SOURCE_MANIFEST="$(readlink -f "${SOURCE_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}")"
[[ -f "${SOURCE_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${SOURCE_MANIFEST}" >&2; exit 2; }

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")")_image_text_retrieval}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/ImageText_Retrieval/${RUN_TAG}}"
N_PAIRS="${N_PAIRS:-60}"
NUM_NEGATIVES="${NUM_NEGATIVES:-50}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-16}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
NEGATIVE_SEED="${NEGATIVE_SEED:-42}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
DEVICE="${DEVICE:-auto}"

[[ "${N_PAIRS}" =~ ^[0-9]+$ ]] || { echo "ERROR: N_PAIRS must be a non-negative integer." >&2; exit 2; }
[[ "${NUM_NEGATIVES}" =~ ^[0-9]+$ ]] || { echo "ERROR: NUM_NEGATIVES must be a non-negative integer." >&2; exit 2; }
[[ "${CANDIDATE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: CANDIDATE_BATCH_SIZE must be positive." >&2; exit 2; }

# Exact held-out real preprocessing used by training/validation.
export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_DATASET_LABELS="high_match,medium_match"
export DATASET_TYPE=real
export DATASET_SPLIT_SEED NEGATIVE_SEED REAL_TEXT_KEY

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-image_text_retrieval}"

export PROJECT_DIR WEIGHTS REAL_DATA_DIR SOURCE_MANIFEST RUN_TAG RESULTS_ROOT
export N_PAIRS NUM_NEGATIVES CANDIDATE_BATCH_SIZE DEVICE CONDA_ENV

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting image-text retrieval diagnostic: ${RESULTS_ROOT}"
  sbatch --job-name="${EVAL_JOB_NAME}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" --time="${TIME_LIMIT}" --export=ALL,PROJECT_DIR="${PROJECT_DIR}" "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${RESULTS_ROOT}"

{
  echo "date=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "weights=${WEIGHTS}"
  echo "manifest=${SOURCE_MANIFEST}"
  echo "n_pairs=${N_PAIRS}"
  echo "num_negatives=${NUM_NEGATIVES}"
  echo "candidate_batch_size=${CANDIDATE_BATCH_SIZE}"
  echo "dataset_split_seed=${DATASET_SPLIT_SEED}"
  echo "negative_seed=${NEGATIVE_SEED}"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
} | tee "${RESULTS_ROOT}/experiment_config.txt"

python -m Evaluation.eval_real_image_text_retrieval \
  --weights "${WEIGHTS}" --data-dir "${REAL_DATA_DIR}" --manifest "${SOURCE_MANIFEST}" \
  --output-dir "${RESULTS_ROOT}" --device "${DEVICE}" --n-pairs "${N_PAIRS}" \
  --num-negatives "${NUM_NEGATIVES}" --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --split-seed "${DATASET_SPLIT_SEED}" --negative-seed "${NEGATIVE_SEED}" --text-key "${REAL_TEXT_KEY}" \
  2>&1 | tee "${RESULTS_ROOT}/run.log"

echo "=== Image-text retrieval summary ==="
cat "${RESULTS_ROOT}/summary.json"
