#!/usr/bin/env bash
# Evaluate precision/recall/IoU/Dice/F1 on the exact fixed-63 synthetic test split.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: [WEIGHTS=<checkpoint>] bash Evaluation/evaluate_synthetic_fixed63_mask_metrics.sh" >&2
  echo "Optional: RUN_ID, DATA_DIR, N_SAMPLES (0=all test), TEST_START, OUTPUT_DIR." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"
DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
N_SAMPLES="${N_SAMPLES:-0}"
TEST_START="${TEST_START:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-raw}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
SAVE_PREDICTED_MASKS="${SAVE_PREDICTED_MASKS:-0}"
COUNT_LABEL="${N_SAMPLES}"
[[ "${N_SAMPLES}" == "0" ]] && COUNT_LABEL="all"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/training_speed_optimization/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_mask_metrics/test_start_${TEST_START}_count_${COUNT_LABEL}}"

for name in NUM_SAMPLES N_SAMPLES TEST_START DATASET_SPLIT_SEED; do
  value="${!name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: ${name} must be a non-negative integer." >&2
    exit 2
  }
done
(( NUM_SAMPLES > 0 && TEST_START > 0 )) || {
  echo "ERROR: NUM_SAMPLES and TEST_START must be greater than zero." >&2
  exit 2
}

DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" && -d "${DATA_DIR}/masks" ]] || {
  echo "ERROR: dataset must contain images/, texts/, and masks/: ${DATA_DIR}" >&2
  exit 2
}

if [[ -z "${WEIGHTS:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/Weights/${RUN_ID}/model_best.pth" \
    "${PROJECT_DIR}/Weights/${RUN_ID}/model_latest.pth" \
    "${PROJECT_DIR}/Weights/${RUN_ID}/checkpoint_latest.pth"; do
    if [[ -f "${candidate}" ]]; then
      WEIGHTS="${candidate}"
      break
    fi
  done
fi
: "${WEIGHTS:?Set WEIGHTS, or place the checkpoint under Weights/${RUN_ID}.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

# Keep metric prediction identical to the current component-aware NW evaluator.
export NW_COMPONENT_SEED_SCORE="${NW_COMPONENT_SEED_SCORE:-0.22}"
export NW_COMPONENT_SEED_MUTUAL_Z="${NW_COMPONENT_SEED_MUTUAL_Z:-0.25}"
export NW_COMPONENT_SEED_PERCENTILE="${NW_COMPONENT_SEED_PERCENTILE:-0.82}"
export NW_COMPONENT_SUPPORT_SCORE="${NW_COMPONENT_SUPPORT_SCORE:-0.04}"
export NW_COMPONENT_SUPPORT_MUTUAL_Z="${NW_COMPONENT_SUPPORT_MUTUAL_Z:--0.10}"
export NW_COMPONENT_SUPPORT_PERCENTILE="${NW_COMPONENT_SUPPORT_PERCENTILE:-0.62}"
export NW_COMPONENT_MAX_PATH_GAP="${NW_COMPONENT_MAX_PATH_GAP:-3}"
export NW_COMPONENT_MAX_WINDOW_GAP="${NW_COMPONENT_MAX_WINDOW_GAP:-2}"
export NW_COMPONENT_MERGE_PATH_GAP="${NW_COMPONENT_MERGE_PATH_GAP:-3}"
export NW_COMPONENT_MERGE_WINDOW_GAP="${NW_COMPONENT_MERGE_WINDOW_GAP:-2}"
export NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-7}"
export NW_COMPONENT_MIN_SPAN_WINDOWS="${NW_COMPONENT_MIN_SPAN_WINDOWS:-7}"
export NW_COMPONENT_MIN_SPAN_FRACTION="${NW_COMPONENT_MIN_SPAN_FRACTION:-0.13}"
export NW_COMPONENT_MIN_SEEDS="${NW_COMPONENT_MIN_SEEDS:-2}"
export NW_COMPONENT_MIN_MEAN_SCORE="${NW_COMPONENT_MIN_MEAN_SCORE:-0.12}"
export NW_COMPONENT_MIN_MEAN_MUTUAL_Z="${NW_COMPONENT_MIN_MEAN_MUTUAL_Z:-0.10}"
export NW_COMPONENT_MIN_MEAN_PERCENTILE="${NW_COMPONENT_MIN_MEAN_PERCENTILE:-0.72}"
export NW_COMPONENT_MIN_DENSITY="${NW_COMPONENT_MIN_DENSITY:-0.50}"
export NW_COMPONENT_MIN_SPAN_BALANCE="${NW_COMPONENT_MIN_SPAN_BALANCE:-0.55}"
export NW_COMPONENT_MIN_QUALITY="${NW_COMPONENT_MIN_QUALITY:-1.25}"
export NW_COMPONENT_MIN_RELATIVE_QUALITY="${NW_COMPONENT_MIN_RELATIVE_QUALITY:-0.35}"
export NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"
export NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--0.05}"
export NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE="${NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE:-0.16}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_trainopt_maskmetrics}"

has_gpu_allocation() {
  local name value
  for name in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_GPU_INDEX; do
    value="${!name:-}"
    if [[ -n "${value}" && "${value}" != "NoDevFiles" && "${value}" != "(null)" ]]; then
      return 0
    fi
  done
  return 1
}

print_config() {
  local sample_text="${N_SAMPLES}"
  [[ "${N_SAMPLES}" == "0" ]] && sample_text="all remaining held-out test pairs"
  printf '%s\n' \
    "Synthetic fixed-63 alignment-mask metrics" \
    "  branch       = $(git branch --show-current 2>/dev/null || true)" \
    "  checkpoint   = ${WEIGHTS}" \
    "  dataset      = ${DATA_DIR}" \
    "  split        = exact 60/20/20 test split" \
    "  split seed   = ${DATASET_SPLIT_SEED}" \
    "  test start   = ${TEST_START}" \
    "  samples      = ${sample_text}" \
    "  metrics      = precision, recall, IoU, Dice, F1" \
    "  metric space = horizontal mask columns" \
    "  feature      = ${FEATURE}" \
    "  output       = ${OUTPUT_DIR}"
}

if ! has_gpu_allocation; then
  print_config
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Detected CPU-only Slurm context; submitting a separate GPU metrics job."
  fi
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --error="${PROJECT_DIR}/out/%x_%J.err" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",WEIGHTS="${WEIGHTS}",DATA_DIR="${DATA_DIR}",RUN_ID="${RUN_ID}",OUTPUT_DIR="${OUTPUT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi

resolve_python() {
  local conda_base=""
  local candidate=""
  local -a candidates=()
  [[ -n "${CONDA_ENV_PYTHON:-}" ]] && candidates+=("${CONDA_ENV_PYTHON}")
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  candidates+=(
    "${HOME}/.conda/envs/${CONDA_ENV}/bin/python"
    "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
    "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python"
  )
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    [[ -n "${conda_base}" ]] && candidates+=("${conda_base}/envs/${CONDA_ENV}/bin/python")
  fi
  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import torch' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  echo "ERROR: could not find a Python executable with PyTorch for ${CONDA_ENV}." >&2
  printf '  %s\n' "${candidates[@]}" >&2
  echo "Set CONDA_ENV_PYTHON to the working environment's bin/python path." >&2
  return 1
}

EVAL_PYTHON="$(resolve_python)"
export CONDA_ENV_PYTHON="${EVAL_PYTHON}"
"${EVAL_PYTHON}" - <<'PY'
import sys
import torch
print(f"Evaluation Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated CUDA GPU")
PY

mkdir -p "${OUTPUT_DIR}"
export SYNTHETIC_MANUSCRIPT_AUGMENT=0
export REAL_AUGMENT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

print_config
ARGS=(
  --weights "${WEIGHTS}"
  --data-dir "${DATA_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --split-seed "${DATASET_SPLIT_SEED}"
  --test-start "${TEST_START}"
  --n-samples "${N_SAMPLES}"
  --device cuda
  --feature "${FEATURE}"
  --score-mode "${SCORE_MODE}"
  --score-clip "${SCORE_CLIP}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${SAVE_PREDICTED_MASKS}" == "1" ]]; then
  ARGS+=(--save-predicted-masks)
fi

"${EVAL_PYTHON}" -m Evaluation.evaluate_synthetic_mask_metrics "${ARGS[@]}"

printf '%s\n' \
  "Synthetic mask metrics finished." \
  "  per-sample CSV = ${OUTPUT_DIR}/mask_metrics.csv" \
  "  summary JSON   = ${OUTPUT_DIR}/mask_metrics_summary.json" \
  "  test indices   = ${OUTPUT_DIR}/selected_test_indices.json"
