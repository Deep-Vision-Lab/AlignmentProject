#!/usr/bin/env bash
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
ROOT="$(readlink -f "${ROOT}")"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_TYPE="${DATASET_TYPE:?Set DATASET_TYPE=synthetic or real}"
case "${DATASET_TYPE}" in synthetic|real) ;; *) echo "ERROR: DATASET_TYPE must be synthetic or real" >&2; exit 2 ;; esac

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
resolve_python() {
  local candidate conda_base=""
  local -a candidates=()
  [[ -n "${CONDA_ENV_PYTHON:-}" ]] && candidates+=("${CONDA_ENV_PYTHON}")
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
  echo "ERROR: no Python with PyTorch found for ${CONDA_ENV}" >&2
  return 1
}
PYTHON_BIN="$(resolve_python)"

MODEL_NAME="$(${PYTHON_BIN} -c "import sys; sys.path.insert(0, r'${ROOT}'); import model_backend; print(model_backend.MODEL_NAME)")"
case "${MODEL_NAME}" in
  cnn_bilstm)
    RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"
    MODEL_RESULT_TAG="training_speed_optimization"
    ;;
  vit)
    RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
    MODEL_RESULT_TAG="use_vit_encoder"
    ;;
  *)
    RUN_ID="${RUN_ID:-${MODEL_NAME}}"
    MODEL_RESULT_TAG="${MODEL_NAME}"
    ;;
esac

if [[ -n "${WEIGHTS:-}" && ! -f "${WEIGHTS}" ]]; then
  echo "WARNING: ignoring missing WEIGHTS=${WEIGHTS}" >&2
  unset WEIGHTS
fi
if [[ -z "${WEIGHTS:-}" ]]; then
  for candidate in \
    "${ROOT}/Weights/${RUN_ID}/model_best.pth" \
    "${ROOT}/Weights/${RUN_ID}/model_latest.pth" \
    "${ROOT}/Weights/${RUN_ID}/checkpoint_latest.pth"; do
    if [[ -f "${candidate}" ]]; then WEIGHTS="${candidate}"; break; fi
  done
fi
: "${WEIGHTS:?No checkpoint found under Weights/${RUN_ID}; set WEIGHTS explicitly.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"

N_SAMPLES="${N_SAMPLES:-20}"
N_ALIGNED_CONTROLS="${N_ALIGNED_CONTROLS:-5}"
START_INDEX="${START_INDEX:-1}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SEED="${SEED:-42}"
FEATURE="${FEATURE:-contextual}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
REAL_SPLIT="${REAL_SPLIT:-test}"

if [[ "${DATASET_TYPE}" == "synthetic" ]]; then
  DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/AugmentedArabicDataset63}"
  SCORE_MODE="${SCORE_MODE:-raw}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/Evaluation/${MODEL_RESULT_TAG}/Synthetic_Experiments/${RUN_ID}/Unaligned_with_aligned_controls/test_neg_${N_SAMPLES}_ctrl_${N_ALIGNED_CONTROLS}}"
else
  DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
  ARABIC_MANIFEST="${ARABIC_MANIFEST:-${DATA_DIR}/dataset_manifest.jsonl}"
  SCORE_MODE="${SCORE_MODE:-auto}"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/Results/Evaluation/${MODEL_RESULT_TAG}/Real_Experiments/${RUN_ID}/Unaligned_with_aligned_controls/test_neg_${N_SAMPLES}_ctrl_${N_ALIGNED_CONTROLS}}"
fi

[[ -d "${DATA_DIR}" ]] || { echo "ERROR: data directory not found: ${DATA_DIR}" >&2; exit 2; }
mkdir -p "${ROOT}/out"

PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_${MODEL_NAME}_sanity_${DATASET_TYPE}}"

has_gpu_allocation() {
  local value name
  for name in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_GPU_INDEX; do
    value="${!name:-}"
    if [[ -n "${value}" && "${value}" != "NoDevFiles" && "${value}" != "(null)" ]]; then return 0; fi
  done
  return 1
}

print_config() {
  printf '%s\n' \
    "Mixed aligned/unaligned sanity evaluation" \
    "  branch             = $(git branch --show-current 2>/dev/null || true)" \
    "  model              = ${MODEL_NAME}" \
    "  checkpoint         = ${WEIGHTS}" \
    "  dataset            = ${DATASET_TYPE}" \
    "  unaligned pairs    = ${N_SAMPLES}" \
    "  aligned controls   = ${N_ALIGNED_CONTROLS}" \
    "  same predictor     = Evaluation.evaluate_unaligned_pairs._evaluate_one" \
    "  output             = ${OUTPUT_DIR}" \
    "  slurm logs         = ${ROOT}/out"
}

if ! has_gpu_allocation; then
  print_config
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${ROOT}/out/%x_%J.out" \
    --error="${ROOT}/out/%x_%J.err" \
    --chdir="${ROOT}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${ROOT}",CONDA_ENV_PYTHON="${PYTHON_BIN}",PYTHONPATH="${PYTHONPATH}" \
    "${SCRIPT_PATH}"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Allocated job cannot see CUDA")
print(f"PyTorch={torch.__version__}; CUDA={torch.cuda.is_available()}")
PY

export SYNTHETIC_MANUSCRIPT_AUGMENT=0
export REAL_AUGMENT=0
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--1000000.0}"

mkdir -p "${OUTPUT_DIR}"
print_config
ARGS=(
  --weights "${WEIGHTS}"
  --dataset-type "${DATASET_TYPE}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --n-samples "${N_SAMPLES}"
  --n-aligned-controls "${N_ALIGNED_CONTROLS}"
  --start-index "${START_INDEX}"
  --num-samples "${NUM_SAMPLES}"
  --split-seed "${SPLIT_SEED}"
  --seed "${SEED}"
  --real-split "${REAL_SPLIT}"
  --device cuda
  --feature "${FEATURE}"
  --score-mode "${SCORE_MODE}"
  --score-clip "${SCORE_CLIP}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
)
if [[ "${DATASET_TYPE}" == "real" ]]; then ARGS+=(--arabic-manifest "${ARABIC_MANIFEST}"); fi

"${PYTHON_BIN}" -m Evaluation.evaluate_unaligned_with_controls "${ARGS[@]}"
printf '%s\n' \
  "Sanity evaluation finished." \
  "  images       = ${OUTPUT_DIR}/pair_*.png" \
  "  samples CSV  = ${OUTPUT_DIR}/samples.csv" \
  "  summary JSON = ${OUTPUT_DIR}/summary.json"
