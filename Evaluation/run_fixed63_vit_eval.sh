#!/usr/bin/env bash
# Shared Slurm launcher for fixed-63 synthetic ViT evaluation.
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

EVAL_MODE="${EVAL_MODE:-nw}"
case "${EVAL_MODE}" in
  sw|nw|metrics) ;;
  *) echo "ERROR: EVAL_MODE must be sw, nw, or metrics." >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
DATA_DIR="${DATA_DIR:-${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset63}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
N_SAMPLES="${N_SAMPLES:-20}"
TEST_START="${TEST_START:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-raw}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"

case "${EVAL_MODE}" in
  sw)
    RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/SmithWaterman}"
    EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_vit_sw63}"
    ;;
  nw)
    RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_components_v2}"
    EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_vit_nw63_cmp2}"
    ;;
  metrics)
    RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/use_vit_encoder/Synthetic_Experiments/${RUN_ID}/NeedlemanWunsch_mask_metrics}"
    EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_vit_mask63}"
    ;;
esac

COUNT_LABEL="${N_SAMPLES}"
[[ "${N_SAMPLES}" == "0" ]] && COUNT_LABEL="all"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/test_start_${TEST_START}_count_${COUNT_LABEL}}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${OUTPUT_DIR}/selected_test_pairs.jsonl}"
SELECTED_INDICES="${SELECTED_INDICES:-${OUTPUT_DIR}/selected_test_indices.json}"

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
  echo "ERROR: fixed-63 dataset must contain images/, texts/, and masks/: ${DATA_DIR}" >&2
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
: "${WEIGHTS:?Set WEIGHTS, or place the ViT checkpoint under Weights/${RUN_ID}.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }

# Same component-v2 prediction rule used by the current CNN+BiLSTM evaluation.
NW_COMPONENT_SEED_SCORE="${NW_COMPONENT_SEED_SCORE:-0.22}"
NW_COMPONENT_SEED_MUTUAL_Z="${NW_COMPONENT_SEED_MUTUAL_Z:-0.25}"
NW_COMPONENT_SEED_PERCENTILE="${NW_COMPONENT_SEED_PERCENTILE:-0.82}"
NW_COMPONENT_SUPPORT_SCORE="${NW_COMPONENT_SUPPORT_SCORE:-0.04}"
NW_COMPONENT_SUPPORT_MUTUAL_Z="${NW_COMPONENT_SUPPORT_MUTUAL_Z:--0.10}"
NW_COMPONENT_SUPPORT_PERCENTILE="${NW_COMPONENT_SUPPORT_PERCENTILE:-0.62}"
NW_COMPONENT_MAX_PATH_GAP="${NW_COMPONENT_MAX_PATH_GAP:-3}"
NW_COMPONENT_MAX_WINDOW_GAP="${NW_COMPONENT_MAX_WINDOW_GAP:-2}"
NW_COMPONENT_MERGE_PATH_GAP="${NW_COMPONENT_MERGE_PATH_GAP:-3}"
NW_COMPONENT_MERGE_WINDOW_GAP="${NW_COMPONENT_MERGE_WINDOW_GAP:-2}"
NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-7}"
NW_COMPONENT_MIN_SPAN_WINDOWS="${NW_COMPONENT_MIN_SPAN_WINDOWS:-7}"
NW_COMPONENT_MIN_SPAN_FRACTION="${NW_COMPONENT_MIN_SPAN_FRACTION:-0.13}"
NW_COMPONENT_MIN_SEEDS="${NW_COMPONENT_MIN_SEEDS:-2}"
NW_COMPONENT_MIN_MEAN_SCORE="${NW_COMPONENT_MIN_MEAN_SCORE:-0.12}"
NW_COMPONENT_MIN_MEAN_MUTUAL_Z="${NW_COMPONENT_MIN_MEAN_MUTUAL_Z:-0.10}"
NW_COMPONENT_MIN_MEAN_PERCENTILE="${NW_COMPONENT_MIN_MEAN_PERCENTILE:-0.72}"
NW_COMPONENT_MIN_DENSITY="${NW_COMPONENT_MIN_DENSITY:-0.50}"
NW_COMPONENT_MIN_SPAN_BALANCE="${NW_COMPONENT_MIN_SPAN_BALANCE:-0.55}"
NW_COMPONENT_MIN_QUALITY="${NW_COMPONENT_MIN_QUALITY:-1.25}"
NW_COMPONENT_MIN_RELATIVE_QUALITY="${NW_COMPONENT_MIN_RELATIVE_QUALITY:-0.35}"
NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"
NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--0.05}"
NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE="${NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE:-0.16}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

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
  printf '%s\n' \
    "Fixed-63 synthetic ViT evaluation" \
    "  mode         = ${EVAL_MODE}" \
    "  branch       = $(git branch --show-current 2>/dev/null || true)" \
    "  checkpoint   = ${WEIGHTS}" \
    "  dataset      = ${DATA_DIR}" \
    "  split seed   = ${DATASET_SPLIT_SEED}" \
    "  test start   = ${TEST_START}" \
    "  samples      = ${N_SAMPLES} (0=all remaining test pairs)" \
    "  feature      = ${FEATURE}" \
    "  threshold    = ${THRESHOLD}" \
    "  gap          = ${GAP}" \
    "  output       = ${OUTPUT_DIR}"
}

if ! has_gpu_allocation; then
  print_config
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "Detected CPU-only Slurm context; submitting a separate GPU job."
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
    --export=ALL \
    "${SCRIPT_PATH}"
  exit 0
fi

command -v module >/dev/null 2>&1 && module load anaconda || true

resolve_python() {
  local candidate conda_base=""
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
  return 1
}

EVAL_PYTHON="$(resolve_python)"
CONDA_ENV_PYTHON="${EVAL_PYTHON}"
"${EVAL_PYTHON}" - <<'PY'
import sys, torch
print(f"Evaluation Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated CUDA GPU")
PY

mkdir -p "${OUTPUT_DIR}"
SELECTED_COUNT="$("${EVAL_PYTHON}" Evaluation/fixed63_test_manifest.py \
  --data-dir "${DATA_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --split-seed "${DATASET_SPLIT_SEED}" \
  --test-start "${TEST_START}" \
  --n-samples "${N_SAMPLES}" \
  --manifest "${PAIR_MANIFEST}" \
  --indices "${SELECTED_INDICES}" | tail -1)"

SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
TOKENIZERS_PARALLELISM=false

print_config
echo "Selected held-out test pairs: ${SELECTED_COUNT}"

case "${EVAL_MODE}" in
  sw)
    "${EVAL_PYTHON}" -m Evaluation.eval_img_align_sw \
      --weights "${WEIGHTS}" \
      --device cuda \
      --data-dir "${DATA_DIR}" \
      --dataset-type synthetic \
      --pair-manifest "${PAIR_MANIFEST}" \
      --batch \
      --start-index 1 \
      --n-samples "${SELECTED_COUNT}" \
      --feature "${FEATURE}" \
      --score-mode "${SCORE_MODE}" \
      --score-clip "${SCORE_CLIP}" \
      --threshold "${THRESHOLD}" \
      --gap "${GAP}" \
      --heatmap-source "${HEATMAP_SOURCE}" \
      --no-save-binarized-images \
      --output-dir "${OUTPUT_DIR}"
    ;;
  nw)
    "${EVAL_PYTHON}" Evaluation/vit_fixed63_nw_eval.py \
      --weights "${WEIGHTS}" \
      --pair-manifest "${PAIR_MANIFEST}" \
      --output-dir "${OUTPUT_DIR}" \
      --device cuda \
      --feature "${FEATURE}" \
      --score-mode "${SCORE_MODE}" \
      --score-clip "${SCORE_CLIP}" \
      --threshold "${THRESHOLD}" \
      --gap "${GAP}"
    ;;
  metrics)
    metric_args=()
    [[ "${SAVE_PREDICTED_MASKS:-0}" == "1" ]] && metric_args+=(--save-predicted-masks)
    "${EVAL_PYTHON}" Evaluation/vit_fixed63_nw_eval.py \
      --weights "${WEIGHTS}" \
      --pair-manifest "${PAIR_MANIFEST}" \
      --output-dir "${OUTPUT_DIR}" \
      --device cuda \
      --feature "${FEATURE}" \
      --score-mode "${SCORE_MODE}" \
      --score-clip "${SCORE_CLIP}" \
      --threshold "${THRESHOLD}" \
      --gap "${GAP}" \
      --metrics-only \
      "${metric_args[@]}"
    ;;
esac

printf '%s\n' \
  "ViT evaluation finished." \
  "  output       = ${OUTPUT_DIR}" \
  "  test indices = ${SELECTED_INDICES}"
