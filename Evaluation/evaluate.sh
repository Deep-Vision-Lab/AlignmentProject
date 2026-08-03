#!/usr/bin/env bash
# Offline real-data quantitative evaluation from Excel subword boxes.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to a model checkpoint.}"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }

MODEL_TAG="${MODEL_TAG:-legacy_main}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
LABELS="${LABELS:-high_match,medium_match}"
REAL_SPLIT="${REAL_SPLIT:-test}"
SPLIT_SEED="${SPLIT_SEED:-42}"
START_INDEX="${START_INDEX:-1}"
N_SAMPLES="${N_SAMPLES:-100}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"
REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-0}"
REAL_BOX_IN_MASK_RULE="${REAL_BOX_IN_MASK_RULE:-center}"
REAL_BOX_MIN_COVERAGE="${REAL_BOX_MIN_COVERAGE:-0.50}"
REAL_BOX_BBOX_FORMAT="${REAL_BOX_BBOX_FORMAT:-xyxy}"
REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Real_Boxes/${RUN_TAG}}"

[[ -f "${ARABIC_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2; exit 2; }
[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: N_SAMPLES must be positive." >&2; exit 2; }

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_${MODEL_TAG}_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Legacy real Excel-box evaluation" \
    "  branch       = $(git branch --show-current)" \
    "  checkpoint   = ${WEIGHTS}" \
    "  manifest     = ${ARABIC_MANIFEST}" \
    "  split        = ${REAL_SPLIT}" \
    "  labels       = ${LABELS}" \
    "  samples      = ${N_SAMPLES}" \
    "  box rule     = ${REAL_BOX_IN_MASK_RULE}" \
    "  box root     = ${REAL_BOX_ANNOTATIONS_ROOT:-auto-near-line-image}" \
    "  results      = ${RESULTS_ROOT}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --tasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export REAL_BOX_IN_MASK_RULE REAL_BOX_MIN_COVERAGE REAL_BOX_BBOX_FORMAT

mkdir -p "${RESULTS_ROOT}"
print_config
ARGS=(
  --weights "${WEIGHTS}"
  --arabic-manifest "${ARABIC_MANIFEST}"
  --output-dir "${RESULTS_ROOT}"
  --labels "${LABELS}"
  --split "${REAL_SPLIT}"
  --split-seed "${SPLIT_SEED}"
  --start-index "${START_INDEX}"
  --n-samples "${N_SAMPLES}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
  --annotation-root "${REAL_BOX_ANNOTATIONS_ROOT}"
)
if [[ "${REAL_REQUIRE_BOX_ANNOTATIONS}" == "1" ]]; then
  ARGS+=(--require-annotations)
fi
python -m Evaluation.eval_real_subword_boxes "${ARGS[@]}"
