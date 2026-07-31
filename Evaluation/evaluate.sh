#!/usr/bin/env bash
# The only public real-data evaluation command.
# Run from the login node with:
#   WEIGHTS=Weights/<job_id>/model_best.pth bash Evaluation/evaluate.sh
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate.sh" >&2
  echo "Configure optional settings through environment variables." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to model_best.pth or model_latest.pth.}"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

MODEL_TAG="${MODEL_TAG:-$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
LABELS="${LABELS:-high_match,medium_match}"
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
REAL_SPLIT="${REAL_SPLIT:-test}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${MODEL_TAG}/Real_Experiments/${RUN_TAG}}"

[[ -d "${REAL_DATA_DIR}" ]] || {
  echo "ERROR: real dataset directory not found: ${REAL_DATA_DIR}" >&2
  exit 2
}
[[ -f "${ARABIC_MANIFEST}" ]] || {
  echo "ERROR: manifest not found: ${ARABIC_MANIFEST}" >&2
  exit 2
}
[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: N_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${START_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: START_INDEX must be a non-negative integer." >&2
  exit 2
}

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_${MODEL_TAG}_${RUN_TAG}}"

print_config() {
  printf '%s\n' \
    "Canonical real evaluation" \
    "  branch       = $(git branch --show-current)" \
    "  model        = ${MODEL_TAG}" \
    "  run          = ${RUN_TAG}" \
    "  checkpoint   = ${WEIGHTS}" \
    "  split        = ${REAL_SPLIT}" \
    "  labels       = ${LABELS}" \
    "  samples      = ${N_SAMPLES}" \
    "  results root = ${RESULTS_ROOT}"
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
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Keep evaluation preprocessing identical to training. Model window/stride and
# backend are reconstructed from the checkpoint configuration.
export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_EVAL_BALANCED="${REAL_EVAL_BALANCED:-1}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

print_config
IFS=',' read -r -a LABEL_ARRAY <<< "${LABELS}"
for LABEL in "${LABEL_ARRAY[@]}"; do
  LABEL="${LABEL//[[:space:]]/}"
  [[ -n "${LABEL}" ]] || continue
  OUTPUT_DIR="${RESULTS_ROOT}/${LABEL}"
  mkdir -p "${OUTPUT_DIR}"

  printf '%s\n' \
    "Evaluating label=${LABEL}" \
    "  output=${OUTPUT_DIR}"

  python -m Evaluation.eval_img_align_sw \
    --weights "${WEIGHTS}" \
    --device cuda \
    --data-dir "${REAL_DATA_DIR}" \
    --arabic-manifest "${ARABIC_MANIFEST}" \
    --dataset-type real \
    --batch \
    --real-split "${REAL_SPLIT}" \
    --real-labels "${LABEL}" \
    --real-text-key "${REAL_TEXT_KEY}" \
    --real-min-text-score "${REAL_MIN_TEXT_SCORE}" \
    --start-index "${START_INDEX}" \
    --n-samples "${N_SAMPLES}" \
    --feature "${FEATURE}" \
    --score-mode "${SCORE_MODE}" \
    --score-clip "${SCORE_CLIP}" \
    --threshold "${THRESHOLD}" \
    --gap "${GAP}" \
    --heatmap-source "${HEATMAP_SOURCE}" \
    --no-save-binarized-images \
    --output-dir "${OUTPUT_DIR}"
done
