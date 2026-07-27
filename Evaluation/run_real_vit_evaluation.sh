#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash Evaluation/run_real_vit_evaluation.sh WEIGHTS_PATH [REAL_DATA_DIR] [N_SAMPLES] [OUTPUT_DIR]

Example:
  bash Evaluation/run_real_vit_evaluation.sh \
    Weights/synthetic_arabic_zero_shot_vit_8k/model_latest.pth \
    DataSet/ArabicDataset \
    50 \
    Results/Evaluation/SW/real_vit_test
EOF
}

WEIGHTS="${1:-${WEIGHTS:-}}"
DATA_DIR="${2:-${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}}"
N_SAMPLES="${3:-${N_SAMPLES:-50}}"
OUTPUT_DIR="${4:-${OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/SW/real_vit_test}}"
MANIFEST="${REAL_MANIFEST:-${DATA_DIR}/dataset_manifest.jsonl}"

if [[ -z "${WEIGHTS}" ]]; then
  usage
  exit 2
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: real dataset directory not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: real dataset manifest not found: ${MANIFEST}" >&2
  echo "Expected architecture: DataSet/ArabicDataset/dataset_manifest.jsonl" >&2
  exit 1
fi
if ! [[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_SAMPLES must be a positive integer, got: ${N_SAMPLES}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE

# Match real manuscript stroke size to the synthetic 128x1024 source domain.
# Foreground height remains stable; only overlong lines are compressed in x.
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

echo "===================================================="
echo "Real ArabicDataset ViT evaluation"
echo "  branch              = $(git branch --show-current 2>/dev/null || true)"
echo "  weights             = ${WEIGHTS}"
echo "  dataset             = ${DATA_DIR}"
echo "  manifest            = ${MANIFEST}"
echo "  split               = ${REAL_SPLIT:-test}"
echo "  labels              = ${REAL_LABELS:-high_match,medium_match}"
echo "  text key            = ${REAL_TEXT_KEY:-text_original_path}"
echo "  samples             = ${N_SAMPLES}"
echo "  target ink ratio    = ${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO}"
echo "  output              = ${OUTPUT_DIR}"
echo "===================================================="

exec python Evaluation/eval_img_align_sw.py \
  --weights "${WEIGHTS}" \
  --data-dir "${DATA_DIR}" \
  --arabic-manifest "${MANIFEST}" \
  --dataset-type real \
  --batch \
  --start-index "${START_INDEX:-1}" \
  --n-samples "${N_SAMPLES}" \
  --real-split "${REAL_SPLIT:-test}" \
  --real-labels "${REAL_LABELS:-high_match,medium_match}" \
  --real-text-key "${REAL_TEXT_KEY:-text_original_path}" \
  --real-min-text-score "${REAL_MIN_TEXT_SCORE:-0.0}" \
  --split-seed "${DATASET_SPLIT_SEED}" \
  --real-validate-paths \
  --feature "${FEATURE:-contextual}" \
  --score-mode "${SCORE_MODE:-auto}" \
  --score-clip "${SCORE_CLIP:-4.0}" \
  --threshold "${THRESHOLD:-0.0}" \
  --gap "${GAP:--0.30}" \
  --heatmap-source "${HEATMAP_SOURCE:-dp-score}" \
  --heatmap-value-decimals "${HEATMAP_VALUE_DECIMALS:-2}" \
  --heatmap-annotation-fontsize "${HEATMAP_ANNOTATION_FONTSIZE:-5.0}" \
  --no-save-binarized-images \
  --output-dir "${OUTPUT_DIR}"
