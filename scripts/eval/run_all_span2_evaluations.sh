#!/usr/bin/env bash
# Run the improve_neg evaluation suite on its training dataset or real Arabic data.
#
# Trained-on synthetic dataset (default):
#   bash scripts/eval/run_all_span2_evaluations.sh
#
# Binarized real manifest dataset with black background / white text:
#   DATASET_TYPE=real bash scripts/eval/run_all_span2_evaluations.sh
#
# Common overrides:
#   WEIGHTS=Weights/<JOB_ID>/model_latest.pth START_INDEX=1 END_INDEX=10 ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET_TYPE="${DATASET_TYPE:-synthetic}"
case "${DATASET_TYPE}" in
  synthetic)
    SOURCE_DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
    ;;
  real)
    SOURCE_DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
    ;;
  *)
    echo "ERROR: DATASET_TYPE must be synthetic or real, got: ${DATASET_TYPE}" >&2
    exit 2
    ;;
esac

# This is the checkpoint and dataset used by the improve_neg fast image-pair run.
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/improve_model_win32_fastpair/model_latest.pth}"
INDEX="${INDEX:-1}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-10}"

FEATURE_SPACE="${FEATURE_SPACE:-local}"
EMBEDDING_SPACE="${EMBEDDING_SPACE:-local}"
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-local}"
MAX_SPAN_CHARS="${MAX_SPAN_CHARS:-3}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY:-0.05}"
GROUP_POOLING="${GROUP_POOLING:-mean}"

# Real-data filters and image preprocessing.
REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest.jsonl}"
REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"
REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
# Final requested polarity after binarization: black background, white text.
REAL_INVERT_COLORS="${REAL_INVERT_COLORS:-1}"
REAL_EVAL_HEIGHT="${REAL_EVAL_HEIGHT:-128}"
REAL_EVAL_WIDTH="${REAL_EVAL_WIDTH:-1024}"
REAL_LINK_MODE="${REAL_LINK_MODE:-auto}"

if [[ "${DATASET_TYPE}" == "real" ]]; then
  if is_true "${REAL_BINARIZE}"; then
    REAL_OUTPUT_TAG="real_binarized_${REAL_BINARIZE_METHOD}"
  else
    REAL_OUTPUT_TAG="real_resized_grayscale"
  fi

  if is_true "${REAL_INVERT_COLORS}"; then
    REAL_OUTPUT_TAG+="_black_bg_white_text"
  else
    REAL_OUTPUT_TAG+="_white_bg_black_text"
  fi
else
  REAL_OUTPUT_TAG="synthetic"
fi

REAL_EVAL_VIEW_DIR="${REAL_EVAL_VIEW_DIR:-${PROJECT_DIR}/Results/Evaluation/dataset_views/${REAL_OUTPUT_TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/${REAL_OUTPUT_TAG}}"

MAX_REQUESTED_INDEX="${END_INDEX}"
if (( INDEX > MAX_REQUESTED_INDEX )); then
  MAX_REQUESTED_INDEX="${INDEX}"
fi

PREPARE_ARGS=(
  --dataset-type "${DATASET_TYPE}"
  --data-dir "${SOURCE_DATA_DIR}"
  --max-samples "${MAX_REQUESTED_INDEX}"
)
if [[ "${DATASET_TYPE}" == "real" ]]; then
  PREPARE_ARGS+=(
    --output-dir "${REAL_EVAL_VIEW_DIR}"
    --manifest-name "${REAL_MANIFEST_NAME}"
    --labels "${REAL_DATASET_LABELS}"
    --min-text-score "${REAL_MIN_TEXT_SCORE}"
    --text-key "${REAL_TEXT_KEY}"
    --link-mode "${REAL_LINK_MODE}"
    --binarize-method "${REAL_BINARIZE_METHOD}"
    --binarize-threshold "${REAL_BINARIZE_THRESHOLD}"
    --height "${REAL_EVAL_HEIGHT}"
    --width "${REAL_EVAL_WIDTH}"
  )
  if is_true "${REAL_BINARIZE}"; then
    PREPARE_ARGS+=(--binarize-real)
  else
    PREPARE_ARGS+=(--no-binarize-real)
  fi
  if is_true "${REAL_BINARIZE_AUTOCONTRAST}"; then
    PREPARE_ARGS+=(--binarize-autocontrast)
  else
    PREPARE_ARGS+=(--no-binarize-autocontrast)
  fi
  if is_true "${REAL_BINARIZE_AUTO_INVERT}"; then
    PREPARE_ARGS+=(--binarize-auto-invert)
  else
    PREPARE_ARGS+=(--no-binarize-auto-invert)
  fi
  if is_true "${REAL_INVERT_COLORS}"; then
    PREPARE_ARGS+=(--invert-real-colors)
  else
    PREPARE_ARGS+=(--no-invert-real-colors)
  fi
fi

RESOLVED_DATA_DIR="$(python scripts/eval/prepare_eval_dataset_view.py "${PREPARE_ARGS[@]}")"
mkdir -p "${RESULTS_ROOT}"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: Cannot find weights: ${WEIGHTS}" >&2
  echo "Available .pth files:" >&2
  find "${PROJECT_DIR}/Weights" -name "*.pth" 2>/dev/null >&2 || true
  exit 1
fi

# The evaluation suite runs offline, but the line-to-part, feature-concentration,
# and span-group scripts need the frozen text backbone. Resolve a complete local
# snapshot from the checkpoint and common cache locations, then pass that exact
# directory to every child evaluation script.
if [[ -z "${ARABIC_TEXT_MODEL_NAME:-}" ]]; then
  export ARABIC_TEXT_MODEL_NAME="$(
    python scripts/eval/resolve_cached_text_model.py \
      --weights "${WEIGHTS}" \
      --project-dir "${PROJECT_DIR}"
  )"
fi

for numeric_name in INDEX START_INDEX END_INDEX; do
  value="${!numeric_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "ERROR: ${numeric_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
if (( START_INDEX > END_INDEX )); then
  echo "ERROR: START_INDEX (${START_INDEX}) cannot exceed END_INDEX (${END_INDEX})" >&2
  exit 2
fi

echo "===================================================="
echo "Running improve_neg evaluation suite"
echo "  DATASET_TYPE        = ${DATASET_TYPE}"
echo "  SOURCE_DATA_DIR     = ${SOURCE_DATA_DIR}"
echo "  RESOLVED_DATA_DIR   = ${RESOLVED_DATA_DIR}"
echo "  WEIGHTS             = ${WEIGHTS}"
echo "  TEXT_MODEL          = ${ARABIC_TEXT_MODEL_NAME:-checkpoint-char-encoder}"
echo "  INDEX               = ${INDEX}"
echo "  START_INDEX         = ${START_INDEX}"
echo "  END_INDEX           = ${END_INDEX}"
echo "  RESULTS_ROOT        = ${RESULTS_ROOT}"
echo "  FEATURE_SPACE       = ${FEATURE_SPACE}"
echo "  EMBEDDING_SPACE     = ${EMBEDDING_SPACE}"
echo "  ALIGNMENT_SPACE     = ${ALIGNMENT_SPACE}"
echo "  MAX_SPAN_CHARS      = ${MAX_SPAN_CHARS}"
echo "  MAX_WINDOWS_PER_SPAN= ${MAX_WINDOWS_PER_SPAN}"
if [[ "${DATASET_TYPE}" == "real" ]]; then
  echo "  REAL_LABELS         = ${REAL_DATASET_LABELS}"
  echo "  REAL_TEXT_KEY       = ${REAL_TEXT_KEY}"
  echo "  REAL_MIN_TEXT_SCORE = ${REAL_MIN_TEXT_SCORE}"
  echo "  REAL_BINARIZE       = ${REAL_BINARIZE}"
  echo "  REAL_BIN_METHOD     = ${REAL_BINARIZE_METHOD}"
  echo "  REAL_BIN_THRESHOLD  = ${REAL_BINARIZE_THRESHOLD}"
  echo "  REAL_AUTOCONTRAST   = ${REAL_BINARIZE_AUTOCONTRAST}"
  echo "  REAL_AUTO_INVERT    = ${REAL_BINARIZE_AUTO_INVERT}"
  echo "  REAL_INVERT_COLORS  = ${REAL_INVERT_COLORS}"
  if is_true "${REAL_INVERT_COLORS}"; then
    echo "  REAL_FINAL_POLARITY = black background / white text"
  else
    echo "  REAL_FINAL_POLARITY = white background / black text"
  fi
  echo "  REAL_IMAGE_SIZE     = ${REAL_EVAL_HEIGHT}x${REAL_EVAL_WIDTH}"
fi
echo "===================================================="

echo ""
echo "[1/7] Line-to-line SW visualization"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
BASE_OUTPUT_DIR="${RESULTS_ROOT}/line_to_line" \
INDICES="${START_INDEX}-${END_INDEX}" \
EMBEDDING_SPACE="${EMBEDDING_SPACE}" \
THRESHOLD_PERCENTILE=80 \
MISMATCH=-3.0 \
GAP=-0.8 \
MIN_RUN_LENGTH=4 \
bash scripts/eval/line-to-line/run_visualize_sw.sh

echo ""
echo "[2/7] Line2 parts inside line1 with heatmaps"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/line_to_part" \
HEATMAP_DIR="${RESULTS_ROOT}/line_to_part/heatmaps" \
EMBEDDING_SPACE="${EMBEDDING_SPACE}" \
ALIGNMENT_SPACE="${ALIGNMENT_SPACE}" \
THRESHOLD=0.5 \
ADAPTIVE_THRESHOLD=percentile \
THRESHOLD_PERCENTILE=80 \
MISMATCH=-4.0 \
GAP=-1.0 \
MIN_RUN_LENGTH=4 \
MASK_PADDING_WINDOWS=0 \
HEATMAP=1 \
MAX_SPAN_CHARS="${MAX_SPAN_CHARS}" \
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN}" \
START_INDEX="${START_INDEX}" \
END_INDEX="${END_INDEX}" \
bash scripts/eval/line-to-part/run_line2_parts_multi_samples.sh

echo ""
echo "[3/7] Window feature concentration for line1"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/feature_concentration" \
FEATURE_SPACE="${FEATURE_SPACE}" \
ALIGNMENT_SPACE="${ALIGNMENT_SPACE}" \
MAX_SPAN_CHARS="${MAX_SPAN_CHARS}" \
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN}" \
INDEX="${INDEX}" \
WHICH_LINE=1 \
bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh

echo ""
echo "[4/7] Window feature concentration for line2"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/feature_concentration" \
FEATURE_SPACE="${FEATURE_SPACE}" \
ALIGNMENT_SPACE="${ALIGNMENT_SPACE}" \
MAX_SPAN_CHARS="${MAX_SPAN_CHARS}" \
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN}" \
INDEX="${INDEX}" \
WHICH_LINE=2 \
bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh

echo ""
echo "[5/7] Self-window cosine for line1"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/self_window_cosine" \
FEATURE_SPACE="${FEATURE_SPACE}" \
USE_FLIP=1 \
REVERSE_X_AXIS=1 \
CELL_VALUES=1 \
INDEX="${INDEX}" \
WHICH_LINE=1 \
bash scripts/eval/window-similarity/run_line_self_window_cosine.sh

echo ""
echo "[6/7] Self-window cosine for line2"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/self_window_cosine" \
FEATURE_SPACE="${FEATURE_SPACE}" \
USE_FLIP=1 \
REVERSE_X_AXIS=1 \
CELL_VALUES=1 \
INDEX="${INDEX}" \
WHICH_LINE=2 \
bash scripts/eval/window-similarity/run_line_self_window_cosine.sh

echo ""
echo "[7/7] Line1-line2 span-group cosine for samples ${START_INDEX}-${END_INDEX}"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${RESOLVED_DATA_DIR}" \
OUT_DIR="${RESULTS_ROOT}/line_pair_cosine" \
COMPARISON_MODE=span_group \
FEATURE_SPACE="${FEATURE_SPACE}" \
ALIGNMENT_SPACE="${ALIGNMENT_SPACE}" \
MAX_SPAN_CHARS="${MAX_SPAN_CHARS}" \
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN}" \
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY}" \
GROUP_POOLING="${GROUP_POOLING}" \
SHOW_AXIS_TOKENS=1 \
CELL_VALUES=1 \
START_INDEX="${START_INDEX}" \
END_INDEX="${END_INDEX}" \
bash scripts/eval/window-similarity/run_line_pair_window_cosine.sh

echo ""
echo "Done. Evaluation outputs: ${RESULTS_ROOT}"
if [[ "${DATASET_TYPE}" == "real" ]]; then
  echo "Processed real images: ${RESOLVED_DATA_DIR}/images"
  echo "Real preprocessing metadata: ${RESOLVED_DATA_DIR}/view_metadata.json"
  echo "Real sample-to-pair mapping: ${RESOLVED_DATA_DIR}/view_manifest.jsonl"
fi
