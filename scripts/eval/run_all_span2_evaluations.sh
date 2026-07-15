#!/usr/bin/env bash
#
# Run all recommended evaluation visualizations for the new span2 weights.
#
# Default usage:
#   bash scripts/eval/run_all_span2_evaluations.sh
#
# Override examples:
#   WEIGHTS=Weights/<JOB_ID>/model_latest.pth bash scripts/eval/run_all_span2_evaluations.sh
#   INDEX=5 START_INDEX=5 END_INDEX=5 bash scripts/eval/run_all_span2_evaluations.sh
#   START_INDEX=1 END_INDEX=20 bash scripts/eval/run_all_span2_evaluations.sh
#

set -euo pipefail

WEIGHTS="${WEIGHTS:-Weights/improve_model_win32_fastpair_span2/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
INDEX="${INDEX:-1}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-10}"

FEATURE_SPACE="${FEATURE_SPACE:-local}"
EMBEDDING_SPACE="${EMBEDDING_SPACE:-local}"
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-local}"

MAX_SPAN_CHARS="${MAX_SPAN_CHARS:-2}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-2}"
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY:-0.05}"
GROUP_POOLING="${GROUP_POOLING:-mean}"

mkdir -p Results/Evaluation

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: Cannot find weights: ${WEIGHTS}" >&2
  echo "Available .pth files:" >&2
  find . -name "*.pth" >&2 || true
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: Cannot find DATA_DIR: ${DATA_DIR}" >&2
  exit 1
fi

echo "===================================================="
echo "Running all span2 evaluations"
echo "  WEIGHTS              = ${WEIGHTS}"
echo "  DATA_DIR             = ${DATA_DIR}"
echo "  INDEX                = ${INDEX}"
echo "  START_INDEX          = ${START_INDEX}"
echo "  END_INDEX            = ${END_INDEX}"
echo "  FEATURE_SPACE        = ${FEATURE_SPACE}"
echo "  EMBEDDING_SPACE      = ${EMBEDDING_SPACE}"
echo "  ALIGNMENT_SPACE      = ${ALIGNMENT_SPACE}"
echo "  MAX_SPAN_CHARS       = ${MAX_SPAN_CHARS}"
echo "  MAX_WINDOWS_PER_SPAN = ${MAX_WINDOWS_PER_SPAN}"
echo "  WINDOW_COUNT_PENALTY = ${WINDOW_COUNT_PENALTY}"
echo "  GROUP_POOLING        = ${GROUP_POOLING}"
echo "===================================================="

echo ""
echo "[1/7] Line-to-line SW visualization"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${DATA_DIR}" \
EMBEDDING_SPACE="${EMBEDDING_SPACE}" \
THRESHOLD_PERCENTILE=90 \
MISMATCH=-3.0 \
GAP=-0.8 \
MIN_RUN_LENGTH=4 \
bash scripts/eval/line-to-line/run_visualize_sw.sh

echo ""
echo "[2/7] Line2 parts inside line1 with heatmaps"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${DATA_DIR}" \
EMBEDDING_SPACE="${EMBEDDING_SPACE}" \
THRESHOLD=0.85 \
ADAPTIVE_THRESHOLD=percentile \
THRESHOLD_PERCENTILE=95 \
MISMATCH=-4.0 \
GAP=-1.0 \
MIN_RUN_LENGTH=4 \
MASK_PADDING_WINDOWS=0 \
HEATMAP=1 \
START_INDEX="${START_INDEX}" \
END_INDEX="${END_INDEX}" \
bash scripts/eval/line-to-part/run_line2_parts_multi_samples.sh

echo ""
echo "[3/7] Window Grad-CAM / feature concentration for line1"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${DATA_DIR}" \
FEATURE_SPACE="${FEATURE_SPACE}" \
ALIGNMENT_SPACE="${ALIGNMENT_SPACE}" \
MAX_SPAN_CHARS="${MAX_SPAN_CHARS}" \
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN}" \
INDEX="${INDEX}" \
WHICH_LINE=1 \
bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh

echo ""
echo "[4/7] Window Grad-CAM / feature concentration for line2"
WEIGHTS="${WEIGHTS}" \
DATA_DIR="${DATA_DIR}" \
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
DATA_DIR="${DATA_DIR}" \
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
DATA_DIR="${DATA_DIR}" \
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
DATA_DIR="${DATA_DIR}" \
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
echo "Done. Evaluation outputs are under Results/Evaluation/."
