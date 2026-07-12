#!/bin/bash
set -euo pipefail

SCRIPT="scripts/eval/line-to-line/visualize_sw_longest_alignment.py"

# Default to the improve_neg checkpoint. Override from the shell if needed:
#   WEIGHTS=Weights/other_run/model_latest.pth bash scripts/eval/line-to-line/run_visualize_sw.sh
WEIGHTS="${WEIGHTS:-Weights/improve_neg_win32_offline/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
INDICES="${INDICES:-1-10}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"

# Full line-to-line alignment usually works better with contextual CNN+BiLSTM
# embeddings. Use EMBEDDING_SPACE=local only when debugging local windows.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-contextual}"

# Current recommended default after observing high similarities everywhere.
# Use percentile thresholding so each sample gets its own threshold.
THRESHOLD="${THRESHOLD:-0.8}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-percentile}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-85}"
THRESHOLD_STD_SCALE="${THRESHOLD_STD_SCALE:-1.0}"

MATCH="${MATCH:-1.0}"
MISMATCH="${MISMATCH:--2.5}"
GAP="${GAP:--0.5}"
MIN_RUN_LENGTH="${MIN_RUN_LENGTH:-3}"

# Arabic/right-to-left setup used by this project.
USE_FLIP="${USE_FLIP:-1}"

# Heatmaps help diagnose cases where a high single-cell similarity does not
# become a good consecutive SW segment. Set HEATMAP=0 to disable.
HEATMAP="${HEATMAP:-1}"
STRICT="${STRICT:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-Results/Evaluation/SW_Longest_${EMBEDDING_SPACE}_pct${THRESHOLD_PERCENTILE}_thr${THRESHOLD}_mis${MISMATCH}_gap${GAP}_min${MIN_RUN_LENGTH}}"
mkdir -p "$BASE_OUTPUT_DIR"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find script: $SCRIPT" >&2
  echo "Run this command from the repository root." >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Cannot find weights: $WEIGHTS" >&2
  echo "Available .pth files:" >&2
  find . -name "*.pth" >&2 || true
  exit 1
fi

EXTRA_ARGS=()
if [[ "$USE_FLIP" == "1" || "$USE_FLIP" == "true" ]]; then
  EXTRA_ARGS+=(--use-flip)
fi
if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
  EXTRA_ARGS+=(--heatmap)
fi
if [[ "$STRICT" == "1" || "$STRICT" == "true" ]]; then
  EXTRA_ARGS+=(--strict)
fi

# Only pass the std-scale argument when mean_std thresholding is used. This keeps
# the command compatible with the percentile/default mode while preserving the
# option for debugging.
if [[ "$ADAPTIVE_THRESHOLD" == "mean_std" ]]; then
  EXTRA_ARGS+=(--threshold-std-scale "$THRESHOLD_STD_SCALE")
fi

echo "===================================================="
echo "Running line-to-line SW visualization"
echo "  embedding-space       = $EMBEDDING_SPACE"
echo "  threshold floor       = $THRESHOLD"
echo "  adaptive-threshold    = $ADAPTIVE_THRESHOLD"
echo "  threshold-percentile  = $THRESHOLD_PERCENTILE"
echo "  match                 = $MATCH"
echo "  mismatch              = $MISMATCH"
echo "  gap                   = $GAP"
echo "  min-run-length        = $MIN_RUN_LENGTH"
echo "  use-flip              = $USE_FLIP"
echo "  heatmap               = $HEATMAP"
echo "  output                = $BASE_OUTPUT_DIR"
echo "===================================================="

python "$SCRIPT" \
  --batch \
  --data-dir "$DATA_DIR" \
  --indices "$INDICES" \
  --weights "$WEIGHTS" \
  --output-dir "$BASE_OUTPUT_DIR" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --height "$HEIGHT" \
  --embedding-space "$EMBEDDING_SPACE" \
  --threshold "$THRESHOLD" \
  --adaptive-threshold "$ADAPTIVE_THRESHOLD" \
  --threshold-percentile "$THRESHOLD_PERCENTILE" \
  --match "$MATCH" \
  --mismatch "$MISMATCH" \
  --gap "$GAP" \
  --min-run-length "$MIN_RUN_LENGTH" \
  "${EXTRA_ARGS[@]}"

echo "Done."
echo "Results saved under: $BASE_OUTPUT_DIR"
