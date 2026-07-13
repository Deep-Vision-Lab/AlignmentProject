#!/bin/bash
set -euo pipefail

SCRIPT="scripts/eval/line-to-part/visualize_line2_parts_in_line1.py"

WEIGHTS="${WEIGHTS:-Weights/improve_neg_win32_offline/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
OUT_DIR="${OUT_DIR:-Results/Evaluation/Part_Search_Multi_Local}"

START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-10}"

PART_WIDTH="${PART_WIDTH:-128}"
NUM_PARTS="${NUM_PARTS:-3}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"

# improve_neg recommendation:
# Use local pre-BiLSTM CNN embeddings for part/window matching.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-local}"

# Keep a fixed floor, but by default use per-part percentile thresholding because
# older checkpoints often have high similarity almost everywhere.
THRESHOLD="${THRESHOLD:-0.8}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-percentile}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-70}"

MATCH="${MATCH:-1.0}"
MISMATCH="${MISMATCH:--1.5}"
GAP="${GAP:--0.5}"
MIN_RUN_LENGTH="${MIN_RUN_LENGTH:-3}"

# Visual mask padding in window units.
# 0 = exact Smith-Waterman window span.
# 1 = expand by one window on both sides, useful when the mask looks shifted/tight.
MASK_PADDING_WINDOWS="${MASK_PADDING_WINDOWS:-1}"

# Save one cosine-similarity heatmap per chosen part.
# Enable with: HEATMAP=1 bash scripts/eval/line-to-part/run_line2_parts_multi_samples.sh
HEATMAP="${HEATMAP:-0}"
HEATMAP_DIR="${HEATMAP_DIR:-$OUT_DIR/heatmaps}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find script: $SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Cannot find weights: $WEIGHTS" >&2
  echo "Available .pth files:" >&2
  find . -name "*.pth" >&2 || true
  exit 1
fi

for IDX in $(seq "$START_INDEX" "$END_INDEX"); do
  EXTRA_ARGS=()
  if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
    EXTRA_ARGS+=(--heatmap --heatmap-dir "$HEATMAP_DIR")
  fi

  echo "===================================================="
  echo "Running sample $IDX"
  echo "  embedding-space      = $EMBEDDING_SPACE"
  echo "  threshold floor      = $THRESHOLD"
  echo "  adaptive-threshold   = $ADAPTIVE_THRESHOLD"
  echo "  threshold-percentile = $THRESHOLD_PERCENTILE"
  echo "  mask-padding-windows = $MASK_PADDING_WINDOWS"
  echo "  heatmap              = $HEATMAP"
  if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
    echo "  heatmap-dir          = $HEATMAP_DIR"
  fi
  echo "===================================================="

  python "$SCRIPT" \
    --data-dir "$DATA_DIR" \
    --index "$IDX" \
    --weights "$WEIGHTS" \
    --output "$OUT_DIR/sample_${IDX}_random_parts.png" \
    --part-width "$PART_WIDTH" \
    --num-parts "$NUM_PARTS" \
    --part-mode random \
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
    --mask-padding-windows "$MASK_PADDING_WINDOWS" \
    "${EXTRA_ARGS[@]}"

done

echo "Done."
echo "Results saved in: $OUT_DIR"
if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
  echo "Heatmaps saved in: $HEATMAP_DIR"
fi
