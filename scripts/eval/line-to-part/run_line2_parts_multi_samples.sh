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

# improve_model recommendation:
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

# Heatmap sliced-window display options.
# nonoverlap = show separated central/non-overlapping slices for readability.
# window     = show the full model windows, including overlap.
HEATMAP_AXIS_SLICE_MODE="${HEATMAP_AXIS_SLICE_MODE:-nonoverlap}"
HEATMAP_WINDOW_GAP_PIXELS="${HEATMAP_WINDOW_GAP_PIXELS:-10}"
HEATMAP_AXIS_CELL_PIXELS="${HEATMAP_AXIS_CELL_PIXELS:-42}"
HEATMAP_LINE1_STRIP_HEIGHT="${HEATMAP_LINE1_STRIP_HEIGHT:-84}"
HEATMAP_PART_STRIP_WIDTH="${HEATMAP_PART_STRIP_WIDTH:-96}"

# By default, flip the rotated part-window slices on the y-axis to make the side
# window thumbnails easier to read. Set to 0/false to disable.
HEATMAP_FLIP_PART_AXIS_WINDOWS="${HEATMAP_FLIP_PART_AXIS_WINDOWS:-1}"

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
    EXTRA_ARGS+=(
      --heatmap
      --heatmap-dir "$HEATMAP_DIR"
      --heatmap-axis-slice-mode "$HEATMAP_AXIS_SLICE_MODE"
      --heatmap-window-gap-pixels "$HEATMAP_WINDOW_GAP_PIXELS"
      --heatmap-axis-cell-pixels "$HEATMAP_AXIS_CELL_PIXELS"
      --heatmap-line1-strip-height "$HEATMAP_LINE1_STRIP_HEIGHT"
      --heatmap-part-strip-width "$HEATMAP_PART_STRIP_WIDTH"
    )

    if [[ "$HEATMAP_FLIP_PART_AXIS_WINDOWS" == "0" || "$HEATMAP_FLIP_PART_AXIS_WINDOWS" == "false" ]]; then
      EXTRA_ARGS+=(--no-heatmap-flip-part-axis-windows)
    fi
  fi

  echo "===================================================="
  echo "Running sample $IDX"
  echo "  embedding-space                 = $EMBEDDING_SPACE"
  echo "  threshold floor                 = $THRESHOLD"
  echo "  adaptive-threshold              = $ADAPTIVE_THRESHOLD"
  echo "  threshold-percentile            = $THRESHOLD_PERCENTILE"
  echo "  mask-padding-windows            = $MASK_PADDING_WINDOWS"
  echo "  heatmap                         = $HEATMAP"
  if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
    echo "  heatmap-dir                     = $HEATMAP_DIR"
    echo "  heatmap-axis-slice-mode         = $HEATMAP_AXIS_SLICE_MODE"
    echo "  heatmap-window-gap-pixels       = $HEATMAP_WINDOW_GAP_PIXELS"
    echo "  heatmap-axis-cell-pixels        = $HEATMAP_AXIS_CELL_PIXELS"
    echo "  heatmap-line1-strip-height      = $HEATMAP_LINE1_STRIP_HEIGHT"
    echo "  heatmap-part-strip-width        = $HEATMAP_PART_STRIP_WIDTH"
    echo "  heatmap-flip-part-axis-windows  = $HEATMAP_FLIP_PART_AXIS_WINDOWS"
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
