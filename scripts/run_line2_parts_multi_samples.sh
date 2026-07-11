#!/bin/bash
set -euo pipefail

SCRIPT="scripts/visualize_line2_parts_in_line1.py"

WEIGHTS="Weights/span_jax_best_quality_win32_offline/model_latest.pth"
DATA_DIR="DataSet/Synthetic_Arabic"
OUT_DIR="Results/Evaluation/Part_Search_Multi"

START_INDEX=1
END_INDEX=10

PART_WIDTH=124
NUM_PARTS=3

WINDOW_SIZE=32
STRIDE=16
HEIGHT=128
THRESHOLD=0.85
MIN_RUN_LENGTH=3

mkdir -p "$OUT_DIR"

for IDX in $(seq "$START_INDEX" "$END_INDEX"); do
  echo "===================================================="
  echo "Running sample $IDX"
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
    --threshold "$THRESHOLD" \
    --min-run-length "$MIN_RUN_LENGTH"

done

echo "Done."
echo "Results saved in: $OUT_DIR"
