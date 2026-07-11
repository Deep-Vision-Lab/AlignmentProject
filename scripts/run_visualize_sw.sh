#!/bin/bash
set -euo pipefail

SCRIPT="scripts/visualize_sw_longest_alignment.py"

WEIGHTS="Weights/span_jax_best_quality_win32_offline/model_latest.pth"
DATA_DIR="DataSet/Synthetic_Arabic"
INDICES="1-10"

WINDOW_SIZE=32
STRIDE=16
HEIGHT=128
MATCH=1.0

BASE_OUTPUT_DIR="Results/Evaluation/SW_Longest_Sweep"

# Parameters to search
THRESHOLDS=(0.84 0.85 0.86 0.87 0.88)
MISMATCHES=(-1.2 -1.5 -1.8 -2.2)
GAPS=(-0.10 -0.15 -0.18 -0.25)
MIN_RUN_LENGTHS=(6 8 10)

mkdir -p "$BASE_OUTPUT_DIR"

for THR in "${THRESHOLDS[@]}"; do
  for MISMATCH in "${MISMATCHES[@]}"; do
    for GAP in "${GAPS[@]}"; do
      for MIN_RUN in "${MIN_RUN_LENGTHS[@]}"; do

        THR_NAME=$(echo "$THR" | sed 's/\.//')
        MIS_NAME=$(echo "$MISMATCH" | sed 's/-/neg/; s/\.//')
        GAP_NAME=$(echo "$GAP" | sed 's/-/neg/; s/\.//')

        OUT_DIR="${BASE_OUTPUT_DIR}/thr${THR_NAME}_mis${MIS_NAME}_gap${GAP_NAME}_min${MIN_RUN}"

        mkdir -p "$OUT_DIR"

        echo "===================================================="
        echo "Running:"
        echo "  threshold      = $THR"
        echo "  match          = $MATCH"
        echo "  mismatch       = $MISMATCH"
        echo "  gap            = $GAP"
        echo "  min-run-length = $MIN_RUN"
        echo "  output         = $OUT_DIR"
        echo "===================================================="

        python "$SCRIPT" \
          --batch \
          --data-dir "$DATA_DIR" \
          --indices "$INDICES" \
          --weights "$WEIGHTS" \
          --output-dir "$OUT_DIR" \
          --window-size "$WINDOW_SIZE" \
          --stride "$STRIDE" \
          --height "$HEIGHT" \
          --threshold "$THR" \
          --match "$MATCH" \
          --mismatch "$MISMATCH" \
          --gap "$GAP" \
          --min-run-length "$MIN_RUN" \
          --use-flip \
          --heatmap

      done
    done
  done
done

echo "All runs finished."
echo "Results saved under: $BASE_OUTPUT_DIR"