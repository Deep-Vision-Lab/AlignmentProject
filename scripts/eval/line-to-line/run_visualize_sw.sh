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
MATCH="${MATCH:-1.0}"

# For full line-to-line alignment, contextual CNN+BiLSTM embeddings are still the
# best default. Use EMBEDDING_SPACE=local only when debugging local windows.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-contextual}"

# Recommended search parameters after observing high similarities everywhere.
# Percentile thresholding is safer than a fixed threshold because every sample can
# have a different similarity distribution.
ADAPTIVE_THRESHOLDS=(${ADAPTIVE_THRESHOLDS:-percentile})
THRESHOLDS=(${THRESHOLDS:-0.8})
THRESHOLD_PERCENTILES=(${THRESHOLD_PERCENTILES:-85 90 95})
MISMATCHES=(${MISMATCHES:--2.5 -3.0 -4.0})
GAPS=(${GAPS:--0.5 -0.8 -1.0})
MIN_RUN_LENGTHS=(${MIN_RUN_LENGTHS:-4 6})

# Arabic/right-to-left setup used by this project.
USE_FLIP="${USE_FLIP:-1}"

# Heatmaps help diagnose why a high single-cell similarity did not become a good
# consecutive SW segment. Set HEATMAP=0 to disable.
HEATMAP="${HEATMAP:-1}"
STRICT="${STRICT:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-Results/Evaluation/SW_Longest_Recommended_${EMBEDDING_SPACE}}"
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

for ADAPTIVE_THRESHOLD in "${ADAPTIVE_THRESHOLDS[@]}"; do
  for THR in "${THRESHOLDS[@]}"; do
    for PCT in "${THRESHOLD_PERCENTILES[@]}"; do
      for MISMATCH in "${MISMATCHES[@]}"; do
        for GAP in "${GAPS[@]}"; do
          for MIN_RUN in "${MIN_RUN_LENGTHS[@]}"; do

            THR_NAME=$(echo "$THR" | sed 's/\.//')
            PCT_NAME=$(echo "$PCT" | sed 's/\.//')
            MIS_NAME=$(echo "$MISMATCH" | sed 's/-/neg/; s/\.//')
            GAP_NAME=$(echo "$GAP" | sed 's/-/neg/; s/\.//')

            OUT_DIR="${BASE_OUTPUT_DIR}/emb${EMBEDDING_SPACE}_${ADAPTIVE_THRESHOLD}_pct${PCT_NAME}_thr${THR_NAME}_mis${MIS_NAME}_gap${GAP_NAME}_min${MIN_RUN}"
            mkdir -p "$OUT_DIR"

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

            echo "===================================================="
            echo "Running line-to-line SW visualization"
            echo "  embedding-space       = $EMBEDDING_SPACE"
            echo "  adaptive-threshold    = $ADAPTIVE_THRESHOLD"
            echo "  threshold floor       = $THR"
            echo "  threshold-percentile  = $PCT"
            echo "  match                 = $MATCH"
            echo "  mismatch              = $MISMATCH"
            echo "  gap                   = $GAP"
            echo "  min-run-length        = $MIN_RUN"
            echo "  use-flip              = $USE_FLIP"
            echo "  output                = $OUT_DIR"
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
              --embedding-space "$EMBEDDING_SPACE" \
              --threshold "$THR" \
              --adaptive-threshold "$ADAPTIVE_THRESHOLD" \
              --threshold-percentile "$PCT" \
              --match "$MATCH" \
              --mismatch "$MISMATCH" \
              --gap "$GAP" \
              --min-run-length "$MIN_RUN" \
              "${EXTRA_ARGS[@]}"

          done
        done
      done
    done
  done
done

echo "All runs finished."
echo "Results saved under: $BASE_OUTPUT_DIR"
