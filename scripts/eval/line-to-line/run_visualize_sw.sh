#!/bin/bash
set -euo pipefail

SCRIPT="scripts/eval/line-to-line/visualize_sw_longest_alignment.py"

WEIGHTS="${WEIGHTS:-Weights/span_jax_best_quality_win32_offline/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
INDICES="${INDICES:-1-10}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"
MATCH="${MATCH:-1.0}"

# For full line-to-line alignment the contextual CNN+BiLSTM representation is
# the default. Use EMBEDDING_SPACE=local only when you want to inspect the new
# local pre-BiLSTM embedding space directly.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-contextual}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-none}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-90}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-Results/Evaluation/SW_Longest_Sweep_${EMBEDDING_SPACE}}"

# Parameters to search
THRESHOLDS=(${THRESHOLDS:-0.84 0.85 0.86 0.87 0.88})
MISMATCHES=(${MISMATCHES:--1.2 -1.5 -1.8 -2.2})
GAPS=(${GAPS:--0.10 -0.15 -0.18 -0.25})
MIN_RUN_LENGTHS=(${MIN_RUN_LENGTHS:-6 8 10})

mkdir -p "$BASE_OUTPUT_DIR"

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

for THR in "${THRESHOLDS[@]}"; do
  for MISMATCH in "${MISMATCHES[@]}"; do
    for GAP in "${GAPS[@]}"; do
      for MIN_RUN in "${MIN_RUN_LENGTHS[@]}"; do

        THR_NAME=$(echo "$THR" | sed 's/\.//')
        MIS_NAME=$(echo "$MISMATCH" | sed 's/-/neg/; s/\.//')
        GAP_NAME=$(echo "$GAP" | sed 's/-/neg/; s/\.//')

        OUT_DIR="${BASE_OUTPUT_DIR}/emb${EMBEDDING_SPACE}_thr${THR_NAME}_mis${MIS_NAME}_gap${GAP_NAME}_min${MIN_RUN}"

        mkdir -p "$OUT_DIR"

        echo "===================================================="
        echo "Running:"
        echo "  embedding-space = $EMBEDDING_SPACE"
        echo "  threshold       = $THR"
        echo "  adaptive        = $ADAPTIVE_THRESHOLD"
        echo "  match           = $MATCH"
        echo "  mismatch        = $MISMATCH"
        echo "  gap             = $GAP"
        echo "  min-run-length  = $MIN_RUN"
        echo "  output          = $OUT_DIR"
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
          --threshold-percentile "$THRESHOLD_PERCENTILE" \
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
