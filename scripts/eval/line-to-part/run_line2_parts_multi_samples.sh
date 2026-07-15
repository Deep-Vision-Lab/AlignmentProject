#!/bin/bash
set -euo pipefail

# Default to a thin wrapper that keeps the original line-to-part logic but draws
# the heatmap y-axis with the same style as visualize_line_self_window_cosine.py.
SCRIPT="${SCRIPT:-scripts/eval/line-to-part/visualize_line2_parts_in_line1_self_yaxis.py}"

WEIGHTS="${WEIGHTS:-Weights/improve_model_win32_fastpair_span2/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
OUT_DIR="${OUT_DIR:-Results/Part_Search_Multi_Local}"

START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-1}"

PART_WIDTH="${PART_WIDTH:-128}"
NUM_PARTS="${NUM_PARTS:-3}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"

# Use the same RTL flip setting used by the Arabic training/eval pipeline.
# This affects the model window order. The heatmap display can still be shown in
# readable visual order using HEATMAP_DISPLAY_ORDER=visual.
USE_FLIP="${USE_FLIP:-0}"
NO_BILSTM="${NO_BILSTM:-0}"

# improve_model recommendation:
# Use local pre-BiLSTM CNN embeddings for part/window matching.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-local}"
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-contextual}"

# Stricter defaults than the old script. The previous 80th percentile with a 0.6
# floor allowed many repeated Arabic strokes to become false positives.
THRESHOLD="${THRESHOLD:-0.8}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-percentile}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-80}"

MATCH="${MATCH:-3.0}"
MISMATCH="${MISMATCH:--4.0}"
GAP="${GAP:--1.0}"
MIN_RUN_LENGTH="${MIN_RUN_LENGTH:-4}"

# Visual mask padding in window units.
# 0 = exact Smith-Waterman window span.
MASK_PADDING_WINDOWS="${MASK_PADDING_WINDOWS:-1}"

# Save one cosine-similarity heatmap per chosen part.
# Enable with: HEATMAP=1 bash scripts/eval/line-to-part/run_line2_parts_multi_samples.sh
HEATMAP="${HEATMAP:-1}"
HEATMAP_DIR="${HEATMAP_DIR:-$OUT_DIR/heatmaps}"

# Heatmap sliced-window display options.
# visual = show the x/y axes in the same visual order as the readable image.
# model  = show raw model order.
HEATMAP_DISPLAY_ORDER="${HEATMAP_DISPLAY_ORDER:-visual}"
HEATMAP_AXIS_SLICE_MODE="${HEATMAP_AXIS_SLICE_MODE:-nonoverlap}"

# Match visualize_line_self_window_cosine.py axis defaults.
HEATMAP_WINDOW_GAP_PIXELS="${HEATMAP_WINDOW_GAP_PIXELS:-12}"
HEATMAP_AXIS_CELL_PIXELS="${HEATMAP_AXIS_CELL_PIXELS:-52}"
HEATMAP_LINE1_STRIP_HEIGHT="${HEATMAP_LINE1_STRIP_HEIGHT:-84}"
HEATMAP_PART_STRIP_WIDTH="${HEATMAP_PART_STRIP_WIDTH:-108}"

# Independent axis reversal, same idea as visualize_line_self_window_cosine.py.
# These reverse the displayed heatmap axis, thumbnails, model-id ticks, token labels,
# SW path, and selected red cells together so alignment remains correct.
HEATMAP_REVERSE_X_AXIS="${HEATMAP_REVERSE_X_AXIS:-0}"
HEATMAP_REVERSE_Y_AXIS="${HEATMAP_REVERSE_Y_AXIS:-1}"

HEATMAP_Y_AXIS_ROTATE="${HEATMAP_Y_AXIS_ROTATE:-1}"
HEATMAP_Y_AXIS_FLIP="${HEATMAP_Y_AXIS_FLIP:-0}"
export HEATMAP_REVERSE_X_AXIS HEATMAP_REVERSE_Y_AXIS HEATMAP_Y_AXIS_ROTATE HEATMAP_Y_AXIS_FLIP

# Axis-token labels are inferred by hard Span-DTW and written on the heatmap axes.
# For this experiment, keep labels at most 2 chars and at most 2 windows per span.
SHOW_AXIS_TOKENS="${SHOW_AXIS_TOKENS:-1}"
AXIS_TOKEN_FONTSIZE="${AXIS_TOKEN_FONTSIZE:-6.5}"
X_TOKEN_ROTATION="${X_TOKEN_ROTATION:-90}"
Y_TOKEN_ROTATION="${Y_TOKEN_ROTATION:-0}"
MAX_SPAN_CHARS="${MAX_SPAN_CHARS:-2}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-2}"
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY:-0.05}"
TOKEN_TEMPERATURE="${TOKEN_TEMPERATURE:-0.07}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-}"

# Same y-axis behavior as visualize_line_self_window_cosine.py: rotated by default,
# not mirrored and not extra-flipped unless explicitly requested.
HEATMAP_MIRROR_LINE1_AXIS_WINDOWS="${HEATMAP_MIRROR_LINE1_AXIS_WINDOWS:-0}"
HEATMAP_MIRROR_PART_AXIS_WINDOWS="${HEATMAP_MIRROR_PART_AXIS_WINDOWS:-0}"

# Backward-compatible old variable. 1 means mirror the part thumbnails before rotation.
HEATMAP_FLIP_PART_AXIS_WINDOWS="${HEATMAP_FLIP_PART_AXIS_WINDOWS:-$HEATMAP_MIRROR_PART_AXIS_WINDOWS}"

HEATMAP_CELL_VALUES="${HEATMAP_CELL_VALUES:-1}"
HEATMAP_CELL_VALUE_FONTSIZE="${HEATMAP_CELL_VALUE_FONTSIZE:-4.2}"
HEATMAP_MARK_ABOVE_THRESHOLD="${HEATMAP_MARK_ABOVE_THRESHOLD:-1}"

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

  if [[ "$USE_FLIP" == "1" || "$USE_FLIP" == "true" ]]; then
    EXTRA_ARGS+=(--use-flip)
  fi
  if [[ "$NO_BILSTM" == "1" || "$NO_BILSTM" == "true" ]]; then
    EXTRA_ARGS+=(--no-bilstm)
  fi
  if [[ -n "$TEXT_ENCODER_TYPE" ]]; then
    EXTRA_ARGS+=(--text-encoder-type "$TEXT_ENCODER_TYPE")
  fi
  if [[ -n "$ARABIC_TEXT_MODEL_NAME" ]]; then
    EXTRA_ARGS+=(--arabic-text-model-name "$ARABIC_TEXT_MODEL_NAME")
  fi

  if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
    EXTRA_ARGS+=(
      --heatmap
      --heatmap-dir "$HEATMAP_DIR"
      --heatmap-display-order "$HEATMAP_DISPLAY_ORDER"
      --heatmap-axis-slice-mode "$HEATMAP_AXIS_SLICE_MODE"
      --heatmap-window-gap-pixels "$HEATMAP_WINDOW_GAP_PIXELS"
      --heatmap-axis-cell-pixels "$HEATMAP_AXIS_CELL_PIXELS"
      --heatmap-line1-strip-height "$HEATMAP_LINE1_STRIP_HEIGHT"
      --heatmap-part-strip-width "$HEATMAP_PART_STRIP_WIDTH"
    )
    if [[ "$HEATMAP_MIRROR_LINE1_AXIS_WINDOWS" == "1" || "$HEATMAP_MIRROR_LINE1_AXIS_WINDOWS" == "true" ]]; then
      EXTRA_ARGS+=(--heatmap-mirror-line1-axis-windows)
    fi
    if [[ "$HEATMAP_MIRROR_PART_AXIS_WINDOWS" == "1" || "$HEATMAP_MIRROR_PART_AXIS_WINDOWS" == "true" || "$HEATMAP_FLIP_PART_AXIS_WINDOWS" == "1" || "$HEATMAP_FLIP_PART_AXIS_WINDOWS" == "true" ]]; then
      EXTRA_ARGS+=(--heatmap-mirror-part-axis-windows)
    else
      EXTRA_ARGS+=(--no-heatmap-flip-part-axis-windows)
    fi
    if [[ "$HEATMAP_CELL_VALUES" == "0" || "$HEATMAP_CELL_VALUES" == "false" ]]; then
      EXTRA_ARGS+=(--no-heatmap-cell-values)
    else
      EXTRA_ARGS+=(--heatmap-cell-value-fontsize "$HEATMAP_CELL_VALUE_FONTSIZE")
    fi
    if [[ "$HEATMAP_MARK_ABOVE_THRESHOLD" == "0" || "$HEATMAP_MARK_ABOVE_THRESHOLD" == "false" ]]; then
      EXTRA_ARGS+=(--no-heatmap-mark-above-threshold)
    fi
    if [[ "$SHOW_AXIS_TOKENS" == "0" || "$SHOW_AXIS_TOKENS" == "false" ]]; then
      EXTRA_ARGS+=(--no-axis-tokens)
    else
      EXTRA_ARGS+=(
        --axis-token-fontsize "$AXIS_TOKEN_FONTSIZE"
        --x-token-rotation "$X_TOKEN_ROTATION"
        --y-token-rotation "$Y_TOKEN_ROTATION"
        --max-span-chars "$MAX_SPAN_CHARS"
        --max-windows-per-span "$MAX_WINDOWS_PER_SPAN"
        --window-count-penalty "$WINDOW_COUNT_PENALTY"
        --token-temperature "$TOKEN_TEMPERATURE"
      )
    fi
  fi

  echo "===================================================="
  echo "Running sample $IDX"
  echo "  weights                         = $WEIGHTS"
  echo "  embedding-space                 = $EMBEDDING_SPACE"
  echo "  alignment-space                 = $ALIGNMENT_SPACE"
  echo "  use-flip                        = $USE_FLIP"
  echo "  threshold floor                 = $THRESHOLD"
  echo "  adaptive-threshold              = $ADAPTIVE_THRESHOLD"
  echo "  threshold-percentile            = $THRESHOLD_PERCENTILE"
  echo "  mask-padding-windows            = $MASK_PADDING_WINDOWS"
  echo "  heatmap                         = $HEATMAP"
  if [[ "$HEATMAP" == "1" || "$HEATMAP" == "true" ]]; then
    echo "  heatmap-dir                     = $HEATMAP_DIR"
    echo "  heatmap-display-order           = $HEATMAP_DISPLAY_ORDER"
    echo "  heatmap-reverse-x-axis          = $HEATMAP_REVERSE_X_AXIS"
    echo "  heatmap-reverse-y-axis          = $HEATMAP_REVERSE_Y_AXIS"
    echo "  heatmap-axis-slice-mode         = $HEATMAP_AXIS_SLICE_MODE"
    echo "  heatmap-window-gap-pixels       = $HEATMAP_WINDOW_GAP_PIXELS"
    echo "  heatmap-axis-cell-pixels        = $HEATMAP_AXIS_CELL_PIXELS"
    echo "  heatmap-y-axis-rotate           = $HEATMAP_Y_AXIS_ROTATE"
    echo "  heatmap-y-axis-flip             = $HEATMAP_Y_AXIS_FLIP"
    echo "  show-axis-tokens                = $SHOW_AXIS_TOKENS"
    echo "  max-span-chars                  = $MAX_SPAN_CHARS"
    echo "  max-windows-per-span            = $MAX_WINDOWS_PER_SPAN"
    echo "  mirror-line1-axis-windows       = $HEATMAP_MIRROR_LINE1_AXIS_WINDOWS"
    echo "  mirror-part-axis-windows        = $HEATMAP_MIRROR_PART_AXIS_WINDOWS"
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
    --alignment-space "$ALIGNMENT_SPACE" \
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
