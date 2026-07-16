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

# A wider part produces more y-axis windows. This makes the local sequence
# alignment less brittle than the old 128-pixel crop, where MIN_RUN_LENGTH=4
# required almost every available part window to match.
PART_WIDTH="${PART_WIDTH:-192}"
NUM_PARTS="${NUM_PARTS:-3}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"

# Match the RTL setup used during Arabic training. HEATMAP_DISPLAY_ORDER=visual
# converts the model order back to readable visual order for the plot.
USE_FLIP="${USE_FLIP:-1}"
NO_BILSTM="${NO_BILSTM:-0}"

# Use local pre-BiLSTM CNN embeddings for both visual matching and token
# assignment. Contextual embeddings can blur the exact content of a small crop.
EMBEDDING_SPACE="${EMBEDDING_SPACE:-local}"
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-local}"

# Balanced defaults for short part sequences. These remain selective without
# requiring an isolated 95th-percentile peak in every consecutive part window.
THRESHOLD="${THRESHOLD:-0.80}"
ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-percentile}"
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-90}"

MATCH="${MATCH:-1.0}"
MISMATCH="${MISMATCH:--2.5}"
GAP="${GAP:--0.6}"
MIN_RUN_LENGTH="${MIN_RUN_LENGTH:-3}"

# 0 = draw the exact Smith-Waterman window span without expanding the mask.
MASK_PADDING_WINDOWS="${MASK_PADDING_WINDOWS:-0}"

# Save one cosine-similarity heatmap per chosen part.
HEATMAP="${HEATMAP:-1}"
HEATMAP_DIR="${HEATMAP_DIR:-$OUT_DIR/heatmaps}"

# visual = display windows in physical readable order.
# window = show the complete model window on each axis. Use nonoverlap only for
# a compressed display of the central stride-sized region.
HEATMAP_DISPLAY_ORDER="${HEATMAP_DISPLAY_ORDER:-visual}"
HEATMAP_AXIS_SLICE_MODE="${HEATMAP_AXIS_SLICE_MODE:-nonoverlap}"

# Match visualize_line_self_window_cosine.py axis sizing.
HEATMAP_WINDOW_GAP_PIXELS="${HEATMAP_WINDOW_GAP_PIXELS:-12}"
HEATMAP_AXIS_CELL_PIXELS="${HEATMAP_AXIS_CELL_PIXELS:-52}"
HEATMAP_LINE1_STRIP_HEIGHT="${HEATMAP_LINE1_STRIP_HEIGHT:-84}"
HEATMAP_PART_STRIP_WIDTH="${HEATMAP_PART_STRIP_WIDTH:-108}"

# The y-axis is reversed by default so its last model/displayed window is shown
# first and its first window is shown last. The wrapper reverses the thumbnails,
# heatmap rows, token labels, ticks, SW path, and selected cells together.
HEATMAP_REVERSE_X_AXIS="${HEATMAP_REVERSE_X_AXIS:-0}"
HEATMAP_REVERSE_Y_AXIS="${HEATMAP_REVERSE_Y_AXIS:-1}"

# Same y-axis orientation used by the self-window cosine script.
HEATMAP_Y_AXIS_ROTATE="${HEATMAP_Y_AXIS_ROTATE:-1}"
HEATMAP_Y_AXIS_FLIP="${HEATMAP_Y_AXIS_FLIP:-0}"
export HEATMAP_REVERSE_X_AXIS HEATMAP_REVERSE_Y_AXIS HEATMAP_Y_AXIS_ROTATE HEATMAP_Y_AXIS_FLIP

# Axis-token labels are inferred by hard Span-DTW, with a per-window argmax
# fallback for short cropped parts. Keep each displayed token at most 2 chars.
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

# Keep the actual Arabic window crops readable. Mirroring is available as an
# explicit override but disabled by default.
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
  echo "  part-width                      = $PART_WIDTH"
  echo "  embedding-space                 = $EMBEDDING_SPACE"
  echo "  alignment-space                 = $ALIGNMENT_SPACE"
  echo "  use-flip                        = $USE_FLIP"
  echo "  threshold floor                 = $THRESHOLD"
  echo "  adaptive-threshold              = $ADAPTIVE_THRESHOLD"
  echo "  threshold-percentile            = $THRESHOLD_PERCENTILE"
  echo "  match                           = $MATCH"
  echo "  mismatch                        = $MISMATCH"
  echo "  gap                             = $GAP"
  echo "  min-run-length                  = $MIN_RUN_LENGTH"
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
