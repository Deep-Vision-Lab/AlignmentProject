#!/usr/bin/env bash
#
# Run line1-to-line2 window cosine similarity visualization.
#
# Default usage:
#   bash scripts/eval/window-similarity/run_line_pair_window_cosine.sh
#
# Override examples:
#   INDEX=5 bash scripts/eval/window-similarity/run_line_pair_window_cosine.sh
#   FEATURE_SPACE=contextual bash scripts/eval/window-similarity/run_line_pair_window_cosine.sh
#   START_INDEX=1 END_INDEX=10 bash scripts/eval/window-similarity/run_line_pair_window_cosine.sh
#

set -euo pipefail

SCRIPT="scripts/eval/window-similarity/visualize_line_pair_window_cosine.py"

WEIGHTS="${WEIGHTS:-Weights/improve_model_win32_fastpair/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
OUT_DIR="${OUT_DIR:-Results/Evaluation/LinePair_Window_Cosine}"

INDEX="${INDEX:-1}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"
WIDTH="${WIDTH:-1024}"
FEATURE_SPACE="${FEATURE_SPACE:-local}"
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-contextual}"

# Axis-window visualization.
# nonoverlap = display centered stride-sized slices to avoid visual overlap.
# window     = display the exact full overlapping model windows.
AXIS_SLICE_MODE="${AXIS_SLICE_MODE:-nonoverlap}"
AXIS_CELL_PIXELS="${AXIS_CELL_PIXELS:-52}"
WINDOW_GAP_PIXELS="${WINDOW_GAP_PIXELS:-12}"
X_STRIP_HEIGHT="${X_STRIP_HEIGHT:-84}"
Y_STRIP_WIDTH="${Y_STRIP_WIDTH:-108}"

# Axis token labels. Tokens are inferred by hard Span-DTW and shown:
#   line1 tokens above x-axis windows
#   line2 tokens left of y-axis windows
SHOW_AXIS_TOKENS="${SHOW_AXIS_TOKENS:-1}"
AXIS_TOKEN_FONTSIZE="${AXIS_TOKEN_FONTSIZE:-7.0}"
X_TOKEN_HEIGHT="${X_TOKEN_HEIGHT:-44}"
Y_TOKEN_WIDTH="${Y_TOKEN_WIDTH:-72}"
X_TOKEN_ROTATION="${X_TOKEN_ROTATION:-90}"
Y_TOKEN_ROTATION="${Y_TOKEN_ROTATION:-0}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-}"
MAX_SPAN_CHARS="${MAX_SPAN_CHARS:-}"
MAX_TOKEN_CHARS="${MAX_TOKEN_CHARS:-}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY:-0.01}"

# display-order visual reorders the base view to the physical image layout.
# REVERSE_X_AXIS=1 additionally reverses the line1 top strip and heatmap columns,
# matching the latest self-window visualization default.
DISPLAY_ORDER="${DISPLAY_ORDER:-visual}"
REVERSE_X_AXIS="${REVERSE_X_AXIS:-0}"
REVERSE_Y_AXIS="${REVERSE_Y_AXIS:-1}"
TICK_LABELS="${TICK_LABELS:-model}"

# Mirror controls. By default, mirror only the x-axis thumbnails, matching the
# latest self-window visualization behavior.
MIRROR_AXIS_WINDOWS="${MIRROR_AXIS_WINDOWS:-0}"
MIRROR_X_AXIS_WINDOWS="${MIRROR_X_AXIS_WINDOWS:-0}"
MIRROR_Y_AXIS_WINDOWS="${MIRROR_Y_AXIS_WINDOWS:-0}"

CMAP="${CMAP:-viridis}"
VMIN="${VMIN:--1.0}"
VMAX="${VMAX:-1.0}"
DPI="${DPI:-180}"
DEVICE="${DEVICE:-cuda}"

USE_FLIP="${USE_FLIP:-1}"
NO_BILSTM="${NO_BILSTM:-0}"
Y_AXIS_ROTATE="${Y_AXIS_ROTATE:-1}"
Y_AXIS_FLIP="${Y_AXIS_FLIP:-0}"
CELL_VALUES="${CELL_VALUES:-1}"
CELL_VALUE_FONTSIZE="${CELL_VALUE_FONTSIZE:-4.0}"
SHOW_ALL_TICKS="${SHOW_ALL_TICKS:-0}"
DRAW_MAIN_DIAGONAL="${DRAW_MAIN_DIAGONAL:-0}"
COLORBAR_WIDTH="${COLORBAR_WIDTH:-80}"
FIGURE_DPI_SCALE="${FIGURE_DPI_SCALE:-90}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Cannot find script: ${SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Cannot find weights: ${WEIGHTS}" >&2
  echo "Available .pth files:" >&2
  find . -name "*.pth" >&2 || true
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Cannot find DATA_DIR: ${DATA_DIR}" >&2
  exit 1
fi

run_one() {
  local idx="$1"
  local out_png="${OUT_DIR}/sample_${idx}_line1_line2_${FEATURE_SPACE}_window_cosine.png"

  local extra_args=()
  if [[ "${USE_FLIP}" == "1" || "${USE_FLIP}" == "true" ]]; then
    extra_args+=(--use-flip)
  fi
  if [[ "${NO_BILSTM}" == "1" || "${NO_BILSTM}" == "true" ]]; then
    extra_args+=(--no-bilstm)
  fi
  if [[ "${REVERSE_X_AXIS}" == "1" || "${REVERSE_X_AXIS}" == "true" ]]; then
    extra_args+=(--reverse-x-axis)
  fi
  if [[ "${REVERSE_Y_AXIS}" == "1" || "${REVERSE_Y_AXIS}" == "true" ]]; then
    extra_args+=(--reverse-y-axis)
  fi
  if [[ "${MIRROR_AXIS_WINDOWS}" == "1" || "${MIRROR_AXIS_WINDOWS}" == "true" ]]; then
    extra_args+=(--mirror-axis-windows)
  fi
  if [[ "${MIRROR_X_AXIS_WINDOWS}" == "1" || "${MIRROR_X_AXIS_WINDOWS}" == "true" ]]; then
    extra_args+=(--mirror-x-axis-windows)
  fi
  if [[ "${MIRROR_Y_AXIS_WINDOWS}" == "1" || "${MIRROR_Y_AXIS_WINDOWS}" == "true" ]]; then
    extra_args+=(--mirror-y-axis-windows)
  fi
  if [[ "${Y_AXIS_ROTATE}" == "0" || "${Y_AXIS_ROTATE}" == "false" ]]; then
    extra_args+=(--no-y-axis-rotate)
  fi
  if [[ "${Y_AXIS_FLIP}" == "1" || "${Y_AXIS_FLIP}" == "true" ]]; then
    extra_args+=(--y-axis-flip)
  fi
  if [[ "${CELL_VALUES}" == "0" || "${CELL_VALUES}" == "false" ]]; then
    extra_args+=(--no-cell-values)
  else
    extra_args+=(--cell-value-fontsize "${CELL_VALUE_FONTSIZE}")
  fi
  if [[ "${SHOW_AXIS_TOKENS}" == "0" || "${SHOW_AXIS_TOKENS}" == "false" ]]; then
    extra_args+=(--no-axis-tokens)
  else
    extra_args+=(
      --axis-token-fontsize "${AXIS_TOKEN_FONTSIZE}"
      --x-token-height "${X_TOKEN_HEIGHT}"
      --y-token-width "${Y_TOKEN_WIDTH}"
      --x-token-rotation "${X_TOKEN_ROTATION}"
      --y-token-rotation "${Y_TOKEN_ROTATION}"
    )
  fi
  if [[ -n "${TEXT_ENCODER_TYPE}" ]]; then
    extra_args+=(--text-encoder-type "${TEXT_ENCODER_TYPE}")
  fi
  if [[ -n "${ARABIC_TEXT_MODEL_NAME}" ]]; then
    extra_args+=(--arabic-text-model-name "${ARABIC_TEXT_MODEL_NAME}")
  fi
  if [[ -n "${MAX_SPAN_CHARS}" ]]; then
    extra_args+=(--max-span-chars "${MAX_SPAN_CHARS}")
  fi
  if [[ -n "${MAX_TOKEN_CHARS}" ]]; then
    extra_args+=(--max-token-chars "${MAX_TOKEN_CHARS}")
  fi
  if [[ "${SHOW_ALL_TICKS}" == "1" || "${SHOW_ALL_TICKS}" == "true" ]]; then
    extra_args+=(--show-all-ticks)
  fi
  if [[ "${DRAW_MAIN_DIAGONAL}" == "1" || "${DRAW_MAIN_DIAGONAL}" == "true" ]]; then
    extra_args+=(--draw-main-diagonal)
  fi

  echo "===================================================="
  echo "Line1-line2 window cosine sample=${idx}"
  echo "  weights               = ${WEIGHTS}"
  echo "  data-dir              = ${DATA_DIR}"
  echo "  output                = ${out_png}"
  echo "  feature-space         = ${FEATURE_SPACE}"
  echo "  alignment-space       = ${ALIGNMENT_SPACE}"
  echo "  show-axis-tokens      = ${SHOW_AXIS_TOKENS}"
  echo "  display-order         = ${DISPLAY_ORDER}"
  echo "  reverse-x-axis        = ${REVERSE_X_AXIS}"
  echo "  reverse-y-axis        = ${REVERSE_Y_AXIS}"
  echo "  mirror-axis-windows   = ${MIRROR_AXIS_WINDOWS}"
  echo "  mirror-x-axis-windows = ${MIRROR_X_AXIS_WINDOWS}"
  echo "  mirror-y-axis-windows = ${MIRROR_Y_AXIS_WINDOWS}"
  echo "  axis-slice-mode       = ${AXIS_SLICE_MODE}"
  echo "  axis-cell-pixels      = ${AXIS_CELL_PIXELS}"
  echo "  window-gap-pixels     = ${WINDOW_GAP_PIXELS}"
  echo "  cell-values           = ${CELL_VALUES}"
  echo "  use-flip              = ${USE_FLIP}"
  echo "===================================================="

  python "${SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --index "${idx}" \
    --weights "${WEIGHTS}" \
    --output "${out_png}" \
    --window-size "${WINDOW_SIZE}" \
    --stride "${STRIDE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --feature-space "${FEATURE_SPACE}" \
    --alignment-space "${ALIGNMENT_SPACE}" \
    --max-windows-per-span "${MAX_WINDOWS_PER_SPAN}" \
    --temperature "${TEMPERATURE}" \
    --window-count-penalty "${WINDOW_COUNT_PENALTY}" \
    --display-order "${DISPLAY_ORDER}" \
    --tick-labels "${TICK_LABELS}" \
    --axis-slice-mode "${AXIS_SLICE_MODE}" \
    --axis-cell-pixels "${AXIS_CELL_PIXELS}" \
    --window-gap-pixels "${WINDOW_GAP_PIXELS}" \
    --x-strip-height "${X_STRIP_HEIGHT}" \
    --y-strip-width "${Y_STRIP_WIDTH}" \
    --cmap "${CMAP}" \
    --vmin "${VMIN}" \
    --vmax "${VMAX}" \
    --dpi "${DPI}" \
    --figure-dpi-scale "${FIGURE_DPI_SCALE}" \
    --colorbar-width "${COLORBAR_WIDTH}" \
    --device "${DEVICE}" \
    "${extra_args[@]}"
}

if [[ -n "${START_INDEX}" || -n "${END_INDEX}" ]]; then
  if [[ -z "${START_INDEX}" || -z "${END_INDEX}" ]]; then
    echo "Set both START_INDEX and END_INDEX for batch mode." >&2
    exit 2
  fi
  for idx in $(seq "${START_INDEX}" "${END_INDEX}"); do
    run_one "${idx}"
  done
else
  run_one "${INDEX}"
fi

echo "Done. Results saved in: ${OUT_DIR}"
