#!/usr/bin/env bash
#
# Run self window-to-window cosine similarity visualization for one or more line samples.
#
# Default usage:
#   bash scripts/eval/window-similarity/run_line_self_window_cosine.sh
#
# Override with environment variables, for example:
#   INDEX=5 WHICH_LINE=2 bash scripts/eval/window-similarity/run_line_self_window_cosine.sh
#   START_INDEX=1 END_INDEX=10 WHICH_LINE=1 bash scripts/eval/window-similarity/run_line_self_window_cosine.sh
#   FEATURE_SPACE=contextual bash scripts/eval/window-similarity/run_line_self_window_cosine.sh
#

set -euo pipefail

SCRIPT="scripts/eval/window-similarity/visualize_line_self_window_cosine.py"

WEIGHTS="${WEIGHTS:-Weights/improve_model_win32_fastpair/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
OUT_DIR="${OUT_DIR:-Results/Evaluation/Self_Window_Cosine}"

# Single-sample mode uses INDEX. Batch mode uses START_INDEX and END_INDEX.
INDEX="${INDEX:-1}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
WHICH_LINE="${WHICH_LINE:-1}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE="${STRIDE:-32}"
HEIGHT="${HEIGHT:-128}"
WIDTH="${WIDTH:-1024}"
FEATURE_SPACE="${FEATURE_SPACE:-local}"

# Axis-window visualization.
# nonoverlap = display centered stride-sized slices to avoid visual overlap.
# window     = display the exact full overlapping model windows.
# A slightly larger cell makes the cosine values readable inside every element.
AXIS_SLICE_MODE="${AXIS_SLICE_MODE:-nonoverlap}"
AXIS_CELL_PIXELS="${AXIS_CELL_PIXELS:-52}"
WINDOW_GAP_PIXELS="${WINDOW_GAP_PIXELS:-12}"
X_STRIP_HEIGHT="${X_STRIP_HEIGHT:-84}"
Y_STRIP_WIDTH="${Y_STRIP_WIDTH:-108}"

# display-order visual reorders the base view to the physical image layout.
# REVERSE_X_AXIS=1 additionally reverses the top strip and heatmap columns only,
# which fixes cases where the upper x-axis reads in the wrong direction.
DISPLAY_ORDER="${DISPLAY_ORDER:-visual}"
REVERSE_X_AXIS="${REVERSE_X_AXIS:-0}"
REVERSE_Y_AXIS="${REVERSE_Y_AXIS:-1}"
TICK_LABELS="${TICK_LABELS:-model}"

# Mirror controls. MIRROR_AXIS_WINDOWS mirrors both axes. The x/y-specific flags
# let you fix only one axis if needed.
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
# Show cosine value inside every heatmap element by default.
CELL_VALUES="${CELL_VALUES:-1}"
CELL_VALUE_FONTSIZE="${CELL_VALUE_FONTSIZE:-4.0}"
SHOW_ALL_TICKS="${SHOW_ALL_TICKS:-0}"
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
  local which_line="$2"
  local out_png="${OUT_DIR}/sample_${idx}_line${which_line}_${FEATURE_SPACE}_self_window_cosine.png"

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
  if [[ "${CELL_VALUES}" == "1" || "${CELL_VALUES}" == "true" ]]; then
    extra_args+=(--cell-values --cell-value-fontsize "${CELL_VALUE_FONTSIZE}")
  else
    extra_args+=(--no-cell-values)
  fi
  if [[ "${SHOW_ALL_TICKS}" == "1" || "${SHOW_ALL_TICKS}" == "true" ]]; then
    extra_args+=(--show-all-ticks)
  fi

  echo "===================================================="
  echo "Self window cosine sample=${idx} line=${which_line}"
  echo "  weights               = ${WEIGHTS}"
  echo "  data-dir              = ${DATA_DIR}"
  echo "  output                = ${out_png}"
  echo "  feature-space         = ${FEATURE_SPACE}"
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
  echo "  cell-value-fontsize   = ${CELL_VALUE_FONTSIZE}"
  echo "  use-flip              = ${USE_FLIP}"
  echo "===================================================="

  python "${SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --index "${idx}" \
    --which-line "${which_line}" \
    --weights "${WEIGHTS}" \
    --output "${out_png}" \
    --window-size "${WINDOW_SIZE}" \
    --stride "${STRIDE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --feature-space "${FEATURE_SPACE}" \
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
    run_one "${idx}" "${WHICH_LINE}"
  done
else
  run_one "${INDEX}" "${WHICH_LINE}"
fi

echo "Done. Results saved in: ${OUT_DIR}"
