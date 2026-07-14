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
STRIDE="${STRIDE:-16}"
HEIGHT="${HEIGHT:-128}"
WIDTH="${WIDTH:-1024}"
FEATURE_SPACE="${FEATURE_SPACE:-local}"

# Axis-window visualization.
# nonoverlap = display centered stride-sized slices to avoid visual overlap.
# window     = display the exact full overlapping model windows.
AXIS_SLICE_MODE="${AXIS_SLICE_MODE:-nonoverlap}"
AXIS_CELL_PIXELS="${AXIS_CELL_PIXELS:-44}"
WINDOW_GAP_PIXELS="${WINDOW_GAP_PIXELS:-12}"
X_STRIP_HEIGHT="${X_STRIP_HEIGHT:-84}"
Y_STRIP_WIDTH="${Y_STRIP_WIDTH:-108}"

CMAP="${CMAP:-viridis}"
VMIN="${VMIN:--1.0}"
VMAX="${VMAX:-1.0}"
DPI="${DPI:-180}"
DEVICE="${DEVICE:-cuda}"

USE_FLIP="${USE_FLIP:-1}"
NO_BILSTM="${NO_BILSTM:-0}"
Y_AXIS_ROTATE="${Y_AXIS_ROTATE:-1}"
Y_AXIS_FLIP="${Y_AXIS_FLIP:-0}"
CELL_VALUES="${CELL_VALUES:-0}"
CELL_VALUE_FONTSIZE="${CELL_VALUE_FONTSIZE:-3.2}"
SHOW_ALL_TICKS="${SHOW_ALL_TICKS:-0}"

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
  if [[ "${Y_AXIS_ROTATE}" == "0" || "${Y_AXIS_ROTATE}" == "false" ]]; then
    extra_args+=(--no-y-axis-rotate)
  fi
  if [[ "${Y_AXIS_FLIP}" == "1" || "${Y_AXIS_FLIP}" == "true" ]]; then
    extra_args+=(--y-axis-flip)
  fi
  if [[ "${CELL_VALUES}" == "1" || "${CELL_VALUES}" == "true" ]]; then
    extra_args+=(--cell-values --cell-value-fontsize "${CELL_VALUE_FONTSIZE}")
  fi
  if [[ "${SHOW_ALL_TICKS}" == "1" || "${SHOW_ALL_TICKS}" == "true" ]]; then
    extra_args+=(--show-all-ticks)
  fi

  echo "===================================================="
  echo "Self window cosine sample=${idx} line=${which_line}"
  echo "  weights           = ${WEIGHTS}"
  echo "  data-dir          = ${DATA_DIR}"
  echo "  output            = ${out_png}"
  echo "  feature-space     = ${FEATURE_SPACE}"
  echo "  axis-slice-mode   = ${AXIS_SLICE_MODE}"
  echo "  axis-cell-pixels  = ${AXIS_CELL_PIXELS}"
  echo "  window-gap-pixels = ${WINDOW_GAP_PIXELS}"
  echo "  use-flip          = ${USE_FLIP}"
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
    --axis-slice-mode "${AXIS_SLICE_MODE}" \
    --axis-cell-pixels "${AXIS_CELL_PIXELS}" \
    --window-gap-pixels "${WINDOW_GAP_PIXELS}" \
    --x-strip-height "${X_STRIP_HEIGHT}" \
    --y-strip-width "${Y_STRIP_WIDTH}" \
    --cmap "${CMAP}" \
    --vmin "${VMIN}" \
    --vmax "${VMAX}" \
    --dpi "${DPI}" \
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
