#!/usr/bin/env bash
#
# Run feature-concentration visualization for one or more Arabic line samples.
#
# Default behavior now creates one PNG per model window:
#   original window image + feature concentration image + early-fusion overlay
#
# Default usage:
#   bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh
#
# Override with environment variables, for example:
#   INDEX=5 WHICH_LINE=2 bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh
#   START_INDEX=1 END_INDEX=10 WHICH_LINE=1 bash scripts/eval/window-features/run_visualize_window_feature_concentration.sh
#

set -euo pipefail

SCRIPT="scripts/eval/window-features/visualize_window_feature_concentration.py"

WEIGHTS="${WEIGHTS:-Weights/improve_model_win32_fastpair/model_latest.pth}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"
OUT_DIR="${OUT_DIR:-Results/Evaluation/Feature_Concentration}"

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
ALIGNMENT_SPACE="${ALIGNMENT_SPACE:-contextual}"
CONCENTRATION_METRIC="${CONCENTRATION_METRIC:-topk_mass}"
CONCENTRATION_TOP_K="${CONCENTRATION_TOP_K:-8}"
FEATURE_NORMALIZE="${FEATURE_NORMALIZE:-per_window}"
CSV_TOP_FEATURES="${CSV_TOP_FEATURES:-8}"

TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-}"
MAX_SPAN_CHARS="${MAX_SPAN_CHARS:-}"
MAX_TOKEN_CHARS="${MAX_TOKEN_CHARS:-}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WINDOW_COUNT_PENALTY="${WINDOW_COUNT_PENALTY:-0.01}"

TOKEN_FONTSIZE="${TOKEN_FONTSIZE:-6.0}"
DPI="${DPI:-180}"
CMAP="${CMAP:-viridis}"
DEVICE="${DEVICE:-cuda}"

USE_FLIP="${USE_FLIP:-1}"
NO_BILSTM="${NO_BILSTM:-0}"

# Per-window mode is the default requested visualization.
SAVE_PER_WINDOW_IMAGES="${SAVE_PER_WINDOW_IMAGES:-1}"
SAVE_SUMMARY_IMAGE="${SAVE_SUMMARY_IMAGE:-0}"
SAVE_CSV="${SAVE_CSV:-0}"
EARLY_FUSION_ALPHA="${EARLY_FUSION_ALPHA:-0.55}"
PER_WINDOW_IMAGE_WIDTH="${PER_WINDOW_IMAGE_WIDTH:-256}"
PER_WINDOW_IMAGE_HEIGHT="${PER_WINDOW_IMAGE_HEIGHT:-256}"
PER_WINDOW_SCALE="${PER_WINDOW_SCALE:-8}"
PER_WINDOW_DPI="${PER_WINDOW_DPI:-160}"

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
  local out_png="${OUT_DIR}/sample_${idx}_line${which_line}_${FEATURE_SPACE}_summary.png"
  local out_csv="${OUT_DIR}/sample_${idx}_line${which_line}_${FEATURE_SPACE}_features.csv"
  local per_window_dir="${PER_WINDOW_DIR:-${OUT_DIR}/sample_${idx}_line${which_line}_${FEATURE_SPACE}_early_fusion_windows}"

  local extra_args=()
  if [[ "${USE_FLIP}" == "1" || "${USE_FLIP}" == "true" ]]; then
    extra_args+=(--use-flip)
  fi
  if [[ "${NO_BILSTM}" == "1" || "${NO_BILSTM}" == "true" ]]; then
    extra_args+=(--no-bilstm)
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
  if [[ "${SAVE_PER_WINDOW_IMAGES}" == "1" || "${SAVE_PER_WINDOW_IMAGES}" == "true" ]]; then
    extra_args+=(
      --save-per-window-images
      --per-window-output-dir "${per_window_dir}"
      --early-fusion-alpha "${EARLY_FUSION_ALPHA}"
      --per-window-image-width "${PER_WINDOW_IMAGE_WIDTH}"
      --per-window-image-height "${PER_WINDOW_IMAGE_HEIGHT}"
      --per-window-scale "${PER_WINDOW_SCALE}"
      --per-window-dpi "${PER_WINDOW_DPI}"
    )
  fi
  if [[ "${SAVE_SUMMARY_IMAGE}" == "0" || "${SAVE_SUMMARY_IMAGE}" == "false" ]]; then
    extra_args+=(--no-summary-image)
  fi
  if [[ "${SAVE_CSV}" == "0" || "${SAVE_CSV}" == "false" ]]; then
    extra_args+=(--no-csv)
  else
    extra_args+=(--csv-output "${out_csv}")
  fi

  echo "===================================================="
  echo "Feature concentration sample=${idx} line=${which_line}"
  echo "  weights              = ${WEIGHTS}"
  echo "  data-dir             = ${DATA_DIR}"
  echo "  per-window-dir       = ${per_window_dir}"
  echo "  summary-image        = ${SAVE_SUMMARY_IMAGE} (${out_png})"
  echo "  save-csv             = ${SAVE_CSV} (${out_csv})"
  echo "  feature-space        = ${FEATURE_SPACE}"
  echo "  alignment-space      = ${ALIGNMENT_SPACE}"
  echo "  concentration-metric = ${CONCENTRATION_METRIC}"
  echo "  early-fusion-alpha   = ${EARLY_FUSION_ALPHA}"
  if [[ -n "${ARABIC_TEXT_MODEL_NAME}" ]]; then
    echo "  text-model           = ${ARABIC_TEXT_MODEL_NAME}"
  fi
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
    --alignment-space "${ALIGNMENT_SPACE}" \
    --concentration-metric "${CONCENTRATION_METRIC}" \
    --concentration-top-k "${CONCENTRATION_TOP_K}" \
    --feature-normalize "${FEATURE_NORMALIZE}" \
    --csv-top-features "${CSV_TOP_FEATURES}" \
    --max-windows-per-span "${MAX_WINDOWS_PER_SPAN}" \
    --temperature "${TEMPERATURE}" \
    --window-count-penalty "${WINDOW_COUNT_PENALTY}" \
    --token-fontsize "${TOKEN_FONTSIZE}" \
    --dpi "${DPI}" \
    --cmap "${CMAP}" \
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
