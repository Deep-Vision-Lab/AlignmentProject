#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="${WEIGHTS:-${1:-}}"
REAL_DATA_DIR="${REAL_DATA_DIR:-DataSet/ArabicDataset}"
ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
REAL_SPLIT="${REAL_SPLIT:-test}"
REAL_LABELS="${REAL_LABELS:-high_match,medium_match}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
START_INDEX="${START_INDEX:-1}"
N_SAMPLES="${N_SAMPLES:-20}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-Results/Evaluation/Real_SW/${REAL_SPLIT}}"

if [[ -z "$WEIGHTS" ]]; then
  cat >&2 <<USAGE
Usage:
  WEIGHTS=Weights/<job_id>/model_latest.pth \
    bash Evaluation/run_real_dataset_evaluations.sh

or:
  bash Evaluation/run_real_dataset_evaluations.sh \
    Weights/<job_id>/model_latest.pth
USAGE
  exit 2
fi

[[ -f "$WEIGHTS" ]] || { echo "Checkpoint not found: $WEIGHTS" >&2; exit 2; }
[[ -d "$REAL_DATA_DIR" ]] || { echo "Real dataset root not found: $REAL_DATA_DIR" >&2; exit 2; }
[[ -f "$ARABIC_MANIFEST" ]] || { echo "ArabicDataset manifest not found: $ARABIC_MANIFEST" >&2; exit 2; }

# Match evaluation geometry exactly to canonical synthetic/real training.
export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_EVAL_BALANCED="${REAL_EVAL_BALANCED:-1}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

mkdir -p "$RESULTS_DIR"

printf '%s\n' \
  "Real ArabicDataset evaluation" \
  "  checkpoint : $WEIGHTS" \
  "  dataset    : $REAL_DATA_DIR" \
  "  manifest   : $ARABIC_MANIFEST" \
  "  split      : $REAL_SPLIT" \
  "  labels     : $REAL_LABELS" \
  "  samples    : $START_INDEX..$((START_INDEX + N_SAMPLES - 1))" \
  "  feature    : $FEATURE" \
  "  output     : $RESULTS_DIR" \
  "  canvas     : ${LINE_WIDTH}x${LINE_HEIGHT}" \
  "  ink ratio  : $TARGET_INK_HEIGHT_RATIO"

python -m Evaluation.eval_img_align_sw \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --data-dir "$REAL_DATA_DIR" \
  --arabic-manifest "$ARABIC_MANIFEST" \
  --dataset-type real \
  --batch \
  --real-split "$REAL_SPLIT" \
  --real-labels "$REAL_LABELS" \
  --real-text-key "$REAL_TEXT_KEY" \
  --real-min-text-score "$REAL_MIN_TEXT_SCORE" \
  --start-index "$START_INDEX" \
  --n-samples "$N_SAMPLES" \
  --feature "$FEATURE" \
  --score-mode "$SCORE_MODE" \
  --score-clip "$SCORE_CLIP" \
  --threshold "$THRESHOLD" \
  --gap "$GAP" \
  --heatmap-source "$HEATMAP_SOURCE" \
  --no-save-binarized-images \
  --output-dir "$RESULTS_DIR"
