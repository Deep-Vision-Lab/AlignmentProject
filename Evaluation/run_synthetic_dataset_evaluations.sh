#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS="${WEIGHTS:-${1:-}}"
SYNTHETIC_DATA_DIR="${SYNTHETIC_DATA_DIR:-DataSet/Synthetic_Arabic}"
START_INDEX="${START_INDEX:-1}"
N_SAMPLES="${N_SAMPLES:-20}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-Results/Evaluation/Synthetic_SW}"

if [[ -z "$WEIGHTS" ]]; then
  cat >&2 <<USAGE
Usage:
  WEIGHTS=Weights/<job_id>/model_latest.pth \
    bash Evaluation/run_synthetic_dataset_evaluations.sh

or:
  bash Evaluation/run_synthetic_dataset_evaluations.sh \
    Weights/<job_id>/model_latest.pth
USAGE
  exit 2
fi

[[ -f "$WEIGHTS" ]] || { echo "Checkpoint not found: $WEIGHTS" >&2; exit 2; }
[[ -d "$SYNTHETIC_DATA_DIR/images" ]] || {
  echo "Synthetic images directory not found: $SYNTHETIC_DATA_DIR/images" >&2
  exit 2
}

# Use exactly the same geometry and foreground scale as training and real eval.
export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export SYNTHETIC_BINARIZE="${SYNTHETIC_BINARIZE:-1}"
export SW_INK_AWARE="${SW_INK_AWARE:-1}"
export SW_MIN_INK="${SW_MIN_INK:-0.02}"
export SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
export SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

mkdir -p "$RESULTS_DIR"

printf '%s\n' \
  "Synthetic dataset evaluation" \
  "  checkpoint : $WEIGHTS" \
  "  dataset    : $SYNTHETIC_DATA_DIR" \
  "  samples    : $START_INDEX..$((START_INDEX + N_SAMPLES - 1))" \
  "  feature    : $FEATURE" \
  "  output     : $RESULTS_DIR" \
  "  canvas     : ${LINE_WIDTH}x${LINE_HEIGHT}" \
  "  ink ratio  : $TARGET_INK_HEIGHT_RATIO"

python -m Evaluation.eval_img_align_sw \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --data-dir "$SYNTHETIC_DATA_DIR" \
  --dataset-type synthetic \
  --batch \
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
