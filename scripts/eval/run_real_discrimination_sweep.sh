#!/usr/bin/env bash
# Fixed-manifest real positive vs no-shared Smith-Waterman sweep.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT to the model checkpoint to evaluate.}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${CHECKPOINT}")")}_discrimination"
N_SAMPLES="${N_SAMPLES:-20}"

BASELINE_ROOT="${BASELINE_ROOT:-${PROJECT_DIR}/Results/Evaluation/Alignment_Diagnostics/cnn_phase3_alignment_diagnostics_nobox}"
POS_MANIFEST="${POS_MANIFEST:-${BASELINE_ROOT}/manifests/diagnostic_positive_rows.jsonl}"
NEG_MANIFEST="${NEG_MANIFEST:-${BASELINE_ROOT}/manifests/diagnostic_no_shared_rows.jsonl}"
OUT="${OUT:-${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${RUN_NAME}}"

for path in "${CHECKPOINT}" "${POS_MANIFEST}" "${NEG_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}"

export LINE_HEIGHT=128
export LINE_WIDTH=1024
export REAL_BINARIZE=1
export REAL_BINARIZE_METHOD=otsu
export REAL_BINARIZE_THRESHOLD=180
export REAL_BINARIZE_AUTOCONTRAST=1
export REAL_BINARIZE_AUTO_INVERT=1
export TARGET_INK_HEIGHT_RATIO=0.72
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO=0.72
export ZERO_SHOT_PREPROCESS=1
export ZERO_SHOT_PRESERVE_ASPECT=1
export ZERO_SHOT_FOREGROUND_CROP=1
export ZERO_SHOT_SOURCE_GEOMETRY=1
export REAL_EVAL_BALANCED=0
export REAL_BOX_EVAL=0
export REAL_REQUIRE_BOX_ANNOTATIONS=0

for T in 0.40 0.50 0.60 0.65 0.70; do
  for CLASS in positive negative; do
    if [[ "${CLASS}" == positive ]]; then MANIFEST="${POS_MANIFEST}"; else MANIFEST="${NEG_MANIFEST}"; fi
    echo "===== ${RUN_NAME} T=${T} ${CLASS} ====="
    python -m Evaluation.eval_img_align_sw_no_png \
      --weights "${CHECKPOINT}" \
      --data-dir "${PROJECT_DIR}/DataSet/ArabicDataset" \
      --dataset-type real \
      --batch \
      --arabic-manifest "${MANIFEST}" \
      --real-split all \
      --real-labels all \
      --n-samples "${N_SAMPLES}" \
      --feature local \
      --score-mode raw \
      --threshold "${T}" \
      --gap -0.30 \
      --heatmap-source cosine \
      --no-save-binarized-images \
      --output-dir "${OUT}/raw_t${T}_${CLASS}"
  done
done

python scripts/eval/summarize_real_discrimination.py "${OUT}"
