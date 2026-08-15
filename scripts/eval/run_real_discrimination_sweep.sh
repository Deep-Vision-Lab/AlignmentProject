#!/usr/bin/env bash
# Fixed-manifest real positive vs no-shared Smith-Waterman sweep.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT to the model checkpoint to evaluate.}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${CHECKPOINT}")")}_discrimination"
N_SAMPLES="${N_SAMPLES:-20}"

DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-${PROJECT_DIR}/Results/Evaluation/Alignment_Diagnostics/joint_real_fixed_diagnostics}"
MANIFEST_DIR="${MANIFEST_DIR:-${DIAGNOSTIC_ROOT}/manifests}"
POS_MANIFEST="${POS_MANIFEST:-${MANIFEST_DIR}/diagnostic_positive_rows.jsonl}"
NEG_MANIFEST="${NEG_MANIFEST:-${MANIFEST_DIR}/diagnostic_no_shared_rows.jsonl}"
OUT="${OUT:-${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${RUN_NAME}}"
CANONICAL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest.jsonl"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${CANONICAL_MANIFEST}" ]] || { echo "ERROR: missing ${CANONICAL_MANIFEST}" >&2; exit 2; }

if [[ ! -f "${POS_MANIFEST}" || ! -f "${NEG_MANIFEST}" ]]; then
  echo "Fixed diagnostic manifests are missing; building deterministic leakage-safe manifests."
  python scripts/eval/build_joint_real_diagnostic_manifests.py \
    --manifest "${CANONICAL_MANIFEST}" \
    --output-dir "${MANIFEST_DIR}" \
    --train-fraction "${REAL_TRAIN_FRACTION:-0.80}" \
    --valid-fraction "${REAL_VALID_FRACTION:-0.10}" \
    --seed "${DATASET_SPLIT_SEED:-42}" \
    --max-per-class "${DIAGNOSTIC_MAX_PER_CLASS:-100}" \
    --min-per-class "${DIAGNOSTIC_MIN_PER_CLASS:-20}"
fi

for path in "${POS_MANIFEST}" "${NEG_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}"

echo "Diagnostic manifests:"
echo "  positive=${POS_MANIFEST}"
echo "  negative=${NEG_MANIFEST}"
echo "  n_samples=${N_SAMPLES}"

export LINE_HEIGHT="${LINE_HEIGHT:-128}"
export LINE_WIDTH="${LINE_WIDTH:-1024}"
export REAL_BINARIZE="${REAL_BINARIZE:-1}"
export REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
export REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
export REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
export REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
export TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"
export ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
export ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
export ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
export ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
export REAL_EVAL_BALANCED=0
export REAL_BOX_EVAL=0
export REAL_REQUIRE_BOX_ANNOTATIONS=0

echo "  REAL_BINARIZE=${REAL_BINARIZE} method=${REAL_BINARIZE_METHOD}"

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
