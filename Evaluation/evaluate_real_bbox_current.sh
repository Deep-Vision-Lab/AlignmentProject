#!/usr/bin/env bash
# Convenience launcher for canonical bbox.json quantitative evaluation on real data.
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "${ROOT}"

MODEL_NAME="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
case "${MODEL_NAME}" in
  cnn_bilstm)
    RUN_ID="${RUN_ID:-cnn_bilstm_augmented_fixed63_27k}"
    ;;
  vit)
    RUN_ID="${RUN_ID:-vit_augmented_fixed63_27k}"
    ;;
  *)
    RUN_ID="${RUN_ID:-${MODEL_NAME}}"
    ;;
esac

if [[ -z "${WEIGHTS:-}" ]]; then
  for candidate in \
    "${ROOT}/Weights/${RUN_ID}/model_best.pth" \
    "${ROOT}/Weights/${RUN_ID}/model_latest.pth" \
    "${ROOT}/Weights/${RUN_ID}/checkpoint_latest.pth"; do
    if [[ -f "${candidate}" ]]; then
      WEIGHTS="${candidate}"
      break
    fi
  done
fi
: "${WEIGHTS:?Set WEIGHTS or place a checkpoint under Weights/${RUN_ID}.}"

export WEIGHTS
export MODEL_TAG="${MODEL_TAG:-${MODEL_NAME}}"
export RUN_TAG="${RUN_TAG:-${RUN_ID}_real_bbox}"
export REAL_DATA_DIR="${REAL_DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
export ARABIC_MANIFEST="${ARABIC_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}"
export REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-${REAL_DATA_DIR}}"
export REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"
export REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-1}"
export REAL_SPLIT="${REAL_SPLIT:-test}"
export LABELS="${LABELS:-high_match,medium_match}"
export N_SAMPLES="${N_SAMPLES:-100}"
export START_INDEX="${START_INDEX:-1}"
export FEATURE="${FEATURE:-contextual}"
export SCORE_MODE="${SCORE_MODE:-auto}"
export RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/Results/Evaluation/${MODEL_NAME}/Real_Experiments/${RUN_TAG}}"

exec bash "${ROOT}/Evaluation/evaluate.sh"
