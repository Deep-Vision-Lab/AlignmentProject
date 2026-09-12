#!/usr/bin/env bash
# Evaluate the restoration branch on identical pairs with P_i, L_i and their joint score.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/vit_restore_dtw_s16/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
SCORE_MODE="${SCORE_MODE:-raw}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
LOCAL_WEIGHT="${LOCAL_WEIGHT:-0.5}"
DEVICE="${DEVICE:-cuda}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT="${ROOT_OUT:-${PROJECT_DIR}/Results/Evaluation/Yelda/vit_restore_dtw/compare_${EVAL_SPLIT}_${TAG}}"

cd "${PROJECT_DIR}"

[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }
[[ -d "${DATASET}" || -f "${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }

mkdir -p "${ROOT_OUT}"

echo "============================================================"
echo "Restoration representation comparison"
echo "weights       = ${WEIGHTS}"
echo "dataset       = ${DATASET}"
echo "split         = ${EVAL_SPLIT}"
echo "pairs         = ${N_SAMPLES}"
echo "start_index   = ${START_INDEX}"
echo "score_mode    = ${SCORE_MODE}"
echo "threshold     = ${THRESHOLD:-0.0}"
echo "gap           = ${GAP:--0.30}"
echo "preprocessing = ${IMAGE_PREPROCESSING}"
echo "output        = ${ROOT_OUT}"
echo "============================================================"

for REP in local primary joint; do
  echo
  echo ">>> Evaluating representation=${REP}"
  "${PYTHON_BIN}" -u -m Evaluation.eval_yelda \
    --dataset "${DATASET}" \
    --weights "${WEIGHTS}" \
    --branch restoration \
    --representation "${REP}" \
    --image-preprocessing "${IMAGE_PREPROCESSING}" \
    --split "${EVAL_SPLIT}" \
    --training-samples "${TRAINING_SAMPLES:-6000}" \
    --split-seed "${SPLIT_SEED:-42}" \
    --n-samples "${N_SAMPLES}" \
    --start-index "${START_INDEX}" \
    --device "${DEVICE}" \
    --score-mode "${SCORE_MODE}" \
    --local-weight "${LOCAL_WEIGHT}" \
    --threshold "${THRESHOLD:-0.0}" \
    --gap "${GAP:--0.30}" \
    --output-dir "${ROOT_OUT}/${REP}"
done

"${PYTHON_BIN}" -u Evaluation/compare_restoration_eval_runs.py \
  --root "${ROOT_OUT}"

echo
echo "Open these first:"
echo "  ${ROOT_OUT}/representation_comparison.csv"
echo "  ${ROOT_OUT}/interesting_pairs.md"
echo "  ${ROOT_OUT}/primary/pair_XXXXX/cosine_similarity_values.png"
echo "  ${ROOT_OUT}/local/pair_XXXXX/cosine_similarity_values.png"
