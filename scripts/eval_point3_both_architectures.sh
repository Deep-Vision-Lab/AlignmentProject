#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
OLD_WEIGHTS="${OLD_WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_point2/model_latest.pth}"
NEW_WEIGHTS="${NEW_WEIGHTS:-${PROJECT_DIR}/Weights/res18_physical_window_tinyvit_point2/model_latest.pth}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
N_SAMPLES="${N_SAMPLES:-10}"
START_INDEX="${START_INDEX:-1}"
DEVICE="${DEVICE:-cuda}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
TRAINING_SAMPLES="${TRAINING_SAMPLES:-6000}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Diagnostics/Point3/old_vs_physical_${TAG}}"

cd "${PROJECT_DIR}"

for path in "${OLD_WEIGHTS}" "${NEW_WEIGHTS}"; do
  [[ -f "${path}" ]] || { echo "ERROR: checkpoint not found: ${path}" >&2; exit 2; }
done
[[ -d "${DATASET}" || -f "${DATASET}" ]] || {
  echo "ERROR: dataset not found: ${DATASET}" >&2
  exit 2
}

"${PYTHON_BIN}" -m py_compile   Evaluation/eval_point3_hard_paths.py   Evaluation/point2_runtime.py

echo "============================================================"
echo "Point 3: hard-DTW path correctness"
echo "old weights       = ${OLD_WEIGHTS}"
echo "new weights       = ${NEW_WEIGHTS}"
echo "dataset           = ${DATASET}"
echo "split             = ${EVAL_SPLIT}"
echo "n_samples         = ${N_SAMPLES}"
echo "start_index       = ${START_INDEX}"
echo "preprocessing     = ${IMAGE_PREPROCESSING}"
echo "output            = ${OUTPUT_DIR}"
echo "============================================================"

"${PYTHON_BIN}" -u Evaluation/eval_point3_hard_paths.py   --dataset "${DATASET}"   --old-weights "${OLD_WEIGHTS}"   --new-weights "${NEW_WEIGHTS}"   --output-dir "${OUTPUT_DIR}"   --split "${EVAL_SPLIT}"   --training-samples "${TRAINING_SAMPLES}"   --split-seed "${SPLIT_SEED}"   --start-index "${START_INDEX}"   --n-samples "${N_SAMPLES}"   --device "${DEVICE}"   --image-preprocessing "${IMAGE_PREPROCESSING}"

echo
echo "Main Point-3 outputs:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/point3_architecture_summary.csv"
echo "  ${OUTPUT_DIR}/point3_paired_deltas.csv"
echo "  ${OUTPUT_DIR}/old_resnet_token_vit/pair_*/hard_dtw_heatmap.png"
echo "  ${OUTPUT_DIR}/physical_window_vit/pair_*/hard_dtw_heatmap.png"
