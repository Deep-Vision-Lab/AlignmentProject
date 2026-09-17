#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
OLD_WEIGHTS="${OLD_WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_point2/model_latest.pth}"
NEW_WEIGHTS="${NEW_WEIGHTS:-${PROJECT_DIR}/Weights/res18_physical_window_tinyvit_point2/model_latest.pth}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
DEVICE="${DEVICE:-cuda}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
SCORE_MODE="${SCORE_MODE:-raw}"
MIN_ALIGNED_WINDOWS="${MIN_ALIGNED_WINDOWS:-5}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT="${ROOT_OUT:-${PROJECT_DIR}/Results/Evaluation/Point2/old_vs_physical_${TAG}}"

cd "${PROJECT_DIR}"

[[ -f "${OLD_WEIGHTS}" ]] || {
  echo "ERROR: old checkpoint not found: ${OLD_WEIGHTS}" >&2
  echo "Set OLD_WEIGHTS=/path/to/model_latest.pth" >&2
  exit 2
}
[[ -f "${NEW_WEIGHTS}" ]] || {
  echo "ERROR: physical-window checkpoint not found: ${NEW_WEIGHTS}" >&2
  echo "Set NEW_WEIGHTS=/path/to/model_latest.pth" >&2
  exit 2
}
[[ -d "${DATASET}" || -f "${DATASET}" ]] || {
  echo "ERROR: dataset not found: ${DATASET}" >&2
  exit 2
}

"${PYTHON_BIN}" -m py_compile \
  Evaluation/point2_runtime.py \
  Evaluation/eval_point2.py \
  Evaluation/compare_point2_eval_runs.py \
  Evaluation/compare_point2_architectures.py

mkdir -p "${ROOT_OUT}"

if [[ "${DEVICE}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA evaluation requested but CUDA is unavailable")
print("CUDA device 0 =", torch.cuda.get_device_name(0))
PY
fi

echo "============================================================"
echo "Point-2: old ResNet-token ViT vs physical-window ViT"
echo "old weights       = ${OLD_WEIGHTS}"
echo "new weights       = ${NEW_WEIGHTS}"
echo "dataset           = ${DATASET}"
echo "split             = ${EVAL_SPLIT}"
echo "n_samples         = ${N_SAMPLES}"
echo "start_index       = ${START_INDEX}"
echo "preprocessing     = ${IMAGE_PREPROCESSING}"
echo "min_windows       = ${MIN_ALIGNED_WINDOWS}"
echo "output            = ${ROOT_OUT}"
echo "============================================================"

run_architecture () {
  local label="$1"
  local weights="$2"
  local root="${ROOT_OUT}/${label}"

  for rep in local context fused fused_wrong_context; do
    echo
    echo ">>> ${label}: representation=${rep}"
    "${PYTHON_BIN}" -u -m Evaluation.eval_point2 \
      --point2-representation "${rep}" \
      --dataset "${DATASET}" \
      --weights "${weights}" \
      --branch restoration \
      --alignment-unit window \
      --word-support-floor "${WORD_SUPPORT_FLOOR:-0.0}" \
      --min-aligned-windows "${MIN_ALIGNED_WINDOWS}" \
      --image-preprocessing "${IMAGE_PREPROCESSING}" \
      --split "${EVAL_SPLIT}" \
      --training-samples "${TRAINING_SAMPLES:-6000}" \
      --split-seed "${SPLIT_SEED:-42}" \
      --n-samples "${N_SAMPLES}" \
      --start-index "${START_INDEX}" \
      --device "${DEVICE}" \
      --score-mode "${SCORE_MODE}" \
      --threshold "${THRESHOLD:-0.0}" \
      --gap "${GAP:--0.30}" \
      --output-dir "${root}/${rep}"
  done

  "${PYTHON_BIN}" -u Evaluation/compare_point2_eval_runs.py \
    --root "${root}" \
    --label "${label}"
}

run_architecture "old_resnet_token_vit" "${OLD_WEIGHTS}"
run_architecture "physical_window_vit" "${NEW_WEIGHTS}"

"${PYTHON_BIN}" -u Evaluation/compare_point2_architectures.py \
  --old-root "${ROOT_OUT}/old_resnet_token_vit" \
  --new-root "${ROOT_OUT}/physical_window_vit" \
  --output "${ROOT_OUT}/architecture_comparison.json"

echo
echo "Main outputs:"
echo "  ${ROOT_OUT}/old_resnet_token_vit/point2_representation_comparison.csv"
echo "  ${ROOT_OUT}/physical_window_vit/point2_representation_comparison.csv"
echo "  ${ROOT_OUT}/architecture_comparison.csv"
echo "  ${ROOT_OUT}/architecture_comparison.json"
