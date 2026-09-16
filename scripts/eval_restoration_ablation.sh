#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/manucripts_align/bin/python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_dtw_s16/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/Yelda/res18_tinyvit_ablation}"
START_INDEX="${START_INDEX:-1}"
N_SAMPLES="${N_SAMPLES:-100}"
MAX_FIGURES="${MAX_FIGURES:-10}"
GAP_PENALTY="${GAP_PENALTY:--0.25}"
DEVICE="${DEVICE:-auto}"
SAVE_FIGURES="${SAVE_FIGURES:-1}"

cd "${PROJECT_DIR}"

[[ -f "${WEIGHTS}" ]] || { echo "ERROR: weights not found: ${WEIGHTS}" >&2; exit 2; }
[[ -d "${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }
[[ -d "${DATASET}/masks" ]] || { echo "ERROR: masks not found: ${DATASET}/masks" >&2; exit 2; }

args=(
  Evaluation/eval_restoration_ablation.py
  --weights "${WEIGHTS}"
  --data-dir "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --start-index "${START_INDEX}"
  --n-samples "${N_SAMPLES}"
  --max-figures "${MAX_FIGURES}"
  --gap-penalty "${GAP_PENALTY}"
  --device "${DEVICE}"
)

if [[ "${SAVE_FIGURES}" == "1" ]]; then
  args+=(--save-figures)
fi

"${PYTHON_BIN}" -m py_compile Evaluation/_eval_utils.py Evaluation/eval_restoration_ablation.py
"${PYTHON_BIN}" -m pytest -q tests/test_restoration_ablation_eval.py
"${PYTHON_BIN}" "${args[@]}"
