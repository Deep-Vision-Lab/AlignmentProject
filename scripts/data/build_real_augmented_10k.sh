#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${CONDA_ENV_PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/ArabicDatasetRealAug10K}"
TARGET_TRAIN_PAIRS="${TARGET_TRAIN_PAIRS:-10000}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-128}"
MIN_REGIONS="${MIN_REGIONS:-1}"
MAX_REGIONS="${MAX_REGIONS:-3}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --target-train-pairs "${TARGET_TRAIN_PAIRS}"
  --seed "${SEED}"
  --height "${HEIGHT}"
  --min-regions "${MIN_REGIONS}"
  --max-regions "${MAX_REGIONS}"
)

if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-0}" == "true" ]]; then
  args+=(--overwrite)
fi

"${PYTHON_BIN}" scripts/data/build_real_augmented_training_dataset.py "${args[@]}"
