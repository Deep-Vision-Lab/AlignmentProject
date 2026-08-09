#!/usr/bin/env bash
# Build the standalone real + bbox-injection dataset on a CPU Slurm node.
# Running this file from a login node automatically submits itself with sbatch.
set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
if [[ ! -f "${ROOT}/scripts/data/build_real_augmented_same_skeleton.py" ]]; then
  ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." 2>/dev/null && pwd || true)"
fi
if [[ ! -f "${ROOT}/scripts/data/build_real_augmented_same_skeleton.py" ]]; then
  echo "Could not resolve AlignmentProject root." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/ArabicDatasetRealAug10K}"
TARGET_TRAIN_PAIRS="${TARGET_TRAIN_PAIRS:-10000}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-128}"
MIN_REGIONS="${MIN_REGIONS:-1}"
MAX_REGIONS="${MAX_REGIONS:-3}"
MAX_RUN_BOXES="${MAX_RUN_BOXES:-3}"
OVERWRITE="${OVERWRITE:-0}"
KEEP_BUILD_ARTIFACTS="${KEEP_BUILD_ARTIFACTS:-0}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"

PARTITION="${PARTITION:-main}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
JOB_NAME="${JOB_NAME:-real_aug_10k}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${ROOT}/out"
  echo "Submitting offline real-dataset build..."
  echo "  source       = ${DATA_DIR}"
  echo "  output       = ${OUTPUT_DIR}"
  echo "  train target = ${TARGET_TRAIN_PAIRS}"
  echo "  injection    = ${MIN_REGIONS}-${MAX_REGIONS} region(s), height=${HEIGHT}px"
  echo "  partition    = ${PARTITION}"

  sbatch \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --output="${ROOT}/out/%x_%j.out" \
    --mail-type=ALL \
    --export=ALL,PROJECT_DIR="${ROOT}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",TARGET_TRAIN_PAIRS="${TARGET_TRAIN_PAIRS}",SEED="${SEED}",HEIGHT="${HEIGHT}",MIN_REGIONS="${MIN_REGIONS}",MAX_REGIONS="${MAX_REGIONS}",MAX_RUN_BOXES="${MAX_RUN_BOXES}",OVERWRITE="${OVERWRITE}",KEEP_BUILD_ARTIFACTS="${KEEP_BUILD_ARTIFACTS}",CONDA_ENV="${CONDA_ENV}" \
    "${ROOT}/scripts/data/create_real_augmented_dataset_offline.sh"
  exit 0
fi

cd "${ROOT}"
mkdir -p out
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

printf '%s\n' \
  "----------------------------------------" \
  "Offline real+augmented dataset build" \
  "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  "HOST=$(hostname)" \
  "PROJECT_DIR=${ROOT}" \
  "DATA_DIR=${DATA_DIR}" \
  "OUTPUT_DIR=${OUTPUT_DIR}" \
  "TARGET_TRAIN_PAIRS=${TARGET_TRAIN_PAIRS}" \
  "SEED=${SEED}" \
  "HEIGHT=${HEIGHT}" \
  "REGIONS=${MIN_REGIONS}-${MAX_REGIONS}" \
  "----------------------------------------"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --target-train-pairs "${TARGET_TRAIN_PAIRS}"
  --seed "${SEED}"
  --height "${HEIGHT}"
  --min-regions "${MIN_REGIONS}"
  --max-regions "${MAX_REGIONS}"
  --max-run-boxes "${MAX_RUN_BOXES}"
)

case "${OVERWRITE,,}" in
  1|true|yes|on) args+=(--overwrite) ;;
esac
case "${KEEP_BUILD_ARTIFACTS,,}" in
  1|true|yes|on) args+=(--keep-build-artifacts) ;;
esac

python scripts/data/build_real_augmented_same_skeleton.py "${args[@]}"

echo
printf '%s\n' \
  "Dataset build completed." \
  "  root     = ${OUTPUT_DIR}" \
  "  manifest = ${OUTPUT_DIR}/dataset_manifest.jsonl" \
  "  train    = ${OUTPUT_DIR}/train_manifest.jsonl" \
  "  valid    = ${OUTPUT_DIR}/valid_manifest.jsonl" \
  "  test     = ${OUTPUT_DIR}/test_manifest.jsonl" \
  "  summary  = ${OUTPUT_DIR}/dataset_summary.json"

wc -l \
  "${OUTPUT_DIR}/train_manifest.jsonl" \
  "${OUTPUT_DIR}/valid_manifest.jsonl" \
  "${OUTPUT_DIR}/test_manifest.jsonl" \
  "${OUTPUT_DIR}/dataset_manifest.jsonl"
