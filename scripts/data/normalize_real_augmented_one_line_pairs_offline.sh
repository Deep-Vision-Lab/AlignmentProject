#!/usr/bin/env bash
# Normalize the already-built real+augmented dataset to one A/B line pair per folder.
# Running from a login node submits this script to Slurm automatically.
set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
cd "${ROOT}"
mkdir -p out

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDatasetRealAug10K}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/ArabicDatasetRealAug10KOneLine}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
OVERWRITE="${OVERWRITE:-0}"
PARTITION="${PARTITION:-main}"
CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-24G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
JOB_NAME="${JOB_NAME:-real_aug10k_one_line}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting one-line dataset normalization..."
  echo "  source = ${DATA_DIR}"
  echo "  output = ${OUTPUT_DIR}"
  sbatch \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --output="${ROOT}/out/%x_%j.out" \
    --mail-type=ALL \
    --export=ALL,PROJECT_DIR="${ROOT}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV}",OVERWRITE="${OVERWRITE}" \
    "${ROOT}/scripts/data/normalize_real_augmented_one_line_pairs_offline.sh"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
case "${OVERWRITE,,}" in
  1|true|yes|on) args+=(--overwrite) ;;
esac

python scripts/data/normalize_real_augmented_one_line_pairs.py "${args[@]}"

echo
printf '%s\n' \
  "Normalization complete." \
  "  dataset = ${OUTPUT_DIR}" \
  "  manifest = ${OUTPUT_DIR}/dataset_manifest.jsonl" \
  "  summary = ${OUTPUT_DIR}/dataset_summary.json"

wc -l \
  "${OUTPUT_DIR}/dataset_manifest.jsonl" \
  "${OUTPUT_DIR}/train_manifest.jsonl" \
  "${OUTPUT_DIR}/valid_manifest.jsonl" \
  "${OUTPUT_DIR}/test_manifest.jsonl" \
  "${OUTPUT_DIR}/no_shared_content_manifest.jsonl"
