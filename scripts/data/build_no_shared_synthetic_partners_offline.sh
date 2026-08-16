#!/usr/bin/env bash
# Submit the no-shared synthetic-partner builder as a CPU Slurm job.
# This script always submits when called interactively; the submitted worker then
# builds the dataset and runs the production smoke test.
set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
cd "${ROOT}"
mkdir -p out

EXPECTED_BRANCH="agent/cnn-bilstm-partial-overlap"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expected branch ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH:-<detached>}." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/DataSet/ArabicDatasetSyntheticPartners}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-main}"
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
JOB_NAME="${JOB_NAME:-build_synthetic_partners}"
SEED="${SEED:-42}"
MIN_REGIONS="${MIN_REGIONS:-1}"
MAX_REGIONS="${MAX_REGIONS:-3}"
MAX_RUN_BOXES="${MAX_RUN_BOXES:-3}"
MIN_CHARS="${MIN_CHARS:-3}"
MAX_CHARS="${MAX_CHARS:-28}"
WIDTH_RATIO_MIN="${WIDTH_RATIO_MIN:-0.40}"
WIDTH_RATIO_MAX="${WIDTH_RATIO_MAX:-2.50}"
MULTI_REGION_PROB="${MULTI_REGION_PROB:-0.65}"
THREE_REGION_PROB="${THREE_REGION_PROB:-0.15}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"

if [[ "${SYNTHETIC_PARTNER_OFFLINE_WORKER:-0}" != "1" ]]; then
  echo "Submitting synthetic-partner CPU build..."
  echo "  branch = ${CURRENT_BRANCH}"
  echo "  commit = $(git rev-parse --short HEAD)"
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
    --export=ALL,SYNTHETIC_PARTNER_OFFLINE_WORKER=1,PROJECT_DIR="${ROOT}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",CONDA_ENV="${CONDA_ENV}",SEED="${SEED}",MIN_REGIONS="${MIN_REGIONS}",MAX_REGIONS="${MAX_REGIONS}",MAX_RUN_BOXES="${MAX_RUN_BOXES}",MIN_CHARS="${MIN_CHARS}",MAX_CHARS="${MAX_CHARS}",WIDTH_RATIO_MIN="${WIDTH_RATIO_MIN}",WIDTH_RATIO_MAX="${WIDTH_RATIO_MAX}",MULTI_REGION_PROB="${MULTI_REGION_PROB}",THREE_REGION_PROB="${THREE_REGION_PROB}",MAX_ATTEMPTS="${MAX_ATTEMPTS}",SMOKE_SAMPLES="${SMOKE_SAMPLES}" \
    "${ROOT}/scripts/data/build_no_shared_synthetic_partners_offline.sh"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export REAL_AUGMENT=0
export AUGMENT=0
export REAL_EXTRA_EXCLUDE_EVAL_PAGES="${REAL_EXTRA_EXCLUDE_EVAL_PAGES:-1}"

printf '%s\n' \
  "=== OFFLINE SYNTHETIC-PARTNER BUILD ===" \
  "host=$(hostname)" \
  "job=${SLURM_JOB_ID:-unknown}" \
  "branch=$(git branch --show-current)" \
  "commit=$(git rev-parse --short HEAD)" \
  "data=${DATA_DIR}" \
  "output=${OUTPUT_DIR}" \
  "regions=${MIN_REGIONS}-${MAX_REGIONS}" \
  "chars=${MIN_CHARS}-${MAX_CHARS}" \
  "run_boxes<=${MAX_RUN_BOXES}"

python -m py_compile \
  SyntheticPartnerRealAugmentation.py \
  scripts/data/build_no_shared_synthetic_partners.py \
  scripts/data/smoke_test_partial_overlap.py

python scripts/data/build_no_shared_synthetic_partners.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --min-regions "${MIN_REGIONS}" \
  --max-regions "${MAX_REGIONS}" \
  --max-run-boxes "${MAX_RUN_BOXES}" \
  --min-chars "${MIN_CHARS}" \
  --max-chars "${MAX_CHARS}" \
  --width-ratio-min "${WIDTH_RATIO_MIN}" \
  --width-ratio-max "${WIDTH_RATIO_MAX}" \
  --multi-region-prob "${MULTI_REGION_PROB}" \
  --three-region-prob "${THREE_REGION_PROB}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --overwrite

python scripts/data/smoke_test_partial_overlap.py \
  --root "${DATA_DIR}" \
  --synthetic-manifest "${OUTPUT_DIR}/dataset_manifest.jsonl" \
  --samples "${SMOKE_SAMPLES}"

echo "=== OFFLINE SYNTHETIC-PARTNER BUILD PASS ==="
echo "manifest=${OUTPUT_DIR}/dataset_manifest.jsonl"
echo "summary=${OUTPUT_DIR}/synthetic_partner_summary.json"
