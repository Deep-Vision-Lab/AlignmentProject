#!/usr/bin/env bash
# Submit the one-time RealSyntheticBridge V2 dataset build/audit job.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
CPU_PARTITION="${CPU_PARTITION:-main}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"

[[ -s "${REAL_DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: missing real manifest: ${REAL_DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}

RAW_JOB_ID=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --job-name="build_real_synthetic_bridge_v2" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --mail-type=ALL \
  --mail-user="${MAIL_USER}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},REAL_DATA_DIR=${REAL_DATA_DIR},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},REBUILD_BRIDGE=${REBUILD_BRIDGE},MAX_ANCHORS=${MAX_ANCHORS:-0},NEGATIVES_PER_ANCHOR=${NEGATIVES_PER_ANCHOR:-4},NEGATIVE_NGRAM=${NEGATIVE_NGRAM:-3},SEED=${SEED:-42}" \
  "${PROJECT_DIR}/scripts/data/prepare_real_synthetic_bridge_v2.sh")
JOB_ID="${RAW_JOB_ID%%;*}"

echo "Bridge V2 dataset job submitted: ${JOB_ID}"
echo "Dataset target: ${BRIDGE_DATA_DIR}"
echo "Monitor: squeue -j ${JOB_ID} -o '%.18i %.35j %.2t %.10M %.55R'"
echo "After it finishes successfully, validate with:"
echo "  python scripts/data/smoke_test_real_synthetic_bridge.py --data-dir ${BRIDGE_DATA_DIR}"
echo "Only then submit model training with scripts/slurm/submit_full_research_pipeline.sh"
