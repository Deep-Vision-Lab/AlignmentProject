#!/usr/bin/env bash
# Submit the one-time RealSyntheticBridge V3 dataset build/audit job with tracking.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs/experiments

BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
CPU_PARTITION="${CPU_PARTITION:-main}"
# The generator launches one process per allocated CPU by default.
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
BRIDGE_BUILD_WORKERS="${BRIDGE_BUILD_WORKERS:-${CPUS_PER_TASK}}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"
BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
RUN_PREFIX="${DATASET_RUN_PREFIX:-bridge_v3_dataset_$(date +%Y%m%d_%H%M%S)}"
TRACKER_JSON="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.json"
TRACKER_MD="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.md"
TRACKER_TOOL="${PROJECT_DIR}/scripts/pipeline/experiment_tracker.py"
TRACKED_WRAPPER="${PROJECT_DIR}/scripts/pipeline/run_tracked_stage.sh"
STAGE_SCRIPT="${PROJECT_DIR}/scripts/data/prepare_real_synthetic_bridge_v3.sh"
JOB_NAME="build_real_synthetic_bridge_v3"

[[ -s "${REAL_DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing real manifest" >&2; exit 2; }

python "${TRACKER_TOOL}" init \
  --tracker "${TRACKER_JSON}" --run-prefix "${RUN_PREFIX}" \
  --branch "${BRANCH}" --commit "${COMMIT}" --backend dataset-preparation \
  --bridge-dataset "${BRIDGE_DATA_DIR}" >/dev/null
python "${TRACKER_TOOL}" register \
  --tracker "${TRACKER_JSON}" --stage D0 --description "Parallel build and audit frozen RealSyntheticBridge V3" \
  --kind dataset --job-name "${JOB_NAME}" --artifact "${BRIDGE_DATA_DIR}" >/dev/null

RAW_JOB_ID=$(sbatch --parsable \
  --partition="${CPU_PARTITION}" \
  --job-name="${JOB_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},REAL_DATA_DIR=${REAL_DATA_DIR},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},REBUILD_BRIDGE=${REBUILD_BRIDGE},MAX_ANCHORS=${MAX_ANCHORS:-0},NEGATIVES_PER_ANCHOR=${NEGATIVES_PER_ANCHOR:-8},NEGATIVE_NGRAM=${NEGATIVE_NGRAM:-3},SEED=${SEED:-42},BRIDGE_BUILD_WORKERS=${BRIDGE_BUILD_WORKERS},PIPELINE_TRACKER_JSON=${TRACKER_JSON},PIPELINE_STAGE=D0,TRACKED_STAGE_SCRIPT=${STAGE_SCRIPT}" \
  "${TRACKED_WRAPPER}")
JOB_ID="${RAW_JOB_ID%%;*}"
LOG_PATH="${PROJECT_DIR}/out/${JOB_NAME}_${JOB_ID}.out"

python "${TRACKER_TOOL}" register \
  --tracker "${TRACKER_JSON}" --stage D0 --description "Parallel build and audit frozen RealSyntheticBridge V3" \
  --kind dataset --job-id "${JOB_ID}" --job-name "${JOB_NAME}" \
  --log-path "${LOG_PATH}" --artifact "${BRIDGE_DATA_DIR}" >/dev/null

cat <<EOF
=== BRIDGE V3 DATASET JOB SUBMITTED ===
job_name=${JOB_NAME}
job_id=${JOB_ID}
dataset=${BRIDGE_DATA_DIR}
cpus=${CPUS_PER_TASK}
parallel_workers=${BRIDGE_BUILD_WORKERS}
memory=${MEMORY}
negatives_per_anchor=${NEGATIVES_PER_ANCHOR:-8}
log=${LOG_PATH}
tracker=${TRACKER_MD}
tracker_state=${TRACKER_JSON}

Open tracker:
  cat ${TRACKER_MD}

Monitor:
  squeue -j ${JOB_ID} -o '%.18i %.35j %.2t %.10M %.55R'

Synthetic S1-S3 may run in parallel with D0. Bridge-dependent S4+ must depend on D0 success.
EOF
