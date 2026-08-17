#!/usr/bin/env bash
# Submit the one-time RealSyntheticBridge V2 dataset build/audit job with tracking.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"; mkdir -p out logs/experiments
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"; REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
CPU_PARTITION="${CPU_PARTITION:-main}"; CPUS_PER_TASK="${CPUS_PER_TASK:-8}"; MEMORY="${MEMORY:-32G}"; TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"; MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"; REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"
BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; RUN_PREFIX="${DATASET_RUN_PREFIX:-bridge_v2_dataset_$(date +%Y%m%d_%H%M%S)}"
TRACKER_JSON="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.json"; TRACKER_MD="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.md"; TRACKER_TOOL="${PROJECT_DIR}/scripts/pipeline/experiment_tracker.py"; TRACKED_WRAPPER="${PROJECT_DIR}/scripts/pipeline/run_tracked_stage.sh"; STAGE_SCRIPT="${PROJECT_DIR}/scripts/data/prepare_real_synthetic_bridge_v2.sh"; JOB_NAME="build_real_synthetic_bridge_v2"
[[ -s "${REAL_DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing real manifest: ${REAL_DATA_DIR}/dataset_manifest.jsonl" >&2; exit 2; }
python "${TRACKER_TOOL}" init --tracker "${TRACKER_JSON}" --run-prefix "${RUN_PREFIX}" --branch "${BRANCH}" --commit "${COMMIT}" --backend dataset-preparation --bridge-dataset "${BRIDGE_DATA_DIR}" >/dev/null
python "${TRACKER_TOOL}" register --tracker "${TRACKER_JSON}" --stage D0 --description "Build and audit frozen RealSyntheticBridge V2" --kind dataset --job-name "${JOB_NAME}" --artifact "${BRIDGE_DATA_DIR}" >/dev/null
RAW_JOB_ID=$(sbatch --parsable --partition="${CPU_PARTITION}" --job-name="${JOB_NAME}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" --export="ALL,PROJECT_DIR=${PROJECT_DIR},REAL_DATA_DIR=${REAL_DATA_DIR},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},REBUILD_BRIDGE=${REBUILD_BRIDGE},MAX_ANCHORS=${MAX_ANCHORS:-0},NEGATIVES_PER_ANCHOR=${NEGATIVES_PER_ANCHOR:-4},NEGATIVE_NGRAM=${NEGATIVE_NGRAM:-3},SEED=${SEED:-42},PIPELINE_TRACKER_JSON=${TRACKER_JSON},PIPELINE_STAGE=D0,TRACKED_STAGE_SCRIPT=${STAGE_SCRIPT}" "${TRACKED_WRAPPER}")
JOB_ID="${RAW_JOB_ID%%;*}"; LOG_PATH="${PROJECT_DIR}/out/${JOB_NAME}_${JOB_ID}.out"
python "${TRACKER_TOOL}" register --tracker "${TRACKER_JSON}" --stage D0 --description "Build and audit frozen RealSyntheticBridge V2" --kind dataset --job-id "${JOB_ID}" --job-name "${JOB_NAME}" --log-path "${LOG_PATH}" --artifact "${BRIDGE_DATA_DIR}" >/dev/null
cat <<EOF
=== BRIDGE V2 DATASET JOB SUBMITTED ===
job_name=${JOB_NAME}
job_id=${JOB_ID}
dataset=${BRIDGE_DATA_DIR}
log=${LOG_PATH}
tracker=${TRACKER_MD}
tracker_state=${TRACKER_JSON}
Only after D0 shows ✅ COMPLETED / PASS should you submit model training.
EOF
