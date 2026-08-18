#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"; mkdir -p out logs
BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"
case "${BRANCH}" in agent/training-speed-optimization|agent/use-vit-encoder|agent/use-dinov3-convnext) ;; *) echo "ERROR: non-canonical branch ${BRANCH}" >&2; exit 2;; esac
BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
RUN_PREFIX="${RUN_PREFIX:-${BACKEND}_research_v3_$(date +%Y%m%d_%H%M%S)}"; SYNTH_EPOCHS="${SYNTH_EPOCHS:-20}"; BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-25}"; BRIDGE_LR="${BRIDGE_LR:-1e-6}"; FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}"; CPU_PARTITION="${CPU_PARTITION:-main}"; MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"; BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
DATASET_OUTPUT="$({ REBUILD_BRIDGE="${REBUILD_BRIDGE:-1}" BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR}" NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-8}" bash scripts/slurm/submit_bridge_v3_dataset.sh; } 2>&1)"; printf '%s\n' "${DATASET_OUTPUT}"
D0_JOB_ID="$(awk -F= '/^job_id=/{print $2; exit}' <<<"${DATASET_OUTPUT}" | tr -d '[:space:]')"; [[ "${D0_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: could not parse D0 job id" >&2; exit 2; }
LAUNCHER_NAME="${RUN_PREFIX}_launch_after_D0"; RAW="$(sbatch --parsable --partition="${CPU_PARTITION}" --dependency="afterok:${D0_JOB_ID}" --job-name="${LAUNCHER_NAME}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:30:00 --mail-type=ALL --mail-user="${MAIL_USER}" --export="ALL,PROJECT_DIR=${PROJECT_DIR},EXPECTED_BRANCH=${BRANCH},EXPECTED_COMMIT=${COMMIT},RUN_PREFIX=${RUN_PREFIX},SYNTH_EPOCHS=${SYNTH_EPOCHS},BRIDGE_EPOCHS=${BRIDGE_EPOCHS},BRIDGE_LR=${BRIDGE_LR},FINAL_THRESHOLD=${FINAL_THRESHOLD},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR}" "${PROJECT_DIR}/scripts/slurm/run_pipeline_submit_after_dataset_v3.sh")"; LID="${RAW%%;*}"
echo "D0_dataset_job=${D0_JOB_ID}"; echo "launcher_job=${LID}"; echo "launcher_dependency=afterok:${D0_JOB_ID}"; echo "run_prefix=${RUN_PREFIX}"; echo "IMPORTANT: keep checkout on ${BRANCH} at ${COMMIT} until launcher runs."
