#!/usr/bin/env bash
# Submit Bridge V3 D0 and the synthetic S1 branch concurrently.
# Independent evaluations are parallelized; only true data/weight prerequisites use
# afterok dependencies.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
case "${BRANCH}" in
  agent/training-speed-optimization|agent/use-vit-encoder|agent/use-dinov3-convnext) ;;
  *) echo "ERROR: run from a canonical architecture branch; got ${BRANCH}" >&2; exit 2 ;;
esac

BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

RUN_PREFIX="${RUN_PREFIX:-${BACKEND}_research_v3_$(date +%Y%m%d_%H%M%S)}"
SYNTH_EPOCHS="${SYNTH_EPOCHS:-20}"
BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-25}"
BRIDGE_LR="${BRIDGE_LR:-1e-6}"
FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
DATASET_CPUS="${DATASET_CPUS:-32}"
BRIDGE_BUILD_WORKERS="${BRIDGE_BUILD_WORKERS:-${DATASET_CPUS}}"
DATASET_MEMORY="${DATASET_MEMORY:-64G}"

# D0 starts now on CPU. Force rebuild so the current dense rendering and real
# augmentation policies cannot silently reuse an older Bridge directory.
DATASET_OUTPUT="$({
  DATASET_RUN_PREFIX="${RUN_PREFIX}_D0" \
  REBUILD_BRIDGE="${REBUILD_BRIDGE:-1}" \
  BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR}" \
  NEGATIVES_PER_ANCHOR="${NEGATIVES_PER_ANCHOR:-8}" \
  CPUS_PER_TASK="${DATASET_CPUS}" \
  BRIDGE_BUILD_WORKERS="${BRIDGE_BUILD_WORKERS}" \
  MEMORY="${DATASET_MEMORY}" \
  bash scripts/slurm/submit_bridge_v3_dataset.sh
} 2>&1)"
printf '%s\n' "${DATASET_OUTPUT}"
D0_JOB_ID="$(awk -F= '/^job_id=/{print $2; exit}' <<<"${DATASET_OUTPUT}" | tr -d '[:space:]')"
if [[ -z "${D0_JOB_ID}" || ! "${D0_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: could not parse D0 job id from dataset submitter output" >&2
  exit 2
fi

# S1 is submitted immediately and is independent of D0. The V3 pipeline wrapper
# makes S2/S3 siblings after S1, S4 depend on S1+D0, S5 depend on S4, and the three
# post-Bridge evaluations siblings after S5. S8 is the join/final stage.
PIPELINE_OUTPUT="$({
  DEFER_BRIDGE_VALIDATION=1 \
  BRIDGE_READY_JOB_ID="${D0_JOB_ID}" \
  BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR}" \
  RUN_PREFIX="${RUN_PREFIX}" \
  SYNTH_EPOCHS="${SYNTH_EPOCHS}" \
  BRIDGE_EPOCHS="${BRIDGE_EPOCHS}" \
  BRIDGE_LR="${BRIDGE_LR}" \
  FINAL_THRESHOLD="${FINAL_THRESHOLD}" \
  bash scripts/slurm/submit_full_research_pipeline_v3.sh
} 2>&1)"
printf '%s\n' "${PIPELINE_OUTPUT}"

LEDGER="${PROJECT_DIR}/logs/research_pipeline_${RUN_PREFIX}.jobs"
TRACKER="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.md"

cat <<EOF
=== MAX-PARALLEL D0 + RESEARCH PIPELINE SUBMITTED ===
branch=${BRANCH}
commit=${COMMIT}
backend=${BACKEND}
run_prefix=${RUN_PREFIX}

D0_dataset_job=${D0_JOB_ID}
D0_dataset=${BRIDGE_DATA_DIR}
D0_cpus=${DATASET_CPUS}
D0_parallel_workers=${BRIDGE_BUILD_WORKERS}
D0_memory=${DATASET_MEMORY}

synthetic_epochs=${SYNTH_EPOCHS}
bridge_epochs=${BRIDGE_EPOCHS}
bridge_lr=${BRIDGE_LR}
tracker=${TRACKER}
ledger=${LEDGER}

Dependency graph:

                      D0: parallel Bridge V3 build
                     /                         \
START --------------                             S4 Bridge pre-eval -> S5 Bridge train
                     \\                         /                         |     |     |
                      S1 synthetic train -------------------------------  |     |     |
                         |             |                                  |     |     |
                         +-> S2 qual   +-> S3 quant                       S6   S7A   S7B
                                  (S2/S3 parallel)                         \     |     /
                                                                            S8 final

Concurrency rules:
  * D0 and S1 can run at the same time.
  * S2 and S3 both depend only on S1, so they can run at the same time.
  * S4 depends on BOTH D0 and S1, not on S2/S3.
  * S5 starts after S4 and may overlap S2/S3 if those evaluations are still running.
  * S6, S7A, and S7B all depend only on S5 and can run concurrently.
  * S8 waits for S2, S3, S6, S7A, and S7B so it is genuinely the final stage.
  * Any failed prerequisite blocks only the branch that depends on it.

Monitor all jobs and their dependencies:
  squeue -u "$USER" -o '%.18i %.42j %.2t %.10M %.25E %.45R'

Job map:
  cat ${LEDGER}

Experiment tracker:
  cat ${TRACKER}
EOF
