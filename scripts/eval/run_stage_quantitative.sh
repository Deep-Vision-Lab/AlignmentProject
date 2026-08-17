#!/usr/bin/env bash
# Canonical quantitative real evaluation: bbox/localization + fixed discrimination sweep.
set -euo pipefail
set -a
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out
: "${CHECKPOINT:?Set CHECKPOINT to the checkpoint to evaluate.}"
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing ${CHECKPOINT}" >&2; exit 2; }
RUN_TAG="${RUN_TAG:-quantitative_$(basename "$(dirname "${CHECKPOINT}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_TAG}}"
DIAG_SAMPLES="${DIAG_SAMPLES:-100}"
BBOX_SAMPLES="${BBOX_SAMPLES:-2000}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-10:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  DEP_ARGS=(); [[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS+=(--dependency="${DEPENDENCY}")
  sbatch --job-name="quant_${RUN_TAG}" --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}" \
    --mail-type=ALL --mail-user="${MAIL_USER}" "${DEP_ARGS[@]}" --export=ALL "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${RESULTS_ROOT}"

# A. Held-out positive localization/bbox metrics.
WEIGHTS="${CHECKPOINT}" \
RUN_TAG="${RUN_TAG}_bbox" \
LABELS="high_match,medium_match" \
N_SAMPLES="${BBOX_SAMPLES}" \
REAL_SPLIT=test \
RESULTS_ROOT="${RESULTS_ROOT}/bbox" \
bash Evaluation/evaluate_quantitative_no_png.sh

# B. Exact deterministic positive-vs-no-shared diagnostic sweep.
CHECKPOINT="${CHECKPOINT}" \
RUN_NAME="${RUN_TAG}_discrimination" \
N_SAMPLES="${DIAG_SAMPLES}" \
OUT="${RESULTS_ROOT}/discrimination" \
bash scripts/eval/run_real_discrimination_sweep.sh

echo "QUANTITATIVE_RESULTS=${RESULTS_ROOT}"
