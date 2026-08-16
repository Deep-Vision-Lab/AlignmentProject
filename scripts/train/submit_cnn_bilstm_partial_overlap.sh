#!/usr/bin/env bash
# Smoke-test the CNN+BiLSTM partial-overlap pipeline and submit training only on PASS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_DIR}"

EXPECTED_BRANCH="agent/cnn-bilstm-partial-overlap"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expected branch ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH:-<detached>}." >&2
  echo "Run: git switch ${EXPECTED_BRANCH} && git pull --ff-only origin ${EXPECTED_BRANCH}" >&2
  exit 2
fi

echo "=== CNN+BiLSTM PARTIAL-OVERLAP PREFLIGHT ==="
echo "branch=${CURRENT_BRANCH}"
echo "commit=$(git rev-parse --short HEAD)"

# Activate the same environment used by the public real-data launcher when possible.
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python -m py_compile \
  PartialOverlapRealAugmentation.py \
  extra_real_training.py \
  extra_real_training_partial_overlap.py \
  extra_real_training_v2.py \
  scripts/data/smoke_test_partial_overlap.py

bash -n \
  scripts/train/run_real_finetune.sh \
  scripts/train/run_real_finetune_partial_overlap.sh

SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"
python scripts/data/smoke_test_partial_overlap.py \
  --root "${DATA_DIR:-DataSet/ArabicDataset}" \
  --samples "${SMOKE_SAMPLES}"

echo "=== PREFLIGHT PASS ==="

if [[ -z "${PRETRAINED_WEIGHTS:-}" ]]; then
  echo "ERROR: PRETRAINED_WEIGHTS is required." >&2
  echo "Candidate checkpoints found locally:" >&2
  find . -path '*/model_latest.pth' -not -path './.git/*' -print 2>/dev/null | sort | tail -50 >&2 || true
  echo >&2
  echo "Re-run with:" >&2
  echo "  PRETRAINED_WEIGHTS=/path/to/model_latest.pth bash scripts/train/submit_cnn_bilstm_partial_overlap.sh" >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "ERROR: checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 2
fi

# Conservative first ablation: same sequence-ranking recipe, only train mixture changes.
export JOB_ID="${JOB_ID:-cnn_bilstm_partial_overlap_r1}"
export EPOCHS="${EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export NUM_GPUS="${NUM_GPUS:-2}"
export EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"

cat <<EOF
=== SUBMITTING PARTIAL-OVERLAP EXPERIMENT ===
JOB_ID=${JOB_ID}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS}
EPOCHS=${EPOCHS}
LEARNING_RATE=${LEARNING_RATE}
NUM_GPUS=${NUM_GPUS}
EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE}
REAL_TRAIN_SAMPLES_PER_EPOCH=${REAL_TRAIN_SAMPLES_PER_EPOCH}
NUM_NEGATIVES=${NUM_NEGATIVES}
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}
mixture=25% original positive / 25% partial-overlap positive / 50% no-shared
EOF

exec bash scripts/train/run_real_finetune_partial_overlap.sh
