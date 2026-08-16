#!/usr/bin/env bash
# Preflight and submit the corrected Stage-1 -> clean-real synthetic-partner run.
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

echo "=== CNN+BiLSTM CLEAN SYNTHETIC-PARTNER PREFLIGHT ==="
echo "branch=${CURRENT_BRANCH}"
echo "commit=$(git rev-parse --short HEAD)"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python -m py_compile \
  SyntheticPartnerRealAugmentation.py \
  extra_real_training.py \
  extra_real_training_partial_overlap.py \
  extra_real_training_v2.py \
  scripts/data/build_no_shared_synthetic_partners.py \
  scripts/data/smoke_test_partial_overlap.py

bash -n \
  scripts/train/run_real_finetune.sh \
  scripts/train/run_real_finetune_partial_overlap.sh

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REAL_SYNTHETIC_PARTNER_MANIFEST="${REAL_SYNTHETIC_PARTNER_MANIFEST:-${PROJECT_DIR}/DataSet/ArabicDatasetSyntheticPartners/dataset_manifest.jsonl}"
export DATA_DIR REAL_SYNTHETIC_PARTNER_MANIFEST

if [[ ! -f "${REAL_SYNTHETIC_PARTNER_MANIFEST}" ]]; then
  echo "ERROR: synthetic-partner manifest has not been built yet:" >&2
  echo "  ${REAL_SYNTHETIC_PARTNER_MANIFEST}" >&2
  echo >&2
  echo "Build it first on a CPU node:" >&2
  echo "  python scripts/data/build_no_shared_synthetic_partners.py --data-dir ${DATA_DIR} --output-dir $(dirname "${REAL_SYNTHETIC_PARTNER_MANIFEST}") --overwrite" >&2
  exit 2
fi

SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"
python scripts/data/smoke_test_partial_overlap.py \
  --root "${DATA_DIR}" \
  --synthetic-manifest "${REAL_SYNTHETIC_PARTNER_MANIFEST}" \
  --samples "${SMOKE_SAMPLES}"

echo "=== PREFLIGHT PASS ==="

if [[ -z "${PRETRAINED_WEIGHTS:-}" ]]; then
  echo "ERROR: PRETRAINED_WEIGHTS must be the chosen Stage-1 checkpoint." >&2
  echo "Candidate checkpoints found locally:" >&2
  find . -path '*/model_latest.pth' -not -path './.git/*' -print 2>/dev/null | sort | tail -80 >&2 || true
  exit 2
fi
if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "ERROR: checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 2
fi

case "${PRETRAINED_WEIGHTS,,}" in
  *phase3*|*stage2*|*joint_real*)
    echo "WARNING: checkpoint path looks like a later real-data stage:" >&2
    echo "  ${PRETRAINED_WEIGHTS}" >&2
    echo "This experiment was defined to initialize from Stage 1." >&2
    echo "Set ALLOW_NON_STAGE1_CHECKPOINT=1 only if this naming is misleading." >&2
    [[ "${ALLOW_NON_STAGE1_CHECKPOINT:-0}" == "1" ]] || exit 2
    ;;
esac

export JOB_ID="${JOB_ID:-cnn_bilstm_stage1_clean_synthetic_partner_r1}"
export EPOCHS="${EPOCHS:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export NUM_GPUS="${NUM_GPUS:-2}"
export EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
# Keep the clean Stage-2 behavior: use every natural real/synthetic-partner row
# once per epoch unless the user explicitly asks to repeat the complete mixture.
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export AUGMENT=0
export REAL_AUGMENT=0
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0

cat <<EOF
=== SUBMITTING CLEAN SYNTHETIC-PARTNER EXPERIMENT ===
JOB_ID=${JOB_ID}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS}
SYNTHETIC_PARTNER_MANIFEST=${REAL_SYNTHETIC_PARTNER_MANIFEST}
EPOCHS=${EPOCHS}
LEARNING_RATE=${LEARNING_RATE}
NUM_GPUS=${NUM_GPUS}
EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE}
REAL_TRAIN_SAMPLES_PER_EPOCH=${REAL_TRAIN_SAMPLES_PER_EPOCH}
NUM_NEGATIVES=${NUM_NEGATIVES}
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}
generic_real_augmentation=OFF
sequence_consistency_global_order=OFF
recipe=clean positives + one-sided synthetic-partner positives + clean no-shared negatives
EOF

exec bash scripts/train/run_real_finetune_partial_overlap.sh
