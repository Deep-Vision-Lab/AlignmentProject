#!/usr/bin/env bash
# Train on canonical positive real pairs plus safe training-side
# no_shared_content rows used only for per-line image-text supervision.
set -euo pipefail
set -a

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

EXPECTED_BRANCH="agent/use-extra-real-lines-cnn"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expanded genuine-real training must be launched from ${EXPECTED_BRANCH}; current=${CURRENT_BRANCH}." >&2
  exit 2
fi

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
[[ "${MODEL_BACKEND}" == "cnn_bilstm" ]] || {
  echo "ERROR: ${EXPECTED_BRANCH} must resolve model_backend=cnn_bilstm, got ${MODEL_BACKEND}." >&2
  exit 2
}

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -f "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: missing ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}

TRAIN_EXPECTED_BRANCH="${EXPECTED_BRANCH}"
TRAIN_EXPECTED_BACKEND="cnn_bilstm"
TRAIN_EXPECTED_COMMIT="$(git rev-parse HEAD)"

REAL_USE_EXTRA_NO_SHARED=1
REAL_EXTRA_EXCLUDE_EVAL_PAGES="${REAL_EXTRA_EXCLUDE_EVAL_PAGES:-1}"
REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
REAL_DATASET_LABELS="high_match,medium_match"
REAL_SPLIT_BY_PAIR_ID=1
NUM_SAMPLES=0
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
REAL_AUG_STITCH_PROB=0
AUGMENT="${AUGMENT:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-expanded}"

export PROJECT_DIR DATA_DIR MODEL_BACKEND
export TRAIN_EXPECTED_BRANCH TRAIN_EXPECTED_BACKEND TRAIN_EXPECTED_COMMIT
export REAL_USE_EXTRA_NO_SHARED REAL_EXTRA_EXCLUDE_EVAL_PAGES
export REAL_USE_EXPLICIT_SPLIT_MANIFESTS REAL_DATASET_LABELS REAL_SPLIT_BY_PAIR_ID
export NUM_SAMPLES REAL_TRAIN_SAMPLES_PER_EPOCH REAL_AUG_STITCH_PROB AUGMENT WANDB_PROJECT

printf '%s\n' \
  "Expanded genuine-real training" \
  "  branch=${TRAIN_EXPECTED_BRANCH}" \
  "  pinned commit=${TRAIN_EXPECTED_COMMIT}" \
  "  dataset=${DATA_DIR}" \
  "  use no_shared real lines=1" \
  "  exclude validation/test pages=${REAL_EXTRA_EXCLUDE_EVAL_PAGES}" \
  "  pair loss on no_shared rows=masked" \
  "  online augmentation=${AUGMENT}" \
  "  stitch/injection=disabled" \
  "  train samples/epoch=${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"

exec bash "${PROJECT_DIR}/scripts/train/run_real_finetune.sh"
