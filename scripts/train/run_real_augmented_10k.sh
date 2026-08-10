#!/usr/bin/env bash
# Fine-tune the current branch backend on the already-built real+injection 10K dataset.
set -euo pipefail
set -a

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

EXPECTED_BRANCH="agent/use-vit-encoder"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: ViT real training must be launched from ${EXPECTED_BRANCH}; current branch=${CURRENT_BRANCH}." >&2
  echo "ERROR: do not switch the shared checkout while this Slurm job is queued or running." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDatasetRealAug10K}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
for manifest in dataset_manifest.jsonl train_manifest.jsonl valid_manifest.jsonl test_manifest.jsonl; do
  [[ -f "${DATA_DIR}/${manifest}" ]] || {
    echo "ERROR: missing ${DATA_DIR}/${manifest}. Build ArabicDatasetRealAug10K first." >&2
    exit 2
  }
done

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
[[ "${MODEL_BACKEND}" == "vit" ]] || {
  echo "ERROR: ${EXPECTED_BRANCH} must resolve model_backend=vit, got ${MODEL_BACKEND}." >&2
  exit 2
}

TRAIN_EXPECTED_BRANCH="${EXPECTED_BRANCH}"
TRAIN_EXPECTED_BACKEND="vit"
TRAIN_EXPECTED_COMMIT="$(git rev-parse HEAD)"

DEFAULT_SOURCE_RUN="vit_augmented_fixed63_27k_v2"
DEFAULT_JOB_ID="vit_real_aug10k"
JOB_ID="${JOB_ID:-${DEFAULT_JOB_ID}}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
CHECKPOINT_SELECTOR="${PROJECT_DIR}/scripts/train/select_compatible_pretrained.py"
[[ -f "${CHECKPOINT_SELECTOR}" ]] || {
  echo "ERROR: checkpoint selector missing: ${CHECKPOINT_SELECTOR}" >&2
  exit 2
}

# Never trust a folder name alone. Validate the actual image-model state-dict
# keys before requesting GPUs. With no explicit checkpoint, prefer the clean
# synthetic pretraining run and then scan other backend-named synthetic runs.
if [[ -n "${PRETRAINED_WEIGHTS:-}" ]]; then
  PRETRAINED_WEIGHTS="$(python "${CHECKPOINT_SELECTOR}" \
    --weights-root "${PROJECT_DIR}/Weights" \
    --backend "${MODEL_BACKEND}" \
    --window-size "${WINDOW_SIZE}" \
    --checkpoint "${PRETRAINED_WEIGHTS}")"
else
  PRETRAINED_WEIGHTS="$(python "${CHECKPOINT_SELECTOR}" \
    --weights-root "${PROJECT_DIR}/Weights" \
    --backend "${MODEL_BACKEND}" \
    --window-size "${WINDOW_SIZE}" \
    --preferred-dir "${DEFAULT_SOURCE_RUN}")"
fi
PRETRAINED_WEIGHTS="$(readlink -f "${PRETRAINED_WEIGHTS}")"

# The dataset already contains the bbox-strip augmentation physically. Load the
# explicit train/valid/test manifests written by the builder and do not re-split
# or add another online augmentation layer. Keep the full feasible training pool,
# but draw a fresh random subset each epoch when the requested epoch size is
# smaller than that pool.
AUGMENT=0
REAL_AUGMENT=0
REAL_USE_EXPLICIT_SPLIT_MANIFESTS=1
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
REAL_MIN_TEXT_SCORE=0.0
REAL_SPLIT_BY_PAIR_ID=0
REAL_MANIFEST_NAME=dataset_manifest.jsonl
WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-aug10k}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"

export PROJECT_DIR DATA_DIR MODEL_BACKEND JOB_ID PRETRAINED_WEIGHTS WINDOW_SIZE
export AUGMENT REAL_AUGMENT REAL_USE_EXPLICIT_SPLIT_MANIFESTS
export REAL_TRAIN_SAMPLES_PER_EPOCH EFFECTIVE_GLOBAL_BATCH_SIZE NUM_SAMPLES REAL_DATASET_LABELS
export REAL_MIN_TEXT_SCORE REAL_SPLIT_BY_PAIR_ID REAL_MANIFEST_NAME WANDB_PROJECT TIME_LIMIT
export TRAIN_EXPECTED_BRANCH TRAIN_EXPECTED_BACKEND TRAIN_EXPECTED_COMMIT

has_gpu_allocation() {
  local name value
  for name in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS SLURM_GPU_INDEX; do
    value="${!name:-}"
    if [[ -n "${value}" && "${value}" != "NoDevFiles" && "${value}" != "(null)" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${SLURM_JOB_ID:-}" ]] && ! has_gpu_allocation; then
  echo "Detected CPU-only Slurm context ${SLURM_JOB_ID}; submitting the GPU training job instead."
  unset SLURM_JOB_ID SLURM_STEP_ID SLURM_STEP_GPUS SLURM_JOB_GPUS \
    SLURM_GPU_INDEX CUDA_VISIBLE_DEVICES
fi

printf '%s\n' \
  "Real+augmented 10K training" \
  "  backend=${MODEL_BACKEND}" \
  "  branch=${TRAIN_EXPECTED_BRANCH}" \
  "  pinned commit=${TRAIN_EXPECTED_COMMIT}" \
  "  dataset=${DATA_DIR}" \
  "  train manifest=${DATA_DIR}/train_manifest.jsonl" \
  "  valid manifest=${DATA_DIR}/valid_manifest.jsonl" \
  "  test manifest=${DATA_DIR}/test_manifest.jsonl" \
  "  train samples/epoch=${REAL_TRAIN_SAMPLES_PER_EPOCH}" \
  "  effective global batch=${EFFECTIVE_GLOBAL_BATCH_SIZE}" \
  "  sampling=epoch-random subset when target < feasible train pool" \
  "  online augmentation=disabled" \
  "  pretrained=${PRETRAINED_WEIGHTS}" \
  "  time limit=${TIME_LIMIT}" \
  "  job id=${JOB_ID}"

exec bash "${PROJECT_DIR}/scripts/train/run_real_finetune.sh"
