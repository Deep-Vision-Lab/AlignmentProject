#!/usr/bin/env bash
# Fine-tune the current branch backend on the already-built real+injection 10K dataset.
set -euo pipefail
set -a

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

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
case "${MODEL_BACKEND}" in
  cnn_bilstm)
    DEFAULT_SOURCE_RUN="cnn_bilstm_augmented_fixed63_27k"
    DEFAULT_JOB_ID="cnn_bilstm_real_aug10k"
    ;;
  vit)
    DEFAULT_SOURCE_RUN="vit_augmented_fixed63_27k"
    DEFAULT_JOB_ID="vit_real_aug10k"
    ;;
  *)
    echo "ERROR: unsupported model backend ${MODEL_BACKEND}." >&2
    exit 2
    ;;
esac

JOB_ID="${JOB_ID:-${DEFAULT_JOB_ID}}"
if [[ -z "${PRETRAINED_WEIGHTS:-}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/Weights/${DEFAULT_SOURCE_RUN}/model_best.pth" \
    "${PROJECT_DIR}/Weights/${DEFAULT_SOURCE_RUN}/model_latest.pth" \
    "${PROJECT_DIR}/Weights/${DEFAULT_SOURCE_RUN}/checkpoint_latest.pth"; do
    if [[ -f "${candidate}" ]]; then
      PRETRAINED_WEIGHTS="${candidate}"
      break
    fi
  done
fi
: "${PRETRAINED_WEIGHTS:?Could not find the synthetic pretrained checkpoint for ${MODEL_BACKEND}. Set PRETRAINED_WEIGHTS explicitly.}"
PRETRAINED_WEIGHTS="$(readlink -f "${PRETRAINED_WEIGHTS}")"

# The dataset already contains the bbox-strip augmentation physically.  Load the
# explicit train/valid/test manifests written by the builder and do not re-split
# or add another online augmentation layer.
AUGMENT=0
REAL_AUGMENT=0
REAL_USE_EXPLICIT_SPLIT_MANIFESTS=1
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-10000}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
REAL_MIN_TEXT_SCORE=0.0
REAL_SPLIT_BY_PAIR_ID=0
REAL_MANIFEST_NAME=dataset_manifest.jsonl
WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-aug10k}"

export PROJECT_DIR DATA_DIR MODEL_BACKEND JOB_ID PRETRAINED_WEIGHTS
export AUGMENT REAL_AUGMENT REAL_USE_EXPLICIT_SPLIT_MANIFESTS
export REAL_TRAIN_SAMPLES_PER_EPOCH NUM_SAMPLES REAL_DATASET_LABELS
export REAL_MIN_TEXT_SCORE REAL_SPLIT_BY_PAIR_ID REAL_MANIFEST_NAME WANDB_PROJECT

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
  "  dataset=${DATA_DIR}" \
  "  train manifest=${DATA_DIR}/train_manifest.jsonl" \
  "  valid manifest=${DATA_DIR}/valid_manifest.jsonl" \
  "  test manifest=${DATA_DIR}/test_manifest.jsonl" \
  "  train samples/epoch=${REAL_TRAIN_SAMPLES_PER_EPOCH}" \
  "  online augmentation=disabled" \
  "  pretrained=${PRETRAINED_WEIGHTS}" \
  "  job id=${JOB_ID}"

exec bash "${PROJECT_DIR}/scripts/train/run_real_finetune.sh"
