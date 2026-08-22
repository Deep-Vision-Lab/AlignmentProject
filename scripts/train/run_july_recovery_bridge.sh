#!/usr/bin/env bash
# Controlled July-recovery fine-tuning on the offline real<->synthetic bridge.
#
# This is intentionally separate from the canonical launchers.  It restores the
# late-July window geometry / local supervision behavior while avoiding bridge
# objectives that incorrectly treat synthetic distractors as wholly positive.
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  SHARED_PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  SHARED_PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${SHARED_PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${SHARED_PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: bridge manifest missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}

: "${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS to the architecture-matched synthetic/base checkpoint.}"
PRETRAINED_WEIGHTS="$(readlink -f "${PRETRAINED_WEIGHTS}")"
[[ -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.35}"
LOCAL_HARD_NEGATIVE_MIN_INK="${LOCAL_HARD_NEGATIVE_MIN_INK:-0.02}"
IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.40}"
ZERO_SHOT_GROUPED_BLEND="${ZERO_SHOT_GROUPED_BLEND:-0.50}"
ZERO_SHOT_NORM_MODE="${ZERO_SHOT_NORM_MODE:-frozen-bn}"

USE_SEQUENCE_ALIGNMENT_RANKING=0
SEQUENCE_RANKING_WEIGHT=0.0
SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0.0
BRIDGE_CROSS_TEXT_WEIGHT=0.0
BRIDGE_OBJECTIVE_VARIANT=standard

REAL_USE_EXTRA_NO_SHARED=1
REAL_UNIQUE_LINE_ADAPTATION=0
REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
NO_SHARED_IMAGE_OBJECTIVE=synthetic_bridge
REAL_AUGMENT=0
AUGMENT=0
IMAGE_TEXT_LOSS_ON_BOTH_LINES=1
REAL_TRAIN_SAMPLES_PER_EPOCH=0
BRIDGE_TRAIN_SAMPLES_PER_EPOCH="${BRIDGE_TRAIN_SAMPLES_PER_EPOCH:-0}"

TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"

EPOCHS="${EPOCHS:-15}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-${MAX_TEXT_SPAN_CHARS}}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"

case "${MODEL_BACKEND}" in
  cnn_bilstm)
    CNN_BACKBONE="${CNN_BACKBONE:-resnet34}"
    USE_BILSTM=1
    BILSTM_LAYERS="${BILSTM_LAYERS:-2}"
    BILSTM_HIDDEN_DIM="${BILSTM_HIDDEN_DIM:-128}"
    USE_LOCAL_WINDOW_GROUPING=1
    LOCAL_WINDOW_GROUP_SIZE="${LOCAL_WINDOW_GROUP_SIZE:-3}"
    DEFAULT_ACCUMULATION_STEPS=1
    ;;
  vit)
    USE_BILSTM=0
    USE_LOCAL_WINDOW_GROUPING=0
    DEFAULT_ACCUMULATION_STEPS=4
    ;;
  *)
    echo "ERROR: recovery launcher supports cnn_bilstm or vit, got ${MODEL_BACKEND}." >&2
    exit 2
    ;;
esac

python - "${PRETRAINED_WEIGHTS}" "${MODEL_BACKEND}" "${CNN_BACKBONE:-}" <<'PY'
import sys, torch
path, expected_backend, requested_backbone = sys.argv[1:4]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")
config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
actual = str(config.get("model_backend", config.get("visual_encoder_type", ""))).strip().lower()
if actual in {"cnn", "cnn_only"}:
    actual = "cnn_bilstm"
if not actual:
    actual = "cnn_bilstm"
if actual != expected_backend:
    raise SystemExit(f"Checkpoint/backend mismatch: checkpoint={actual} branch={expected_backend}")
if expected_backend == "cnn_bilstm":
    checkpoint_backbone = str(config.get("cnn_backbone", "resnet34")).lower().replace("-", "")
    requested = requested_backbone.lower().replace("-", "")
    if checkpoint_backbone != requested:
        raise SystemExit(f"Recovery CNN requires {requested}; checkpoint uses {checkpoint_backbone}.")
print(f"checkpoint_ok backend={actual} backbone={config.get('cnn_backbone', '-')}")
PY

NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION_STEPS}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} must divide GPUs*accum=${DENOM}." >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOM))

JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-july-recovery-bridge-v3}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-training_runtime/july_recovery_entrypoint.py}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

TRAIN_EXPECTED_BRANCH="${TRAIN_EXPECTED_BRANCH:-$(git branch --show-current)}"
TRAIN_EXPECTED_COMMIT="${TRAIN_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"
TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"
TRAIN_SHARED_PROJECT_DIR="${TRAIN_SHARED_PROJECT_DIR:-${SHARED_PROJECT_DIR}}"

export DATA_DIR PRETRAINED_WEIGHTS MODEL_BACKEND JOB_ID TRAIN_ENTRYPOINT
export WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE NUM_NEGATIVES
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE LOCAL_HARD_NEGATIVE_WEIGHT LOCAL_HARD_NEGATIVE_MIN_INK
export IMAGE_PAIR_LOSS_WEIGHT ZERO_SHOT_GROUPED_BLEND ZERO_SHOT_NORM_MODE
export USE_SEQUENCE_ALIGNMENT_RANKING SEQUENCE_RANKING_WEIGHT SEQUENCE_CONSISTENCY_LOSS_WEIGHT
export BRIDGE_CROSS_TEXT_WEIGHT BRIDGE_OBJECTIVE_VARIANT
export REAL_USE_EXTRA_NO_SHARED REAL_UNIQUE_LINE_ADAPTATION REAL_USE_EXPLICIT_SPLIT_MANIFESTS
export NO_SHARED_IMAGE_OBJECTIVE REAL_AUGMENT AUGMENT IMAGE_TEXT_LOSS_ON_BOTH_LINES
export REAL_TRAIN_SAMPLES_PER_EPOCH BRIDGE_TRAIN_SAMPLES_PER_EPOCH
export TARGET_INK_HEIGHT_RATIO ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP ZERO_SHOT_SOURCE_GEOMETRY
export EPOCHS LEARNING_RATE VALID_EVERY_N_EPOCHS VALID_MAX_BATCHES
export MAX_TEXT_SPAN_CHARS REAL_MAX_TEXT_SPAN_CHARS MAX_WINDOWS_PER_SPAN
export CNN_BACKBONE USE_BILSTM BILSTM_LAYERS BILSTM_HIDDEN_DIM USE_LOCAL_WINDOW_GROUPING LOCAL_WINDOW_GROUP_SIZE
export NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS BATCH_SIZE
export CONDA_ENV GPU_RESOURCE PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT MAIL_USER
export TRAIN_EXPECTED_BRANCH TRAIN_EXPECTED_COMMIT TRAIN_EXPECTED_BACKEND TRAIN_SHARED_PROJECT_DIR

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cat <<EOF
=== JULY RECOVERY BRIDGE SUBMISSION ===
branch=${TRAIN_EXPECTED_BRANCH}
commit=${TRAIN_EXPECTED_COMMIT}
backend=${MODEL_BACKEND}
checkpoint=${PRETRAINED_WEIGHTS}
dataset=${DATA_DIR}
job=${JOB_ID}
window/stride=${WINDOW_SIZE}/$(python - <<PY
print(max(1, int(${WINDOW_SIZE} * ${STRIDE_RATIO})))
PY
)
negatives=${NUM_NEGATIVES}
local_hard_negative_weight=${LOCAL_HARD_NEGATIVE_WEIGHT}
image_pair_weight=${IMAGE_PAIR_LOSS_WEIGHT}
grouped_blend=${ZERO_SHOT_GROUPED_BLEND}
norm_mode=${ZERO_SHOT_NORM_MODE}
whole_line_sequence_ranking=OFF
bridge_cross_text=OFF
entrypoint=${TRAIN_ENTRYPOINT}
EOF
  sbatch \
    --partition="${PARTITION}" \
    --job-name="${JOB_ID}" \
    --output="${SHARED_PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${SHARED_PROJECT_DIR}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL \
    "${SCRIPT_PATH}"
  exit 0
fi

if [[ "${RECOVERY_RUNTIME_CLONE_ACTIVE:-0}" != "1" ]]; then
  if [[ -n "${SLURM_SCRATCH_DIR:-}" && -d "${SLURM_SCRATCH_DIR}" ]]; then
    RUNTIME_PARENT="${SLURM_SCRATCH_DIR}"
  elif [[ -d "/scratch/${USER}/${SLURM_JOB_ID}" ]]; then
    RUNTIME_PARENT="/scratch/${USER}/${SLURM_JOB_ID}"
  else
    RUNTIME_PARENT="${TMPDIR:-/tmp}"
  fi
  RUNTIME_PROJECT_DIR="${RUNTIME_PARENT}/AlignmentProject-${SLURM_JOB_ID}-${TRAIN_EXPECTED_COMMIT:0:12}"
  rm -rf "${RUNTIME_PROJECT_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --shared --no-checkout \
    "${TRAIN_SHARED_PROJECT_DIR}" "${RUNTIME_PROJECT_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${RUNTIME_PROJECT_DIR}" checkout --quiet \
    -B "${TRAIN_EXPECTED_BRANCH}" "${TRAIN_EXPECTED_COMMIT}"

  for name in DataSet Weights out logs .hf_cache .jax_cache wandb; do
    source_path="${TRAIN_SHARED_PROJECT_DIR}/${name}"
    target_path="${RUNTIME_PROJECT_DIR}/${name}"
    if [[ -e "${source_path}" || -L "${source_path}" ]]; then
      rm -rf "${target_path}"
      ln -s "${source_path}" "${target_path}"
    fi
  done

  exec env \
    PROJECT_DIR="${RUNTIME_PROJECT_DIR}" \
    TRAIN_SHARED_PROJECT_DIR="${TRAIN_SHARED_PROJECT_DIR}" \
    RECOVERY_RUNTIME_CLONE_ACTIVE=1 \
    bash "${RUNTIME_PROJECT_DIR}/scripts/train/run_july_recovery_bridge.sh"
fi

cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_ASYNC_ERROR_HANDLING=1

ARGS=(
  "${TRAIN_ENTRYPOINT}"
  --job_id "${JOB_ID}"
  --dataset_type real
  --data_dir "${DATA_DIR}"
  --pretrained_weights "${PRETRAINED_WEIGHTS}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --window_size "${WINDOW_SIZE}"
  --stride_ratio "${STRIDE_RATIO}"
  --window_overlap_mode "${WINDOW_OVERLAP_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  --no-augment
)

if (( NUM_GPUS > 1 )); then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${ARGS[@]}"
fi
exec python "${ARGS[@]}"
