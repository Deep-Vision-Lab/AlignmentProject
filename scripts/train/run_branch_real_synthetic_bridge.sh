#!/usr/bin/env bash
# Train the ViT visual backend on the OFFLINE Bridge V3 corpus.
# Fine-tune mode reconstructs the exact synthetic checkpoint architecture.
# Scratch mode uses random ViT initialization with the fixed63 geometry.
set -euo pipefail
set -a

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
TRAIN_FROM_SCRATCH="${TRAIN_FROM_SCRATCH:-0}"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-bridge-v3-train}"

case "${TRAIN_FROM_SCRATCH}" in
  0|1) ;;
  *) echo "ERROR: TRAIN_FROM_SCRATCH must be 0 or 1, got ${TRAIN_FROM_SCRATCH}." >&2; exit 2 ;;
esac

[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: Bridge V3 dataset missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
[[ "${MODEL_BACKEND}" == "vit" ]] || {
  echo "ERROR: this launcher expects the ViT branch backend, got ${MODEL_BACKEND}." >&2
  exit 2
}

if [[ "${TRAIN_FROM_SCRATCH}" == "1" ]]; then
  [[ -z "${PRETRAINED_WEIGHTS}" ]] || {
    echo "ERROR: TRAIN_FROM_SCRATCH=1 must not be combined with PRETRAINED_WEIGHTS." >&2
    exit 2
  }

  # Match the synthetic-good fixed63 ViT architecture, but initialize all visual
  # parameters randomly. This keeps the sequence geometry controlled while asking
  # whether Bridge V3 alone is sufficient to learn the visual representation.
  export WINDOW_SIZE="${WINDOW_SIZE:-32}"
  export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
  export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
  export VECTOR_SIZE="${VECTOR_SIZE:-128}"
  export USE_BILSTM=0
  export USE_LOCAL_WINDOW_GROUPING=0
  export VIT_INPUT_HEIGHT="${VIT_INPUT_HEIGHT:-128}"
  export VIT_LAYERS="${VIT_LAYERS:-4}"
  export VIT_HEADS="${VIT_HEADS:-4}"
  export VIT_MLP_DIM="${VIT_MLP_DIM:-512}"
  export VIT_DROPOUT="${VIT_DROPOUT:-0.10}"
  export VIT_MAX_TOKENS="${VIT_MAX_TOKENS:-256}"
  export VIT_POSITION_BASE_TOKENS="${VIT_POSITION_BASE_TOKENS:-63}"

  read -r _scratch_stride _scratch_windows < <(
    python - "${WINDOW_SIZE}" "${STRIDE_RATIO}" <<'PY'
import sys
window = int(sys.argv[1])
ratio = float(sys.argv[2])
stride = max(1, int(window * ratio))
windows = ((1024 - window) // stride) + 1
print(stride, windows)
PY
  )
  if [[ "${WINDOW_SIZE}" != "32" || "${_scratch_stride}" != "16" || "${_scratch_windows}" != "63" ]]; then
    echo "ERROR: scratch Bridge V3 ViT is pinned to window=32 stride=16 fixed63; got window=${WINDOW_SIZE} stride=${_scratch_stride} windows=${_scratch_windows}." >&2
    exit 2
  fi
  echo "visual_initialization random_vit backend=vit window=${WINDOW_SIZE} stride=${_scratch_stride} vector=${VECTOR_SIZE} layers=${VIT_LAYERS} heads=${VIT_HEADS} mlp=${VIT_MLP_DIM}"
else
  [[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" ]] || {
    echo "ERROR: set PRETRAINED_WEIGHTS to the synthetic ViT checkpoint, or set TRAIN_FROM_SCRATCH=1." >&2
    exit 2
  }

  # The synthetic checkpoint is authoritative in fine-tune mode. Rebuild the same
  # token geometry and Transformer dimensions instead of relying on branch defaults.
  eval "$(python - "${PRETRAINED_WEIGHTS}" "${MODEL_BACKEND}" <<'PY'
from __future__ import annotations
import shlex
import sys
import torch

path, expected_backend = sys.argv[1], sys.argv[2]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")

config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
state = {}
if isinstance(payload, dict):
    state = payload.get("image_model_state_dict") or payload.get("model_state_dict") or {}

def strip(key):
    key = str(key)
    while key.startswith("module."):
        key = key[len("module."):]
    return key

keys = [strip(key) for key in state]
backend = str(config.get("model_backend", config.get("visual_encoder_type", ""))).strip().lower()
if not backend:
    if any(key.startswith("vit_encoder.") for key in keys):
        backend = "vit"
    elif any(key.startswith("dinov3_encoder.") for key in keys):
        backend = "dinov3_convnext"
    else:
        backend = "cnn_bilstm"
if backend in {"cnn", "cnn_only"}:
    backend = "cnn_bilstm"
if backend != expected_backend:
    raise SystemExit(
        f"checkpoint/backend mismatch: checkpoint={backend} branch={expected_backend}"
    )
if backend != "vit":
    raise SystemExit(f"Expected ViT checkpoint, got backend={backend}.")

window = int(config.get("window_size", 32))
stride = int(config.get("stride", round(window * float(config.get("stride_ratio", 0.5)))))
vector = int(config.get("vector_size", 128))
if window != 32 or stride != 16:
    raise SystemExit(
        "Bridge V3 fine-tuning is pinned to the synthetic-good fixed63 geometry: "
        f"checkpoint has window={window}, stride={stride}; expected 32/16."
    )
if (1024 - window) // stride + 1 != 63:
    raise SystemExit(f"checkpoint geometry does not produce 63 windows: {window=}, {stride=}")

values = {
    "WINDOW_SIZE": window,
    "STRIDE_RATIO": stride / window,
    "WINDOW_OVERLAP_MODE": "custom",
    "VECTOR_SIZE": vector,
    "USE_BILSTM": 0,
    "USE_LOCAL_WINDOW_GROUPING": 0,
    "VIT_INPUT_HEIGHT": int(config.get("vit_input_height", 128)),
    "VIT_LAYERS": int(config.get("vit_layers", 4)),
    "VIT_HEADS": int(config.get("vit_heads", 4)),
    "VIT_MLP_DIM": int(config.get("vit_mlp_dim", 512)),
    "VIT_DROPOUT": float(config.get("vit_dropout", 0.10)),
    "VIT_MAX_TOKENS": int(config.get("vit_max_tokens", 256)),
    "VIT_POSITION_BASE_TOKENS": int(config.get("vit_position_base_tokens", 63)),
}
for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
print(
    "echo " + shlex.quote(
        "checkpoint_visual_config "
        f"backend=vit window={window} stride={stride} vector={vector} "
        f"layers={values['VIT_LAYERS']} heads={values['VIT_HEADS']} "
        f"mlp={values['VIT_MLP_DIM']} pos_base={values['VIT_POSITION_BASE_TOKENS']}"
    )
)
PY
  )"
fi

export DATA_DIR PRETRAINED_WEIGHTS JOB_ID TRAIN_FROM_SCRATCH
export REAL_MANIFEST_NAME=dataset_manifest.jsonl
export REAL_USE_EXTRA_NO_SHARED=1
export REAL_UNIQUE_LINE_ADAPTATION=0
export REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
export NO_SHARED_IMAGE_OBJECTIVE=synthetic_bridge
export REAL_AUGMENT=0
export AUGMENT=0
export IMAGE_TEXT_LOSS_ON_BOTH_LINES=1
export REAL_TRAIN_SAMPLES_PER_EPOCH=0
export BRIDGE_TRAIN_SAMPLES_PER_EPOCH="${BRIDGE_TRAIN_SAMPLES_PER_EPOCH:-6000}"

if [[ "${TRAIN_FROM_SCRATCH}" == "1" ]]; then
  export EPOCHS="${EPOCHS:-40}"
  export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
else
  export EPOCHS="${EPOCHS:-20}"
  export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
fi
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"

# Keep the corrected visible-span recipe used by the synthetic recovery.
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
export REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-${MAX_TEXT_SPAN_CHARS}}"
export SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
export SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${MAX_TEXT_SPAN_CHARS}}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"

export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}"
export USE_IMAGE_PAIR_CONTRASTIVE=1
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.10}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0
# Bridge positives contain unrelated context. Do not reward whole-line alignment;
# the bridge runtime supplies the shared-island-aware ranking objective instead.
export USE_SEQUENCE_ALIGNMENT_RANKING="${USE_SEQUENCE_ALIGNMENT_RANKING:-0}"
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.0}"
export SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.50}"
export SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}"

NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
DEFAULT_ACCUM=4
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} must be divisible by GPUs*accum=${DENOM}." >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOM))
export NUM_GPUS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ "${TRAIN_FROM_SCRATCH}" == "1" ]]; then
    INIT_DESC="random ViT initialization"
  else
    INIT_DESC="checkpoint=${PRETRAINED_WEIGHTS}"
  fi
  cat <<EOF
=== SUBMIT VIT BRIDGE V3 TRAINING ===
backend=${MODEL_BACKEND}
initialization=${INIT_DESC}
job=${JOB_ID}
data=${DATA_DIR}
geometry=window${WINDOW_SIZE}/stride16/fixed63
epochs_max=${EPOCHS} lr=${LEARNING_RATE} samples_per_epoch=${BRIDGE_TRAIN_SAMPLES_PER_EPOCH}
vit=${VIT_LAYERS}layers/${VIT_HEADS}heads/mlp${VIT_MLP_DIM}
local_grouping=OFF bilstm=OFF
negatives=${NUM_NEGATIVES} active_negatives=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}
whole_line_sequence_ranking=${USE_SEQUENCE_ALIGNMENT_RANKING}
shared_island_bridge_ranking=ON
best_checkpoint=Weights/${JOB_ID}/checkpoint_best_val.pth
EOF
  sbatch \
    --partition="${PARTITION}" \
    --job-name="${JOB_ID}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_ASYNC_ERROR_HANDLING=1

BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"
export TRAIN_EXPECTED_BRANCH="${BRANCH}" TRAIN_EXPECTED_COMMIT="${COMMIT}" TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"

ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type real
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --window_size "${WINDOW_SIZE}"
  --stride_ratio "${STRIDE_RATIO}"
  --window_overlap_mode "${WINDOW_OVERLAP_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  --no-augment
)
if [[ "${TRAIN_FROM_SCRATCH}" == "0" ]]; then
  ARGS+=(--pretrained_weights "${PRETRAINED_WEIGHTS}")
fi

if (( NUM_GPUS > 1 )); then
  exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${ARGS[@]}"
fi
exec python "${ARGS[@]}"
