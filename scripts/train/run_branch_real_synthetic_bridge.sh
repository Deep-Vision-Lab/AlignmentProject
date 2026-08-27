#!/usr/bin/env bash
# Fine-tune the active visual backend on the OFFLINE Bridge V3 corpus.
# For ViT this launcher reconstructs the exact synthetic checkpoint architecture
# before loading weights so window geometry / RTL direction / Transformer shape
# cannot silently drift during real-domain adaptation.
set -euo pipefail
set -a

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v3}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-bridge-v3-finetune}"

[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: Bridge V3 dataset missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}
[[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: PRETRAINED_WEIGHTS must point to the synthetic checkpoint to fine-tune." >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

# The synthetic checkpoint is authoritative. Rebuild the same token geometry and
# Transformer dimensions instead of relying on branch defaults that may change.
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
    raise SystemExit(
        "This branch is expected to fine-tune the ViT synthetic model; "
        f"got backend={backend}."
    )

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

export DATA_DIR PRETRAINED_WEIGHTS JOB_ID
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

# Real adaptation is deliberately conservative relative to the 1e-4 synthetic run.
# Validate every epoch and preserve checkpoint_best_val.pth.
export EPOCHS="${EPOCHS:-20}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
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
  cat <<EOF
=== SUBMIT VIT BRIDGE V3 FINE-TUNE ===
backend=${MODEL_BACKEND}
checkpoint=${PRETRAINED_WEIGHTS}
job=${JOB_ID}
geometry=window${WINDOW_SIZE}/fixed63
epochs_max=${EPOCHS} lr=${LEARNING_RATE} samples_per_epoch=${BRIDGE_TRAIN_SAMPLES_PER_EPOCH}
vit=${VIT_LAYERS}layers/${VIT_HEADS}heads/mlp${VIT_MLP_DIM}
local_grouping=OFF bilstm=OFF
positive_negative_balance=50/50
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