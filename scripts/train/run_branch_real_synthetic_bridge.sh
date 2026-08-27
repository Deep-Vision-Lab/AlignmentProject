#!/usr/bin/env bash
# Fine-tune a synthetic DINOv3 ConvNeXt checkpoint on OFFLINE Bridge V3 data.
# The checkpoint is authoritative: this launcher reconstructs its fixed63 geometry,
# sequence mode (legacy BiLSTM / Transformer / none), grouping state, and attention
# dimensions before strict state-dict loading.
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
  echo "ERROR: PRETRAINED_WEIGHTS must point to a synthetic DINO checkpoint." >&2
  exit 2
}
: "${DINOV3_REPO_DIR:?Set DINOV3_REPO_DIR to the local official DINOv3 repository.}"
[[ -f "${DINOV3_REPO_DIR}/hubconf.py" ]] || {
  echo "ERROR: DINOV3_REPO_DIR has no hubconf.py: ${DINOV3_REPO_DIR}" >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
[[ "${MODEL_BACKEND}" == "dinov3_convnext" ]] || {
  echo "ERROR: this launcher expects the DINOv3 ConvNeXt branch; got ${MODEL_BACKEND}" >&2
  exit 2
}

# Reconstruct every structural choice from the synthetic checkpoint. Old DINO+
# BiLSTM checkpoints remain loadable; new DINO+Transformer checkpoints are detected
# from metadata or state-dict keys and cannot be accidentally loaded as BiLSTM.
eval "$(python - "${PRETRAINED_WEIGHTS}" <<'PY'
from __future__ import annotations
import shlex
import sys
import torch

path = sys.argv[1]
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

state = {strip(k): v for k, v in state.items()}
keys = tuple(state)
backend = str(config.get("model_backend", config.get("visual_encoder_type", ""))).strip().lower()
if not backend:
    backend = "dinov3_convnext" if any(k.startswith("dinov3_encoder.") for k in keys) else "unknown"
if backend != "dinov3_convnext":
    raise SystemExit(f"checkpoint is not DINOv3 ConvNeXt: backend={backend}")

window = int(config.get("window_size", 32))
stride = int(config.get("stride", round(window * float(config.get("stride_ratio", 0.5)))))
vector = int(config.get("vector_size", 128))
if window != 32 or stride != 16 or (1024 - window) // stride + 1 != 63:
    raise SystemExit(
        "Bridge V3 fine-tuning is pinned to the synthetic-good fixed63 geometry; "
        f"checkpoint has window={window}, stride={stride}."
    )

mode = str(config.get("dinov3_sequence_mode", "")).strip().lower().replace("-", "_")
if not mode:
    if any(k.startswith("sequence_encoder.bilstm.") for k in keys):
        mode = "bilstm"
    elif any(k.startswith("sequence_encoder.position_embedding") or k.startswith("sequence_encoder.encoder.") for k in keys):
        mode = "transformer"
    else:
        mode = "none"
if mode not in {"bilstm", "transformer", "none"}:
    raise SystemExit(f"unsupported checkpoint DINO sequence mode: {mode}")

if "use_local_window_grouping" in config:
    grouping = bool(config["use_local_window_grouping"])
elif "_use_local_grouping_state" in state:
    grouping = bool(int(state["_use_local_grouping_state"].item()))
else:
    grouping = mode == "bilstm"
if mode == "transformer" and grouping:
    raise SystemExit(
        "Invalid DINO transformer checkpoint: global Transformer mode must not use "
        "three-window LocalWindowGrouping."
    )

values = {
    "WINDOW_SIZE": window,
    "STRIDE_RATIO": stride / window,
    "WINDOW_OVERLAP_MODE": "custom",
    "VECTOR_SIZE": vector,
    "DINOV3_SEQUENCE_MODE": mode,
    "USE_BILSTM": 1 if mode == "bilstm" else 0,
    "USE_LOCAL_WINDOW_GROUPING": 1 if grouping else 0,
    "BILSTM_LAYERS": int(config.get("bilstm_layers", 2)),
    "BILSTM_HIDDEN_DIM": int(config.get("bilstm_hidden_dim") or vector),
    "LOCAL_GROUP_SIZE": int(config.get("local_group_size", config.get("local_window_group_size", 3))),
    "DINOV3_FREEZE_BACKBONE": 1 if bool(config.get("dinov3_freeze_backbone", True)) else 0,
    "DINOV3_WINDOW_CHUNK_SIZE": int(config.get("dinov3_window_chunk_size", 256)),
    "DINOV3_TRANSFORMER_LAYERS": int(config.get("dinov3_transformer_layers", 4)),
    "DINOV3_TRANSFORMER_HEADS": int(config.get("dinov3_transformer_heads", 4)),
    "DINOV3_TRANSFORMER_MLP_DIM": int(config.get("dinov3_transformer_mlp_dim", 512)),
    "DINOV3_TRANSFORMER_DROPOUT": float(config.get("dinov3_transformer_dropout", 0.10)),
    "DINOV3_TRANSFORMER_MAX_TOKENS": int(config.get("dinov3_transformer_max_tokens", 256)),
    "DINOV3_TRANSFORMER_POSITION_BASE_TOKENS": int(
        config.get("dinov3_transformer_position_base_tokens", 63)
    ),
}
for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
print(
    "echo " + shlex.quote(
        "checkpoint_visual_config "
        f"backend=dinov3_convnext window={window} stride={stride} vector={vector} "
        f"sequence_mode={mode} grouping={int(grouping)} "
        f"freeze_backbone={values['DINOV3_FREEZE_BACKBONE']}"
    )
)
PY
)"

# The full project checkpoint contains the DINO backbone state. If DINOV3_WEIGHTS
# is omitted, permit constructor initialization and then immediately overwrite it
# with PRETRAINED_WEIGHTS during strict checkpoint loading.
export DINOV3_ALLOW_RANDOM_INIT="${DINOV3_ALLOW_RANDOM_INIT:-1}"

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

export EPOCHS="${EPOCHS:-20}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"

# Preserve the corrected synthetic supervision semantics during domain adaptation.
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
=== SUBMIT DINOV3 BRIDGE V3 FINE-TUNE ===
backend=${MODEL_BACKEND}
checkpoint=${PRETRAINED_WEIGHTS}
job=${JOB_ID}
geometry=window${WINDOW_SIZE}/fixed63
sequence_mode=${DINOV3_SEQUENCE_MODE} grouping=${USE_LOCAL_WINDOW_GROUPING} bilstm=${USE_BILSTM}
epochs_max=${EPOCHS} lr=${LEARNING_RATE} samples_per_epoch=${BRIDGE_TRAIN_SAMPLES_PER_EPOCH}
positive_negative_balance=50/50
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