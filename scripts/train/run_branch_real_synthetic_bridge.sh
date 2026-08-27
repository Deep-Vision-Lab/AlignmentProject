#!/usr/bin/env bash
# Fine-tune the DINOv3 ConvNeXt visual backend on the OFFLINE Bridge V3 corpus.
# The synthetic checkpoint is authoritative: reconstruct its window geometry and
# optional sequence stack before loading weights so real adaptation cannot drift.
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
  echo "ERROR: PRETRAINED_WEIGHTS must point to the synthetic DINOv3 checkpoint." >&2
  exit 2
}
: "${DINOV3_REPO_DIR:?Set DINOV3_REPO_DIR to the local official Meta DINOv3 repository.}"
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
  echo "ERROR: this launcher expects the dinov3_convnext branch backend, got ${MODEL_BACKEND}" >&2
  exit 2
}

# Restore the exact synthetic checkpoint architecture. In particular, do not
# silently re-enable BiLSTM/local grouping if the synthetic DINO run disabled it.
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

keys = [strip(key) for key in state]
backend = str(config.get("model_backend", config.get("visual_encoder_type", ""))).strip().lower()
if not backend and any(key.startswith("dinov3_encoder.") for key in keys):
    backend = "dinov3_convnext"
if backend != "dinov3_convnext":
    raise SystemExit(f"checkpoint/backend mismatch: expected dinov3_convnext, got {backend or '<missing>'}")
if not any(key.startswith("dinov3_encoder.") for key in keys):
    raise SystemExit("checkpoint does not contain DINOv3 encoder weights")

window = int(config.get("window_size", 32))
stride = int(config.get("stride", round(window * float(config.get("stride_ratio", 0.5)))))
vector = int(config.get("vector_size", 128))
if window != 32 or stride != 16 or ((1024 - window) // stride + 1) != 63:
    raise SystemExit(
        "Bridge V3 fine-tuning is pinned to the synthetic fixed63 geometry; "
        f"checkpoint has window={window}, stride={stride}."
    )

use_bilstm = bool(config.get("use_bilstm", True))
use_grouping = bool(config.get("use_local_window_grouping", True))
values = {
    "WINDOW_SIZE": window,
    "STRIDE_RATIO": stride / window,
    "WINDOW_OVERLAP_MODE": "custom",
    "VECTOR_SIZE": vector,
    "USE_BILSTM": int(use_bilstm),
    "USE_LOCAL_WINDOW_GROUPING": int(use_grouping),
    "BILSTM_LAYERS": int(config.get("bilstm_layers", 2)),
    "BILSTM_HIDDEN_DIM": int(config.get("bilstm_hidden_dim") or vector),
    "LOCAL_GROUP_SIZE": int(config.get("local_group_size", config.get("local_window_group_size", 3))),
    "DINOV3_FREEZE_BACKBONE": int(bool(config.get("dinov3_freeze_backbone", True))),
    # Build the official architecture locally; the project checkpoint then loads
    # the complete trained DINO state. This avoids requiring a second weights file.
    "DINOV3_ALLOW_RANDOM_INIT": 1,
}
for name, value in values.items():
    print(f"export {name}={shlex.quote(str(value))}")
print("echo " + shlex.quote(
    "checkpoint_visual_config "
    f"backend=dinov3_convnext window={window} stride={stride} vector={vector} "
    f"bilstm={int(use_bilstm)} grouping={int(use_grouping)} "
    f"freeze_backbone={values['DINOV3_FREEZE_BACKBONE']}"
))
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

export EPOCHS="${EPOCHS:-20}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"

# Keep the corrected visible-span recipe used by the fixed63 synthetic recovery.
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
# Bridge V3 positives have distractor context; do not force whole-line alignment.
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
=== SUBMIT DINOV3 CONVNEXT BRIDGE V3 FINE-TUNE ===
backend=${MODEL_BACKEND}
checkpoint=${PRETRAINED_WEIGHTS}
job=${JOB_ID}
geometry=window${WINDOW_SIZE}/stride$((WINDOW_SIZE * 1 / 2))/fixed63
epochs_max=${EPOCHS} lr=${LEARNING_RATE} samples_per_epoch=${BRIDGE_TRAIN_SAMPLES_PER_EPOCH}
bilstm=${USE_BILSTM} local_grouping=${USE_LOCAL_WINDOW_GROUPING} freeze_dino=${DINOV3_FREEZE_BACKBONE}
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
