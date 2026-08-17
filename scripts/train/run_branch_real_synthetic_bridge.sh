#!/usr/bin/env bash
# Architecture-neutral fine-tuning on the OFFLINE real-conditioned synthetic bridge.
# The active model_backend.py selects CNN / ViT / DINOv3; the checkpoint must match.
set -euo pipefail
set -a

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-real-synthetic-bridge-v2}"

[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: offline bridge dataset missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}
[[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: PRETRAINED_WEIGHTS must point to a checkpoint from this architecture." >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

read -r CKPT_BACKEND CKPT_BACKBONE CKPT_BILSTM < <(
  python - "${PRETRAINED_WEIGHTS}" <<'PY'
import sys, torch
path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")
config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
backend = str(config.get("model_backend", config.get("visual_encoder_type", ""))).strip().lower()
if not backend:
    backend = "cnn_bilstm"
if backend in {"cnn", "cnn_only"}:
    backend = "cnn_bilstm"
backbone = "-"
if backend == "cnn_bilstm":
    backbone = str(config.get("cnn_backbone", "resnet34")).strip().lower().replace("-", "")
use_bilstm = bool(config.get("use_bilstm", backend != "vit"))
print(backend, backbone, "1" if use_bilstm else "0")
PY
)

if [[ "${CKPT_BACKEND}" != "${MODEL_BACKEND}" ]]; then
  echo "ERROR: checkpoint/backend mismatch: checkpoint=${CKPT_BACKEND} branch=${MODEL_BACKEND}" >&2
  exit 2
fi

case "${MODEL_BACKEND}" in
  cnn_bilstm)
    export CNN_BACKBONE="${CNN_BACKBONE:-${CKPT_BACKBONE}}"
    [[ "${CNN_BACKBONE}" == "${CKPT_BACKBONE}" ]] || {
      echo "ERROR: CNN_BACKBONE=${CNN_BACKBONE} but checkpoint uses ${CKPT_BACKBONE}." >&2
      exit 2
    }
    export USE_BILSTM="${USE_BILSTM:-${CKPT_BILSTM}}"
    DEFAULT_ACCUM=1
    ;;
  vit)
    export USE_BILSTM=0
    export USE_LOCAL_WINDOW_GROUPING=0
    DEFAULT_ACCUM=4
    ;;
  dinov3_convnext)
    export USE_BILSTM="${USE_BILSTM:-${CKPT_BILSTM}}"
    export DINOV3_ALLOW_RANDOM_INIT="${DINOV3_ALLOW_RANDOM_INIT:-1}"
    : "${DINOV3_REPO_DIR:?Set DINOV3_REPO_DIR to the local official DINOv3 repository.}"
    DEFAULT_ACCUM=4
    ;;
  *)
    echo "ERROR: unsupported model backend ${MODEL_BACKEND}" >&2
    exit 2
    ;;
esac

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
export BRIDGE_TRAIN_SAMPLES_PER_EPOCH="${BRIDGE_TRAIN_SAMPLES_PER_EPOCH:-0}"

# Bridge V2 is now the direct real-domain adaptation stage. Give it a longer
# maximum run, validate every epoch, and let checkpoint_best_val.pth select the
# checkpoint used by downstream evaluation rather than assuming the last epoch wins.
export EPOCHS="${EPOCHS:-15}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"

export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}"
export REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-${MAX_TEXT_SPAN_CHARS}}"
export MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"

export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}"
export USE_IMAGE_PAIR_CONTRASTIVE=1
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.10}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0
# IMPORTANT for Bridge V2: respect the public wrapper/default instead of forcing
# generic whole-line sequence ranking back on. V2 positives contain distractors;
# shared-island-aware bridge ranking is installed separately by the bridge runtime.
export USE_SEQUENCE_ALIGNMENT_RANKING="${USE_SEQUENCE_ALIGNMENT_RANKING:-0}"
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.0}"
export SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.50}"
export SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}"
export SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR="${SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR:-0.08}"
export SEQUENCE_RANKING_SCORE_MARGIN="${SEQUENCE_RANKING_SCORE_MARGIN:-0.10}"

NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE must divide GPUs*accum=${DENOM}." >&2
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
=== SUBMIT REAL-SYNTHETIC BRIDGE V2 ===
backend=${MODEL_BACKEND}
checkpoint=${PRETRAINED_WEIGHTS}
job=${JOB_ID}
epochs_max=${EPOCHS} lr=${LEARNING_RATE}
positive_negative_balance=50/50
whole_line_sequence_ranking=${USE_SEQUENCE_ALIGNMENT_RANKING}
shared_island_bridge_ranking=ON
online_rendering=NO
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
