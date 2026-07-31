#!/usr/bin/env bash
# The only public real-data training command.
# Run from the login node with:
#   JOB_ID=<name> PRETRAINED_WEIGHTS=<checkpoint> bash scripts/train/run_real_finetune.sh
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: configure with environment variables, then run:" >&2
  echo "  JOB_ID=<name> PRETRAINED_WEIGHTS=<path> bash scripts/train/run_real_finetune.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_DIR}"
mkdir -p out logs

: "${JOB_ID:?Set JOB_ID to the output weights-folder name.}"
: "${PRETRAINED_WEIGHTS:?Set PRETRAINED_WEIGHTS to model_latest.pth.}"
[[ -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: pretrained checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 2
}

AUGMENT="${AUGMENT:-1}"
case "${AUGMENT}" in
  0|1) ;;
  *) echo "ERROR: AUGMENT must be 0 or 1, got ${AUGMENT}." >&2; exit 2 ;;
esac

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"

# ---------------------------------------------------------------------------
# Canonical experiment settings
# ---------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-64}}"
case "${MODEL_BACKEND}" in
  cnn_bilstm) DEFAULT_ACCUMULATION_STEPS=1 ;;
  vit) DEFAULT_ACCUMULATION_STEPS=4 ;;
  *) echo "ERROR: unsupported model backend '${MODEL_BACKEND}'." >&2; exit 2 ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION_STEPS}}"

for value_name in NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${value_name} must be a positive integer, got ${value}." >&2
    exit 2
  }
done
BATCH_DENOMINATOR=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
if (( EFFECTIVE_GLOBAL_BATCH_SIZE % BATCH_DENOMINATOR != 0 )); then
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} must be divisible by ${BATCH_DENOMINATOR}." >&2
  exit 2
fi
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / BATCH_DENOMINATOR))
MICRO_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

LINE_HEIGHT="${LINE_HEIGHT:-128}"
LINE_WIDTH="${LINE_WIDTH:-1024}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.25}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
read -r STRIDE_PIXELS COMPUTED_ALIGNMENT_WINDOWS < <(
  python - "${LINE_WIDTH}" "${WINDOW_SIZE}" "${STRIDE_RATIO}" "${WINDOW_OVERLAP_MODE}" <<'PY'
import sys
line_width = int(sys.argv[1])
window_size = int(sys.argv[2])
ratio = float(sys.argv[3])
mode = sys.argv[4].strip().lower()
if line_width <= 0 or window_size <= 0 or window_size > line_width:
    raise SystemExit("Invalid line/window geometry")
if mode == "no_overlap":
    stride = window_size
elif mode == "light_overlap":
    stride = max(1, window_size // 2)
elif mode == "dense_overlap":
    stride = max(1, window_size // 4)
elif mode == "custom":
    stride = max(1, int(window_size * ratio))
else:
    raise SystemExit(f"Unknown WINDOW_OVERLAP_MODE={mode!r}")
print(stride, ((line_width - window_size) // stride) + 1)
PY
)

REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-2}"
[[ "${REAL_MAX_TEXT_SPAN_CHARS}" =~ ^[12]$ ]] || {
  echo "ERROR: REAL_MAX_TEXT_SPAN_CHARS must be 1 or 2." >&2
  exit 2
}
REAL_FILTER_INFEASIBLE_SPAN_DTW="${REAL_FILTER_INFEASIBLE_SPAN_DTW:-1}"
REAL_MAX_ALIGNMENT_WINDOWS="${REAL_MAX_ALIGNMENT_WINDOWS:-${COMPUTED_ALIGNMENT_WINDOWS}}"
[[ "${REAL_MAX_ALIGNMENT_WINDOWS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: REAL_MAX_ALIGNMENT_WINDOWS must be positive." >&2
  exit 2
}
if (( REAL_MAX_ALIGNMENT_WINDOWS > COMPUTED_ALIGNMENT_WINDOWS )); then
  echo "ERROR: feasibility cap ${REAL_MAX_ALIGNMENT_WINDOWS} exceeds actual windows ${COMPUTED_ALIGNMENT_WINDOWS}." >&2
  exit 2
fi

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
REAL_MANIFEST_NAME="${REAL_MANIFEST_NAME:-dataset_manifest.jsonl}"
MANIFEST_PATH="${DATA_DIR}/${REAL_MANIFEST_NAME}"
[[ -f "${MANIFEST_PATH}" ]] || {
  echo "ERROR: real dataset manifest not found: ${MANIFEST_PATH}" >&2
  exit 2
}

# Verify checkpoint/backend compatibility before requesting GPUs.
python - "${PRETRAINED_WEIGHTS}" "${WINDOW_SIZE}" <<'PY'
import sys
import torch
import model_backend
path, requested_window = sys.argv[1], int(sys.argv[2])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
actual = str(config.get("model_backend", config.get("visual_encoder_type", ""))).lower()
expected = str(model_backend.MODEL_NAME).lower()
checkpoint_window = config.get("window_size")
print(f"branch_backend={expected} checkpoint_backend={actual or '<missing>'} checkpoint_window={checkpoint_window or '<missing>'} requested_window={requested_window}")
if actual and actual != expected:
    raise SystemExit(f"Checkpoint/backend mismatch: branch={expected!r}, checkpoint={actual!r}")
if checkpoint_window is not None and int(checkpoint_window) != requested_window:
    raise SystemExit(f"Checkpoint/window mismatch: expected WINDOW_SIZE={checkpoint_window}")
PY

# ---------------------------------------------------------------------------
# Complete training configuration exported to Slurm and torchrun ranks
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

DATASET_TYPE=real
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
REAL_AUGMENT="${AUGMENT}"
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-6000}"
REAL_AUG_STITCH_PROB="${REAL_AUG_STITCH_PROB:-0}"
REAL_AUG_STITCH_POOL_SIZE="${REAL_AUG_STITCH_POOL_SIZE:-32}"
REAL_AUG_STITCH_MAX_TEXT_CHARS="${REAL_AUG_STITCH_MAX_TEXT_CHARS:-120}"
REAL_AUG_STITCH_PREFER_ADJACENT="${REAL_AUG_STITCH_PREFER_ADJACENT:-1}"
REAL_AUG_STITCH_GAP_MIN="${REAL_AUG_STITCH_GAP_MIN:-0.08}"
REAL_AUG_STITCH_GAP_MAX="${REAL_AUG_STITCH_GAP_MAX:-0.18}"
REAL_AUG_APPEARANCE_PROB="${REAL_AUG_APPEARANCE_PROB:-0.85}"
REAL_AUG_ROTATE_DEG="${REAL_AUG_ROTATE_DEG:-1.25}"
REAL_AUG_TRANSLATE_X="${REAL_AUG_TRANSLATE_X:-0.012}"
REAL_AUG_TRANSLATE_Y="${REAL_AUG_TRANSLATE_Y:-0.035}"
REAL_AUG_BRIGHTNESS="${REAL_AUG_BRIGHTNESS:-0.12}"
REAL_AUG_CONTRAST="${REAL_AUG_CONTRAST:-0.18}"
REAL_AUG_BLUR_PROB="${REAL_AUG_BLUR_PROB:-0.15}"
REAL_AUG_BLUR_RADIUS="${REAL_AUG_BLUR_RADIUS:-0.8}"
REAL_AUG_NOISE_PROB="${REAL_AUG_NOISE_PROB:-0.18}"
REAL_AUG_NOISE_STD="${REAL_AUG_NOISE_STD:-5.0}"
REAL_AUG_MORPH_PROB="${REAL_AUG_MORPH_PROB:-0.25}"
REAL_AUG_SPECKLE_PROB="${REAL_AUG_SPECKLE_PROB:-0.12}"
REAL_AUG_SPECKLE_FRACTION="${REAL_AUG_SPECKLE_FRACTION:-0.0006}"
REAL_DATASET_LABELS="${REAL_DATASET_LABELS:-high_match,medium_match}"
REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
REAL_SPLIT_BY_PAIR_ID="${REAL_SPLIT_BY_PAIR_ID:-1}"
REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"
REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"

LANGUAGE="${LANGUAGE:-Arabic}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-arabic_span}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
MAX_TEXT_TOKEN_CHARS="${MAX_TEXT_TOKEN_CHARS:-2}"
MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS}"
SPAN_MAX_CORE_CHARS_CAP="${REAL_MAX_TEXT_SPAN_CHARS}"
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
SPAN_INCLUDE_SPACE_CONTEXT=0
SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
ALLOW_UNSAFE_SPAN_CONFIG=0
SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
SPAN_DTW_BUCKET_TEXT_LENGTHS="${SPAN_DTW_BUCKET_TEXT_LENGTHS:-1}"
SPAN_DTW_TEXT_BUCKET_SIZE="${SPAN_DTW_TEXT_BUCKET_SIZE:-64}"
SPAN_DTW_MAX_TEXT_BUCKET="${SPAN_DTW_MAX_TEXT_BUCKET:-256}"
SPAN_DTW_BATCH_BUCKET_SIZE="${SPAN_DTW_BATCH_BUCKET_SIZE:-32}"
SPAN_DTW_BATCH_BUCKET_MODE="${SPAN_DTW_BATCH_BUCKET_MODE:-power2}"
SPAN_FEATURE_CACHE_SIZE="${SPAN_FEATURE_CACHE_SIZE:-2048}"
SPAN_FEATURE_CACHE_DTYPE="${SPAN_FEATURE_CACHE_DTYPE:-float16}"
CLEAR_SPAN_CACHE_EACH_EPOCH="${CLEAR_SPAN_CACHE_EACH_EPOCH:-1}"

NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
SPAN_NEGATIVE_GRAD_MODE="${SPAN_NEGATIVE_GRAD_MODE:-hardest}"
NEGATIVE_MODE="${NEGATIVE_MODE:-mixed}"
USE_LOCAL_HARD_NEGATIVES="${USE_LOCAL_HARD_NEGATIVES:-1}"
USE_IMAGE_PAIR_CONTRASTIVE="${USE_IMAGE_PAIR_CONTRASTIVE:-1}"
IMAGE_TEXT_LOSS_ON_BOTH_LINES="${IMAGE_TEXT_LOSS_ON_BOTH_LINES:-1}"

EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
USE_AMP="${USE_AMP:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alignment-real-finetuning}"

TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"

JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PROJECT_DIR}/.jax_cache/span_dtw}"
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-7200}"
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTHONUNBUFFERED=1
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
set +a
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

resolve_hf_home() {
  local slug="models--${ARABIC_TEXT_MODEL_NAME//\//--}"
  local candidate layout snapshots snapshot
  local candidates=("${HF_HOME:-}" "${PROJECT_DIR}/.hf_cache" "${PROJECT_DIR}_clone/.hf_cache" "${HOME}/.cache/huggingface")
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" && -d "${candidate}" ]] || continue
    for layout in "${candidate}" "${candidate}/hub"; do
      snapshots="${layout}/${slug}/snapshots"
      [[ -d "${snapshots}" ]] || continue
      while IFS= read -r -d '' snapshot; do
        [[ -f "${snapshot}/config.json" ]] || continue
        if compgen -G "${snapshot}/model*.safetensors" >/dev/null || compgen -G "${snapshot}/pytorch_model*.bin" >/dev/null; then
          printf '%s\n' "${candidate}"
          return 0
        fi
      done < <(find "${snapshots}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
  done
  return 1
}
HF_HOME="$(resolve_hf_home)" || {
  echo "ERROR: local cache for ${ARABIC_TEXT_MODEL_NAME} was not found." >&2
  exit 1
}
export HF_HOME
unset TRANSFORMERS_CACHE

print_config() {
  printf '%s\n' \
    "Canonical real fine-tuning" \
    "  branch                 = $(git branch --show-current)" \
    "  model backend          = ${MODEL_BACKEND}" \
    "  job id                 = ${JOB_ID}" \
    "  checkpoint             = ${PRETRAINED_WEIGHTS}" \
    "  GPUs                   = ${GPU_RESOURCE}:${NUM_GPUS}" \
    "  per-GPU micro-batch    = ${BATCH_SIZE}" \
    "  accumulation steps     = ${GRADIENT_ACCUMULATION_STEPS}" \
    "  effective global batch = ${EFFECTIVE_GLOBAL_BATCH_SIZE}" \
    "  epochs                 = ${EPOCHS}" \
    "  train samples/epoch    = ${REAL_TRAIN_SAMPLES_PER_EPOCH}" \
    "  window/stride          = ${WINDOW_SIZE}/${STRIDE_PIXELS}" \
    "  visual windows         = ${COMPUTED_ALIGNMENT_WINDOWS}" \
    "  JAX text bucket        = ${SPAN_DTW_TEXT_BUCKET_SIZE}" \
    "  JAX batch bucket       = ${SPAN_DTW_BATCH_BUCKET_MODE}" \
    "  DDP timeout seconds    = ${DIST_TIMEOUT_SECONDS}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${SLURM_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
    --tasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python -c "import torch, transformers, jax; print(f'torch={torch.__version__} transformers={transformers.__version__} jax={jax.__version__}')"

AUGMENT_ARG=--no-augment
[[ "${REAL_AUGMENT}" == "1" ]] && AUGMENT_ARG=--augment
TRAIN_ARGS=(
  scripts/train/train_model.py
  --job_id "${JOB_ID}"
  --dataset_type real
  --data_dir "${DATA_DIR}"
  "${AUGMENT_ARG}"
  --train_samples_per_epoch "${REAL_TRAIN_SAMPLES_PER_EPOCH}"
  --num_samples "${NUM_SAMPLES}"
  --epochs "${EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --negative_mode "${NEGATIVE_MODE}"
  --num_negatives "${NUM_NEGATIVES}"
  --pretrained_weights "${PRETRAINED_WEIGHTS}"
)

print_config
nvidia-smi -L || true
RANK_WRAPPER="${PROJECT_DIR}/scripts/train/run_rank_isolated.sh"
[[ -f "${RANK_WRAPPER}" ]] || { echo "ERROR: missing ${RANK_WRAPPER}" >&2; exit 1; }

if (( NUM_GPUS > 1 )); then
  exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --max_restarts=0 \
    --no_python \
    bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"
fi
exec bash "${RANK_WRAPPER}" "${TRAIN_ARGS[@]}"
