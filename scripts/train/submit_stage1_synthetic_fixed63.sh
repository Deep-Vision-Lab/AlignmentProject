#!/usr/bin/env bash
# Controlled Stage-1 synthetic->synthetic training for CNN+BiLSTM or ViT.
# The two architecture branches use the same data, geometry, loss weights,
# optimizer schedule, effective batch size, and validation settings. Only the
# visual backend selected by model_backend.py is intentionally different.
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  SHARED_PROJECT_DIR="$(readlink -f "${TRAIN_SHARED_PROJECT_DIR:-${PROJECT_DIR}}")"
else
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  SHARED_PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${SHARED_PROJECT_DIR}"
mkdir -p out logs

DATA_DIR="${DATA_DIR:-${SHARED_PROJECT_DIR}/DataSet/AugmentedArabicDataset63}"
DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: fixed63 synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
case "${MODEL_BACKEND}" in
  cnn_bilstm)
    DEFAULT_JOB_ID="cnn_bilstm_stage1_fixed63_27k"
    DEFAULT_ACCUMULATION_STEPS=1
    CNN_BACKBONE="${CNN_BACKBONE:-resnet34}"
    USE_BILSTM=1
    BILSTM_LAYERS="${BILSTM_LAYERS:-2}"
    BILSTM_HIDDEN_DIM="${BILSTM_HIDDEN_DIM:-128}"
    USE_LOCAL_WINDOW_GROUPING=1
    LOCAL_WINDOW_GROUP_SIZE="${LOCAL_WINDOW_GROUP_SIZE:-3}"
    ;;
  vit)
    DEFAULT_JOB_ID="vit_stage1_fixed63_27k"
    DEFAULT_ACCUMULATION_STEPS=4
    USE_BILSTM=0
    USE_LOCAL_WINDOW_GROUPING=0
    CNN_BACKBONE=""
    BILSTM_LAYERS=0
    BILSTM_HIDDEN_DIM=128
    LOCAL_WINDOW_GROUP_SIZE=3
    ;;
  *)
    echo "ERROR: Stage-1 launcher supports cnn_bilstm or vit, got ${MODEL_BACKEND}." >&2
    exit 2
    ;;
esac

JOB_ID="${JOB_ID:-${DEFAULT_JOB_ID}}"
NUM_SAMPLES="${NUM_SAMPLES:-27000}"
EPOCHS="${EPOCHS:-35}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-32}"
STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
VECTOR_SIZE="${VECTOR_SIZE:-128}"

# Historical successful fixed63 synthetic objective. Keep this identical across
# CNN+BiLSTM and ViT so the architecture is the only intended model difference.
MAX_TEXT_SPAN_CHARS=2
MAX_TEXT_TOKEN_CHARS=2
MAX_WINDOWS_PER_SPAN=3
SPAN_MAX_CORE_CHARS_CAP=2
SPAN_CONNECTED_MAX_UNITS_PER_SPAN=2
SPAN_INCLUDE_SPACE_CONTEXT=0
SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
SPAN_DTW_BACKEND="${SPAN_DTW_BACKEND:-jax}"
CONTRASTIVE_SOFT_DTW_GAMMA="${CONTRASTIVE_SOFT_DTW_GAMMA:-0.1}"
CONTRASTIVE_MARGIN="${CONTRASTIVE_MARGIN:-10.0}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.07}"
NUM_NEGATIVES="${NUM_NEGATIVES:-4}"
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.25}"
LOCAL_HARD_NEGATIVE_MARGIN="${LOCAL_HARD_NEGATIVE_MARGIN:-0.35}"
IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.40}"
SEQUENCE_CONSISTENCY_LOSS_WEIGHT="${SEQUENCE_CONSISTENCY_LOSS_WEIGHT:-0.05}"
IMAGE_VARIANCE_LOSS_WEIGHT="${IMAGE_VARIANCE_LOSS_WEIGHT:-0.01}"
IMAGE_VARIANCE_TARGET_STD="${IMAGE_VARIANCE_TARGET_STD:-0.05}"
IMAGE_TEXT_LOSS_ON_BOTH_LINES=1

# Stage 1 is synthetic only. Explicitly defeat stale real/bridge flags inherited
# from an interactive shell so bridge modules cannot leak into this experiment.
DATASET_TYPE=synthetic
SYNTHETIC_MANUSCRIPT_AUGMENT=0
REAL_AUGMENT=0
AUGMENT=0
REAL_USE_EXTRA_NO_SHARED=0
REAL_UNIQUE_LINE_ADAPTATION=0
REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
USE_SEQUENCE_ALIGNMENT_RANKING=0
SEQUENCE_RANKING_WEIGHT=0.0
BRIDGE_CROSS_TEXT_WEIGHT=0.0

VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-2}"
VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUMULATION_STEPS}}"
DENOM=$((NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
(( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || {
  echo "ERROR: EFFECTIVE_GLOBAL_BATCH_SIZE=${EFFECTIVE_GLOBAL_BATCH_SIZE} must divide GPUs*accum=${DENOM}." >&2
  exit 2
}
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / DENOM))

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

# Resolve one complete AraBERT snapshot BEFORE requesting GPUs. General synthetic
# training does not install bridge_frozen_text.py, so use the exact local snapshot
# as the Transformers load target for this job.
ARABIC_TEXT_MODEL_ID="${ARABIC_TEXT_MODEL_ID:-aubmindlab/bert-base-arabertv02}"
HF_RESOLUTION="$(python - "${SHARED_PROJECT_DIR}" "${ARABIC_TEXT_MODEL_ID}" <<'PY'
import sys
from hf_offline_runtime import resolve_hf_model_snapshot
root, model_id = sys.argv[1:3]
r = resolve_hf_model_snapshot(model_id, project_dir=root)
print(str(r.cache_root))
print(str(r.snapshot_path))
PY
)"
HF_HOME="$(printf '%s\n' "${HF_RESOLUTION}" | sed -n '1p')"
ARABIC_TEXT_MODEL_RESOLVED_PATH="$(printf '%s\n' "${HF_RESOLUTION}" | sed -n '2p')"
[[ -n "${HF_HOME}" && -n "${ARABIC_TEXT_MODEL_RESOLVED_PATH}" ]] || {
  echo "ERROR: AraBERT offline preflight returned an empty path." >&2
  exit 2
}
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_RESOLVED_PATH}"
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false

# The fixed63 dataset should really be fixed63. Check all requested transcripts on
# the login node so a malformed sample cannot waste a GPU allocation later.
python - "${DATA_DIR}/texts" "${NUM_SAMPLES}" <<'PY'
from pathlib import Path
import sys
texts = Path(sys.argv[1])
count = int(sys.argv[2])
problems = []
for i in range(1, count + 1):
    for side in (1, 2):
        path = texts / f"text{side}_{i}.txt"
        if not path.is_file():
            problems.append(f"missing:{path.name}")
        else:
            n = len(path.read_text(encoding="utf-8").strip())
            if n != 63:
                problems.append(f"length:{path.name}={n}")
        if len(problems) >= 10:
            break
    if len(problems) >= 10:
        break
if problems:
    raise SystemExit("ERROR: fixed63 transcript preflight failed: " + ", ".join(problems))
print(f"fixed63_preflight_ok transcripts={2*count} chars=63")
PY

TRAIN_EXPECTED_BRANCH="${TRAIN_EXPECTED_BRANCH:-$(git branch --show-current)}"
TRAIN_EXPECTED_COMMIT="${TRAIN_EXPECTED_COMMIT:-$(git rev-parse HEAD)}"
TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"
TRAIN_SHARED_PROJECT_DIR="${TRAIN_SHARED_PROJECT_DIR:-${SHARED_PROJECT_DIR}}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${TRAIN_SHARED_PROJECT_DIR}/.jax_cache/stage1_fixed63}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

export DATA_DIR JOB_ID NUM_SAMPLES EPOCHS LEARNING_RATE WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE VECTOR_SIZE
export MAX_TEXT_SPAN_CHARS MAX_TEXT_TOKEN_CHARS MAX_WINDOWS_PER_SPAN SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN
export SPAN_INCLUDE_SPACE_CONTEXT SPAN_ALLOW_CHARACTER_SPACE_SURFACES SPAN_DTW_BACKEND
export CONTRASTIVE_SOFT_DTW_GAMMA CONTRASTIVE_MARGIN CONTRASTIVE_TEMPERATURE NUM_NEGATIVES SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE
export LOCAL_HARD_NEGATIVE_WEIGHT LOCAL_HARD_NEGATIVE_MARGIN IMAGE_PAIR_LOSS_WEIGHT SEQUENCE_CONSISTENCY_LOSS_WEIGHT
export IMAGE_VARIANCE_LOSS_WEIGHT IMAGE_VARIANCE_TARGET_STD IMAGE_TEXT_LOSS_ON_BOTH_LINES
export DATASET_TYPE SYNTHETIC_MANUSCRIPT_AUGMENT REAL_AUGMENT AUGMENT REAL_USE_EXTRA_NO_SHARED REAL_UNIQUE_LINE_ADAPTATION REAL_USE_EXPLICIT_SPLIT_MANIFESTS
export USE_SEQUENCE_ALIGNMENT_RANKING SEQUENCE_RANKING_WEIGHT BRIDGE_CROSS_TEXT_WEIGHT
export VALID_EVERY_N_EPOCHS VALID_MAX_BATCHES NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS BATCH_SIZE
export CNN_BACKBONE USE_BILSTM BILSTM_LAYERS BILSTM_HIDDEN_DIM USE_LOCAL_WINDOW_GROUPING LOCAL_WINDOW_GROUP_SIZE
export CONDA_ENV GPU_RESOURCE PARTITION CPUS_PER_TASK MEMORY TIME_LIMIT MAIL_USER
export ARABIC_TEXT_MODEL_ID ARABIC_TEXT_MODEL_RESOLVED_PATH ARABIC_TEXT_MODEL_NAME HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TOKENIZERS_PARALLELISM
export TRAIN_EXPECTED_BRANCH TRAIN_EXPECTED_COMMIT TRAIN_EXPECTED_BACKEND TRAIN_SHARED_PROJECT_DIR JAX_COMPILATION_CACHE_DIR

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  STRIDE_PIXELS="$(python - <<PY
print(max(1, int(${WINDOW_SIZE} * ${STRIDE_RATIO})))
PY
)"
  echo "=== STAGE-1 SYNTHETIC FIXED63 SUBMISSION ==="
  echo "branch=${TRAIN_EXPECTED_BRANCH}"
  echo "commit=${TRAIN_EXPECTED_COMMIT}"
  echo "backend=${MODEL_BACKEND}"
  echo "dataset=${DATA_DIR}"
  echo "job=${JOB_ID}"
  echo "window/stride=${WINDOW_SIZE}/${STRIDE_PIXELS}"
  echo "loss=SpanDTW(gamma=${CONTRASTIVE_SOFT_DTW_GAMMA},margin=${CONTRASTIVE_MARGIN},temp=${CONTRASTIVE_TEMPERATURE})"
  echo "negatives=${NUM_NEGATIVES} active=${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE} local_hard=${LOCAL_HARD_NEGATIVE_WEIGHT} pair=${IMAGE_PAIR_LOSS_WEIGHT} sequence=${SEQUENCE_CONSISTENCY_LOSS_WEIGHT} variance=${IMAGE_VARIANCE_LOSS_WEIGHT}"
  echo "hf_preflight_ok model_id=${ARABIC_TEXT_MODEL_ID}"
  echo "hf_preflight_ok snapshot=${ARABIC_TEXT_MODEL_RESOLVED_PATH}"
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
    --export=ALL,PROJECT_DIR="${SHARED_PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

# Run the exact submitted commit from node-local scratch so switching the shared
# checkout to submit the other architecture cannot alter a queued/running job.
if [[ "${STAGE1_RUNTIME_CLONE_ACTIVE:-0}" != "1" ]]; then
  RUNTIME_PARENT="${SLURM_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
  RUNTIME_PROJECT_DIR="${RUNTIME_PARENT}/AlignmentProject-stage1-${SLURM_JOB_ID}-${TRAIN_EXPECTED_COMMIT:0:12}"
  rm -rf "${RUNTIME_PROJECT_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --shared --no-checkout "${TRAIN_SHARED_PROJECT_DIR}" "${RUNTIME_PROJECT_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${RUNTIME_PROJECT_DIR}" checkout --quiet -B "${TRAIN_EXPECTED_BRANCH}" "${TRAIN_EXPECTED_COMMIT}"
  for name in DataSet Weights out logs .jax_cache wandb; do
    source_path="${TRAIN_SHARED_PROJECT_DIR}/${name}"
    target_path="${RUNTIME_PROJECT_DIR}/${name}"
    if [[ -e "${source_path}" || -L "${source_path}" ]]; then
      rm -rf "${target_path}"
      ln -s "${source_path}" "${target_path}"
    fi
  done
  exec env PROJECT_DIR="${RUNTIME_PROJECT_DIR}" TRAIN_SHARED_PROJECT_DIR="${TRAIN_SHARED_PROJECT_DIR}" STAGE1_RUNTIME_CLONE_ACTIVE=1 \
    bash "${RUNTIME_PROJECT_DIR}/scripts/train/submit_stage1_synthetic_fixed63.sh"
fi

cd "${PROJECT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_ASYNC_ERROR_HANDLING=1

ARGS=(
  training_runtime/entrypoint.py
  --job_id "${JOB_ID}"
  --dataset_type synthetic
  --data_dir "${DATA_DIR}"
  --num_samples "${NUM_SAMPLES}"
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
