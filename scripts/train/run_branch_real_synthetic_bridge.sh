#!/usr/bin/env bash
# Architecture-neutral fine-tuning on the OFFLINE real-conditioned synthetic bridge.
set -euo pipefail
set -a
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_DIR}"; mkdir -p out logs
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v1}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
JOB_ID="${JOB_ID:-$(git branch --show-current | tr '/' '-')-real-synthetic-bridge-v1}"
[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing ${DATA_DIR}/dataset_manifest.jsonl" >&2; exit 2; }
[[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: PRETRAINED_WEIGHTS is required." >&2; exit 2; }
MODEL_BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
read -r CKPT_BACKEND CKPT_BACKBONE CKPT_BILSTM < <(python - "${PRETRAINED_WEIGHTS}" <<'PY'
import sys, torch
p=sys.argv[1]
try: x=torch.load(p,map_location='cpu',weights_only=False)
except TypeError: x=torch.load(p,map_location='cpu')
c=x.get('model_config',{}) if isinstance(x,dict) else {}
b=str(c.get('model_backend',c.get('visual_encoder_type',''))).strip().lower() or 'cnn_bilstm'
if b in {'cnn','cnn_only'}: b='cnn_bilstm'
backbone=str(c.get('cnn_backbone','resnet34')).strip().lower().replace('-','') if b=='cnn_bilstm' else '-'
print(b,backbone,'1' if bool(c.get('use_bilstm',b!='vit')) else '0')
PY
)
[[ "${CKPT_BACKEND}" == "${MODEL_BACKEND}" ]] || { echo "ERROR: checkpoint=${CKPT_BACKEND}, branch=${MODEL_BACKEND}" >&2; exit 2; }
case "${MODEL_BACKEND}" in
  cnn_bilstm) export CNN_BACKBONE="${CNN_BACKBONE:-${CKPT_BACKBONE}}" USE_BILSTM="${USE_BILSTM:-${CKPT_BILSTM}}"; DEFAULT_ACCUM=1 ;;
  vit) export USE_BILSTM=0 USE_LOCAL_WINDOW_GROUPING=0; DEFAULT_ACCUM=4 ;;
  dinov3_convnext) export USE_BILSTM="${USE_BILSTM:-${CKPT_BILSTM}}" DINOV3_ALLOW_RANDOM_INIT="${DINOV3_ALLOW_RANDOM_INIT:-1}"; : "${DINOV3_REPO_DIR:?Set DINOV3_REPO_DIR}"; DEFAULT_ACCUM=4 ;;
  *) echo "ERROR: unsupported backend ${MODEL_BACKEND}" >&2; exit 2 ;;
esac
export DATA_DIR PRETRAINED_WEIGHTS JOB_ID REAL_MANIFEST_NAME=dataset_manifest.jsonl REAL_USE_EXTRA_NO_SHARED=1 REAL_UNIQUE_LINE_ADAPTATION=0 REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0 NO_SHARED_IMAGE_OBJECTIVE=synthetic_bridge REAL_AUGMENT=0 AUGMENT=0 IMAGE_TEXT_LOSS_ON_BOTH_LINES=1 REAL_TRAIN_SAMPLES_PER_EPOCH=0
export BRIDGE_TRAIN_SAMPLES_PER_EPOCH="${BRIDGE_TRAIN_SAMPLES_PER_EPOCH:-0}" EPOCHS="${EPOCHS:-8}" LEARNING_RATE="${LEARNING_RATE:-7.5e-7}" VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}" VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
export WINDOW_SIZE="${WINDOW_SIZE:-32}" STRIDE_RATIO="${STRIDE_RATIO:-0.5}" WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}" MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-2}" REAL_MAX_TEXT_SPAN_CHARS="${REAL_MAX_TEXT_SPAN_CHARS:-${MAX_TEXT_SPAN_CHARS}}" MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-3}"
export NUM_NEGATIVES="${NUM_NEGATIVES:-10}" SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}" USE_LOCAL_HARD_NEGATIVES=1 LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}" USE_IMAGE_PAIR_CONTRASTIVE=1 IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.10}" SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0 USE_SEQUENCE_ALIGNMENT_RANKING=1 SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.08}" SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.50}" SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}" SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR="${SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR:-0.08}" SEQUENCE_RANKING_SCORE_MARGIN="${SEQUENCE_RANKING_SCORE_MARGIN:-0.10}"
NUM_GPUS="${NUM_GPUS:-2}"; EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"; GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_ACCUM}}"; DENOM=$((NUM_GPUS*GRADIENT_ACCUMULATION_STEPS)); (( EFFECTIVE_GLOBAL_BATCH_SIZE % DENOM == 0 )) || { echo "ERROR: global batch not divisible by ${DENOM}" >&2; exit 2; }; BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE/DENOM)); export NUM_GPUS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
CONDA_ENV="${CONDA_ENV:-manucripts_align}"; GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"; PARTITION="${PARTITION:-rtx4090}"; CPUS_PER_TASK="${CPUS_PER_TASK:-$((8*NUM_GPUS))}"; MEMORY="${MEMORY:-96G}"; TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"; MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"; SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting bridge backend=${MODEL_BACKEND} job=${JOB_ID}; offline rendering; best checkpoint preserved"
  sbatch --partition="${PARTITION}" --job-name="${JOB_ID}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --gpus="${GPU_RESOURCE}:${NUM_GPUS}" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" --export=ALL "${SCRIPT_PATH}"; exit 0
fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE=false NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_ASYNC_ERROR_HANDLING=1
BRANCH="$(git branch --show-current)"; COMMIT="$(git rev-parse HEAD)"; export TRAIN_EXPECTED_BRANCH="${BRANCH}" TRAIN_EXPECTED_COMMIT="${COMMIT}" TRAIN_EXPECTED_BACKEND="${MODEL_BACKEND}"
ARGS=(training_runtime/entrypoint.py --job_id "${JOB_ID}" --dataset_type real --data_dir "${DATA_DIR}" --pretrained_weights "${PRETRAINED_WEIGHTS}" --epochs "${EPOCHS}" --learning_rate "${LEARNING_RATE}" --window_size "${WINDOW_SIZE}" --stride_ratio "${STRIDE_RATIO}" --window_overlap_mode "${WINDOW_OVERLAP_MODE}" --num_negatives "${NUM_NEGATIVES}" --no-augment)
if (( NUM_GPUS > 1 )); then exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${ARGS[@]}"; fi
exec python "${ARGS[@]}"
