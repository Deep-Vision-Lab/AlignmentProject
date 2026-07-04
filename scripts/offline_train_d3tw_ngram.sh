#!/usr/bin/env bash
set -euo pipefail

echo "[offline_train] date: $(date)"
echo "[offline_train] host: $(hostname)"
echo "[offline_train] pwd: $(pwd)"
echo "[offline_train] git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
echo "[offline_train] git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[offline_train] python path: $(command -v python)"
echo "[offline_train] python version: $(python --version)"
python - <<'PY'
try:
    import torch
    print("[offline_train] cuda available:", torch.cuda.is_available())
    print("[offline_train] cuda device count:", torch.cuda.device_count())
except Exception as exc:
    print("[offline_train] cuda availability: unavailable because torch import failed:", exc)
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[offline_train] nvidia-smi not found"
fi

mkdir -p out logs paper_figures/outputs

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PROFILE_TIMING="${PROFILE_TIMING:-0}"
export USE_AMP="${USE_AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-fp16}"

export GRADIENT_CHECK_ENABLED="${GRADIENT_CHECK_ENABLED:-1}"
export GRADIENT_CHECK_INTERVAL="${GRADIENT_CHECK_INTERVAL:-50}"
export GRADIENT_CHECK_FIRST_N_BATCHES="${GRADIENT_CHECK_FIRST_N_BATCHES:-3}"
export GRADIENT_FAIL_FAST="${GRADIENT_FAIL_FAST:-1}"

export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
JOB_ID="${JOB_ID:-offline_d3tw_ngram_w32_light_100k_v1}"
DATA_DIR="${DATA_DIR:-DataSet/Synthetic_Arabic}"

EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-4}"
NUM_NEGATIVES="${NUM_NEGATIVES:-10}"

WINDOW_SIZE="${WINDOW_SIZE:-32}"
WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-light_overlap}"

ALIGNMENT_LOSS_TYPE="${ALIGNMENT_LOSS_TYPE:-d3tw_char_pool}"
LOSS_TYPE="${LOSS_TYPE:-hybrid}"

TEXT_UNIT_TYPE="${TEXT_UNIT_TYPE:-ngram}"
TEXT_EMBEDDER="${TEXT_EMBEDDER:-orthogonal_char}"

NEGATIVE_MODE="${NEGATIVE_MODE:-length_controlled}"
SEQUENCE_ENCODER_TYPE="${SEQUENCE_ENCODER_TYPE:-bilstm}"

NGRAM_MIN_N="${NGRAM_MIN_N:-1}"
NGRAM_MAX_N="${NGRAM_MAX_N:-3}"
NGRAM_MIN_FREQ="${NGRAM_MIN_FREQ:-2}"
NGRAM_MAX_VOCAB_SIZE="${NGRAM_MAX_VOCAB_SIZE:-5000}"
NGRAM_SKIP_SPACES="${NGRAM_SKIP_SPACES:-true}"
NGRAM_INCLUDE_LIGATURES="${NGRAM_INCLUDE_LIGATURES:-true}"

TOKEN_POOL_WEIGHT="${TOKEN_POOL_WEIGHT:-0.25}"
TOKEN_POOL_TAU="${TOKEN_POOL_TAU:-0.07}"
TOKEN_POOL_WARMUP_EPOCHS="${TOKEN_POOL_WARMUP_EPOCHS:-8}"
TOKEN_POOL_RAMP_EPOCHS="${TOKEN_POOL_RAMP_EPOCHS:-12}"
TOKEN_POOL_DETACH_ALIGNMENT="${TOKEN_POOL_DETACH_ALIGNMENT:-true}"

USE_CHAR_AUX_LOSS="${USE_CHAR_AUX_LOSS:-true}"
CHAR_AUX_WEIGHT="${CHAR_AUX_WEIGHT:-0.25}"
CHAR_AUX_TAU="${CHAR_AUX_TAU:-0.07}"
CHAR_AUX_WARMUP_EPOCHS="${CHAR_AUX_WARMUP_EPOCHS:-8}"
CHAR_AUX_RAMP_EPOCHS="${CHAR_AUX_RAMP_EPOCHS:-10}"

USE_BIGRAM_TOKEN_LOSS="${USE_BIGRAM_TOKEN_LOSS:-false}"

if [[ "$TEXT_EMBEDDER" == "fasttext" && -z "${TEXT_EMBEDDER_MODEL_PATH:-}" ]]; then
  echo "ERROR: offline FastText requires TEXT_EMBEDDER_MODEL_PATH to point to a local model.bin." >&2
  exit 2
fi

export JOB_ID DATA_DIR

echo "[offline_train] JOB_ID=$JOB_ID"
echo "[offline_train] NUM_SAMPLES=$NUM_SAMPLES"
echo "[offline_train] DATA_DIR=$DATA_DIR"
echo "[offline_train] TEXT_EMBEDDER=$TEXT_EMBEDDER TEXT_UNIT_TYPE=$TEXT_UNIT_TYPE"
echo "[offline_train] ALIGNMENT_LOSS_TYPE=$ALIGNMENT_LOSS_TYPE LOSS_TYPE=$LOSS_TYPE"

python train.py \
  --job_id "$JOB_ID" \
  --data_dir "$DATA_DIR" \
  --alignment_loss_type "$ALIGNMENT_LOSS_TYPE" \
  --loss_type "$LOSS_TYPE" \
  --text_unit_type "$TEXT_UNIT_TYPE" \
  --text_embedder_type "$TEXT_EMBEDDER" \
  --window_size "$WINDOW_SIZE" \
  --window_overlap_mode "$WINDOW_OVERLAP_MODE" \
  --negative_mode "$NEGATIVE_MODE" \
  --num_negatives "$NUM_NEGATIVES" \
  --ngram_min_n "$NGRAM_MIN_N" \
  --ngram_max_n "$NGRAM_MAX_N" \
  --ngram_min_freq "$NGRAM_MIN_FREQ" \
  --ngram_max_vocab_size "$NGRAM_MAX_VOCAB_SIZE" \
  --ngram_skip_spaces "$NGRAM_SKIP_SPACES" \
  --ngram_include_ligatures "$NGRAM_INCLUDE_LIGATURES" \
  --token_pool_weight "$TOKEN_POOL_WEIGHT" \
  --token_pool_tau "$TOKEN_POOL_TAU" \
  --token_pool_warmup_epochs "$TOKEN_POOL_WARMUP_EPOCHS" \
  --token_pool_ramp_epochs "$TOKEN_POOL_RAMP_EPOCHS" \
  --token_pool_detach_alignment "$TOKEN_POOL_DETACH_ALIGNMENT" \
  --use_char_aux_loss "$USE_CHAR_AUX_LOSS" \
  --char_aux_weight "$CHAR_AUX_WEIGHT" \
  --char_aux_tau "$CHAR_AUX_TAU" \
  --char_aux_warmup_epochs "$CHAR_AUX_WARMUP_EPOCHS" \
  --char_aux_ramp_epochs "$CHAR_AUX_RAMP_EPOCHS" \
  --use_bigram_token_loss "$USE_BIGRAM_TOKEN_LOSS" \
  --sequence_encoder_type "$SEQUENCE_ENCODER_TYPE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR"

required_files=(
  "Weights/$JOB_ID/model_latest.pth"
  "Weights/$JOB_ID/checkpoint_latest.pth"
  "Weights/$JOB_ID/token_bank.json"
  "Weights/$JOB_ID/ngram_vocab.json"
  "Weights/$JOB_ID/token_bank_embeddings.pt"
)

if [[ "${USE_CHAR_AUX_LOSS,,}" == "true" || "$USE_CHAR_AUX_LOSS" == "1" ]]; then
  required_files+=(
    "Weights/$JOB_ID/char_bank.json"
    "Weights/$JOB_ID/char_bank_embeddings.pt"
  )
fi

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required output missing: $path" >&2
    missing=1
  else
    echo "[offline_train] verified: $path"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 3
fi

python - <<'PY'
import os
import torch

job_id = os.environ["JOB_ID"]
ckpt_path = f"Weights/{job_id}/model_latest.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
print("checkpoint keys:", sorted(ckpt.keys()))
print("has image_model_state_dict:", "image_model_state_dict" in ckpt)
print("has text_embedder_state_dict:", "text_embedder_state_dict" in ckpt)
print("text_embedder_type:", ckpt.get("text_embedder_type"))
print("model_config:", ckpt.get("model_config", {}))
PY
