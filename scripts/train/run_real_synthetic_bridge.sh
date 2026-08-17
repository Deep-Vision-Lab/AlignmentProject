#!/usr/bin/env bash
# Fine-tune a CNN/CNN+BiLSTM checkpoint on the OFFLINE real-synthetic bridge.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v1}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-}"
JOB_ID="${JOB_ID:-cnn_real_synthetic_bridge_v1}"

[[ -s "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: offline bridge dataset is missing: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  echo "Build it first with: sbatch scripts/data/build_real_conditioned_synthetic_bridge.sbatch" >&2
  exit 2
}
[[ -n "${PRETRAINED_WEIGHTS}" && -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: PRETRAINED_WEIGHTS must point to the CNN checkpoint to adapt." >&2
  exit 2
}

# Resolve architecture from the checkpoint. Historical checkpoints have no
# cnn_backbone field and are therefore correctly interpreted as ResNet-34.
read -r CKPT_BACKBONE CKPT_BILSTM < <(
  python - "${PRETRAINED_WEIGHTS}" <<'PY'
import sys
import torch
from checkpoint_backbone_runtime import checkpoint_cnn_backbone
path = sys.argv[1]
backbone = checkpoint_cnn_backbone(path)
if backbone is None:
    raise SystemExit("Bridge launcher currently expects a CNN checkpoint, not ViT/DINOv3.")
payload = torch.load(path, map_location="cpu")
config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
use_bilstm = bool(config.get("use_bilstm", True))
print(backbone, "1" if use_bilstm else "0")
PY
)

if [[ -n "${CNN_BACKBONE:-}" && "${CNN_BACKBONE}" != "${CKPT_BACKBONE}" ]]; then
  echo "ERROR: CNN_BACKBONE=${CNN_BACKBONE} does not match checkpoint backbone=${CKPT_BACKBONE}." >&2
  exit 2
fi
if [[ -n "${USE_BILSTM:-}" && "${USE_BILSTM}" != "${CKPT_BILSTM}" ]]; then
  echo "ERROR: USE_BILSTM=${USE_BILSTM} does not match checkpoint use_bilstm=${CKPT_BILSTM}." >&2
  exit 2
fi
export CNN_BACKBONE="${CKPT_BACKBONE}"
export USE_BILSTM="${CKPT_BILSTM}"

export DATA_DIR PRETRAINED_WEIGHTS JOB_ID
export REAL_MANIFEST_NAME="dataset_manifest.jsonl"
export REAL_USE_EXTRA_NO_SHARED=1
export REAL_UNIQUE_LINE_ADAPTATION=0
export REAL_USE_EXPLICIT_SPLIT_MANIFESTS=0
export NO_SHARED_IMAGE_OBJECTIVE="synthetic_bridge"
export REAL_AUGMENT=0
export AUGMENT=0
export IMAGE_TEXT_LOSS_ON_BOTH_LINES=1

# A longer MAXIMUM run is safe because bridge training keeps checkpoint_best_val.pth.
export EPOCHS="${EPOCHS:-8}"
export VALID_EVERY_N_EPOCHS="${VALID_EVERY_N_EPOCHS:-1}"
export VALID_MAX_BATCHES="${VALID_MAX_BATCHES:-20}"
export LEARNING_RATE="${LEARNING_RATE:-7.5e-7}"
export BRIDGE_TRAIN_SAMPLES_PER_EPOCH="${BRIDGE_TRAIN_SAMPLES_PER_EPOCH:-0}"

# Existing image-text representation remains the main anchor. Pair/sequence terms
# are intentionally moderate for bridge v1.
export USE_LOCAL_HARD_NEGATIVES=1
export LOCAL_HARD_NEGATIVE_WEIGHT="${LOCAL_HARD_NEGATIVE_WEIGHT:-0.10}"
export USE_IMAGE_PAIR_CONTRASTIVE=1
export IMAGE_PAIR_LOSS_WEIGHT="${IMAGE_PAIR_LOSS_WEIGHT:-0.10}"
export SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0
export USE_SEQUENCE_ALIGNMENT_RANKING=1
export SEQUENCE_RANKING_WEIGHT="${SEQUENCE_RANKING_WEIGHT:-0.08}"
export SEQUENCE_RANKING_THRESHOLD="${SEQUENCE_RANKING_THRESHOLD:-0.50}"
export SEQUENCE_RANKING_GAP="${SEQUENCE_RANKING_GAP:--0.30}"
export SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR="${SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR:-0.08}"
export SEQUENCE_RANKING_SCORE_MARGIN="${SEQUENCE_RANKING_SCORE_MARGIN:-0.10}"

export NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
export SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"
export REAL_TRAIN_SAMPLES_PER_EPOCH=0

# Keep the same local geometry used by the successful fixed-63 line.
export WINDOW_SIZE="${WINDOW_SIZE:-32}"
export STRIDE_RATIO="${STRIDE_RATIO:-0.5}"
export WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
export MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"

cat <<EOF
=== REAL-CONDITIONED SYNTHETIC BRIDGE TRAINING ===
job_id=${JOB_ID}
data_dir=${DATA_DIR}
pretrained=${PRETRAINED_WEIGHTS}
cnn_backbone=${CNN_BACKBONE}
use_bilstm=${USE_BILSTM}
epochs_max=${EPOCHS}
learning_rate=${LEARNING_RATE}
bridge_train_samples_per_epoch=${BRIDGE_TRAIN_SAMPLES_PER_EPOCH}
best_checkpoint=Weights/${JOB_ID}/checkpoint_best_val.pth
latest_checkpoint=Weights/${JOB_ID}/checkpoint_latest.pth
EOF

exec bash "${PROJECT_DIR}/scripts/train/run_real_finetune.sh"
