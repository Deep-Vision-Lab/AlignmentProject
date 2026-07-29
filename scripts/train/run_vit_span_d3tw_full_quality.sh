#!/usr/bin/env bash
# ViT front-end for the optimized generic full-quality SLURM/DDP launcher.
#
# The base launcher remains the single source of truth for dataset handling,
# SLURM resources, NCCL workarounds, torchrun, batch sizing, losses, and resume.
# This wrapper changes only the Python training entry point to train_vit.py.
#
# Example:
#   DATASET_TYPE=synthetic NUM_SAMPLES=8000 NUM_GPUS=2 \
#     JOB_ID=vit_synthetic_8k_gpu2 \
#     bash scripts/train/run_vit_span_d3tw_full_quality.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_vit_span_d3tw_full_quality.sh" >&2
  echo "Override configuration through environment variables." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_LAUNCHER="${PROJECT_DIR}/scripts/train/run_span_d3tw_full_quality.sh"

[[ -f "${BASE_LAUNCHER}" ]] || {
  echo "ERROR: optimized base launcher not found: ${BASE_LAUNCHER}" >&2
  exit 1
}
[[ -f "${PROJECT_DIR}/train_vit.py" ]] || {
  echo "ERROR: ViT training entry point not found: ${PROJECT_DIR}/train_vit.py" >&2
  exit 1
}
[[ -f "${PROJECT_DIR}/vit_embedding_model.py" ]] || {
  echo "ERROR: ViT model not found: ${PROJECT_DIR}/vit_embedding_model.py" >&2
  exit 1
}

export PROJECT_DIR
export VISUAL_ENCODER_TYPE="vit"
export USE_BILSTM="0"
export USE_LOCAL_WINDOW_GROUPING="0"

# ViT architecture. These are checkpointed by train_vit.py.
export VIT_INPUT_HEIGHT="${VIT_INPUT_HEIGHT:-128}"
export VIT_LAYERS="${VIT_LAYERS:-4}"
export VIT_HEADS="${VIT_HEADS:-4}"
export VIT_MLP_DIM="${VIT_MLP_DIM:-512}"
export VIT_DROPOUT="${VIT_DROPOUT:-0.10}"
export VIT_MAX_TOKENS="${VIT_MAX_TOKENS:-256}"
export TORCH_COMPILE_VISUAL="${TORCH_COMPILE_VISUAL:-0}"
export DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"

# Match the proven optimized two-GPU defaults. The base launcher still permits
# every value to be overridden by the calling environment.
export NUM_GPUS="${NUM_GPUS:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
case "${DATASET_TYPE:-synthetic}" in
  synthetic) DEFAULT_VIT_SAMPLES="${NUM_SAMPLES:-8000}" ;;
  real) DEFAULT_VIT_SAMPLES="${NUM_SAMPLES:-10000}" ;;
  *) DEFAULT_VIT_SAMPLES="${NUM_SAMPLES:-8000}" ;;
esac
export JOB_ID="${JOB_ID:-${DATASET_TYPE:-synthetic}_arabic_vit_fullquality_${DEFAULT_VIT_SAMPLES}_gpu${NUM_GPUS}}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-align_vit}"

# Generate a temporary copy of the proven launcher and replace exactly one line:
# the Python entry point inside TRAIN_ARGS. sbatch copies this generated script
# into its spool, so deleting the local temporary file after submission is safe.
TMP_LAUNCHER="$(mktemp "${TMPDIR:-/tmp}/alignment_vit_launcher.XXXXXX.sh")"
cleanup() {
  rm -f "${TMP_LAUNCHER}"
}
trap cleanup EXIT

python3 - "${BASE_LAUNCHER}" "${TMP_LAUNCHER}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
needle = "  train.py\n"
count = text.count(needle)
if count != 1:
    raise SystemExit(
        f"Expected exactly one TRAIN_ARGS entry {needle!r} in {source}, found {count}."
    )
text = text.replace(needle, "  train_vit.py\n", 1)
text = text.replace(
    "Generic full-quality training launcher",
    "ViT full-quality training launcher",
    1,
)
target.write_text(text, encoding="utf-8")
PY
chmod +x "${TMP_LAUNCHER}"

bash "${TMP_LAUNCHER}"
