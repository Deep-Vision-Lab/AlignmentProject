#!/usr/bin/env bash
#SBATCH --job-name=real_arabic_finetune
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/ArabicDataset}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_point2/model_latest.pth}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${HOME}/.conda/envs/manucripts_align/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$(dirname "${TORCHRUN_BIN}")/python}"

cd "${PROJECT_DIR}"
mkdir -p out Results/Diagnostics/ArabicDatasetSyntheticStyle

[[ -x "${TORCHRUN_BIN}" ]] || { echo "ERROR: torchrun not executable: ${TORCHRUN_BIN}" >&2; exit 2; }
[[ -d "${DATASET}" ]] || { echo "ERROR: ArabicDataset not found: ${DATASET}" >&2; exit 2; }
[[ -f "${DATASET}/dataset_manifest_full_pairs.jsonl" ]] || { echo "ERROR: missing ${DATASET}/dataset_manifest_full_pairs.jsonl" >&2; exit 2; }
[[ -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: synthetic checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2; exit 2; }

# Use the architecture that produced the stronger fused Point-2 results.
export MODEL_BACKEND_VARIANT="resnet_token"
export JOB_NAME="${JOB_NAME:-res18_tinyvit_real_arabic_synthstyle}"
export RESTORATION_EPOCHS="${RESTORATION_EPOCHS:-30}"
export RESTORATION_LEARNING_RATE="${RESTORATION_LEARNING_RATE:-2e-5}"
# 0 means use all manifest rows that pass the real-data label filters.
export RESTORATION_NUM_SAMPLES="${RESTORATION_NUM_SAMPLES:-0}"
# 0 means do not repeat the same real rows to an artificial epoch length.
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"

# ArabicDataset preprocessing requested for this run:
# crop foreground -> aspect-preserving 1024x128 -> binary white ink on black.
export DATASET_TYPE=real
export REAL_SYNTHETIC_STYLE=1
export REAL_AUGMENT=0
export REAL_MANIFEST_NAME="dataset_manifest_full_pairs.jsonl"
export REAL_INDEPENDENT_LINES=1
export REAL_DATASET_LABELS="all"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
# Group independent lines by their source page so lines from one page do not
# leak across train/validation/test.
export REAL_SPLIT_BY_PAIR_ID=1
export REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"
export REAL_FILTER_INFEASIBLE_SPAN_DTW=0
export ZERO_SHOT_PREPROCESS=1
export ZERO_SHOT_FOREGROUND_CROP=1
export ZERO_SHOT_PRESERVE_ASPECT=1
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"

# Current positive letter-DTW objective.
export RESTORATION_TRAINING_STAGE=align
export RESTORATION_NUM_NEGATIVES="${RESTORATION_NUM_NEGATIVES:-0}"
export RESTORATION_CONTRASTIVE_WEIGHT="${RESTORATION_CONTRASTIVE_WEIGHT:-0.0}"
export RESTORATION_DTW_STRIDE_RATIO="${RESTORATION_DTW_STRIDE_RATIO:-0.50}"
export POSITIVE_LETTER_DTW_WEIGHT="${POSITIVE_LETTER_DTW_WEIGHT:-1.0}"
export POSITIVE_LETTER_DTW_GAMMA_START="${POSITIVE_LETTER_DTW_GAMMA_START:-0.50}"
export POSITIVE_LETTER_DTW_GAMMA_END="${POSITIVE_LETTER_DTW_GAMMA_END:-0.05}"
export POSITIVE_LETTER_DTW_ANNEAL_EPOCHS="${POSITIVE_LETTER_DTW_ANNEAL_EPOCHS:-10}"
export POSITIVE_LETTER_DTW_VERTICAL_PENALTY="${POSITIVE_LETTER_DTW_VERTICAL_PENALTY:-0.05}"
export POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY="${POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY:-0.30}"
export POSITIVE_LETTER_DTW_POSITION_PRIOR="${POSITIVE_LETTER_DTW_POSITION_PRIOR:-0.15}"
export POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE=1
export POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE="${POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE:-0.10}"
export POSITIVE_LETTER_DTW_COST_MODE="${POSITIVE_LETTER_DTW_COST_MODE:-full_alphabet_nll}"
export LETTER_CODEBOOK_SEED="${LETTER_CODEBOOK_SEED:-1234}"

export PRETRAINED_CACHE_DIR="${PRETRAINED_CACHE_DIR:-${PROJECT_DIR}/Pretrained}"
export TORCH_HOME="${TORCH_HOME:-${PRETRAINED_CACHE_DIR}/torch}"
export HF_HOME="${HF_HOME:-${PRETRAINED_CACHE_DIR}/huggingface}"
export RESNET18_PRETRAINED=1
export TINY_VIT_PRETRAINED=1
export TINY_VIT_PRETRAINED_MODEL="${TINY_VIT_PRETRAINED_MODEL:-facebook/deit-tiny-patch16-224}"
export PRETRAINED_LOCAL_ONLY=1
export DDP_STATIC_GRAPH=1
export FORCE_DDP_STATIC_GRAPH=0
export SPAN_DTW_BACKEND=torch
export USE_LOCAL_HARD_NEGATIVES=0
export USE_IMAGE_PAIR_CONTRASTIVE=0
export TINY_VIT_DROPOUT="${TINY_VIT_DROPOUT:-0.10}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_VISUAL=0
export GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_DROP_LAST=0

"${PYTHON_BIN}" -m py_compile \
  train.py model_backend.py vlm_restoration_positive_dtw.py \
  zero_shot_preprocessing.py RealDataSet.py AugmentedRealDataLoader.py

PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS}" "${PYTHON_BIN}" - <<'PY'
import os
import torch

path = os.environ["PRETRAINED_WEIGHTS"]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
backend = str(config.get("model_backend", ""))
physical = "physical_window" in backend or str(
    config.get("model_backend_variant", "")
).lower() == "physical_window"
print("Checkpoint preflight:")
print("  path       =", path)
print("  backend    =", backend or "<legacy/unset>")
print("  family     =", config.get("architecture_family"))
print("  local      =", config.get("local_encoder_type"))
print("  context    =", config.get("context_input", "ResNet-token TinyViT (legacy config)"))
if physical:
    raise SystemExit(
        "Expected the ResNet-token TinyViT synthetic checkpoint, but this checkpoint "
        "is the physical-window ablation."
    )
PY

"${PYTHON_BIN}" -m pytest -q \
  tests/test_real_synthetic_style_preprocessing.py \
  tests/test_real_independent_lines.py \
  tests/test_restoration_positive_dtw.py

DATASET="${DATASET}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from RealDataSet import ArabicManifestIndependentLineDataset

root = Path(os.environ["DATASET"])
dataset = ArabicManifestIndependentLineDataset(
    root / "dataset_manifest_full_pairs.jsonl",
    transform=None,
    text_key="text_original_path",
    validate_paths=False,
)
print("Independent-line dataset census:")
print("  unique lines =", len(dataset))
print("  source pages =", len({sample["pair_id"] for sample in dataset.samples}))
print("  objective    = each line vs its own transcript; pair labels unused")
PY

# Save a few transformed ArabicDataset examples before spending GPU time.
REAL_SYNTHETIC_STYLE=1 "${PYTHON_BIN}" scripts/preview_arabic_synthetic_style.py \
  --dataset "${DATASET}" \
  --output-dir Results/Diagnostics/ArabicDatasetSyntheticStyle \
  --samples 6

echo "============================================================"
echo "Real ArabicDataset fine-tune"
echo "dataset              = ${DATASET}"
echo "manifest             = ${DATASET}/dataset_manifest_full_pairs.jsonl"
echo "mode                 = FINE-TUNE (never train from zero)"
echo "initial weights      = ${PRETRAINED_WEIGHTS}"
echo "backend              = resnet_token (ResNet18 -> TinyViT -> fused)"
echo "epochs               = ${RESTORATION_EPOCHS}"
echo "learning rate        = ${RESTORATION_LEARNING_RATE}"
echo "preprocessing        = foreground crop + white ink on black"
echo "output geometry      = 1024x128, aspect preserved"
echo "online augmentation  = disabled"
echo "training view        = ALL unique independent lines (A+B, deduplicated)"
echo "pair labels          = ignored"
echo "TinyViT dropout      = ${TINY_VIT_DROPOUT}"
echo "============================================================"

"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --dataset "${DATASET}" \
  --weights "${PRETRAINED_WEIGHTS}"
