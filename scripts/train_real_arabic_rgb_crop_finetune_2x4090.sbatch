#!/usr/bin/env bash
#SBATCH --job-name=real_arabic_gray_bbox_aug
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
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_real_arabic_rgb_crop/model_latest.pth}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${HOME}/.conda/envs/manucripts_align/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-$(dirname "${TORCHRUN_BIN}")/python}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
mkdir -p out Results/Diagnostics/RealBBoxCrop Results/Diagnostics/RealScanAugmentation

[[ -x "${TORCHRUN_BIN}" ]] || { echo "ERROR: torchrun not executable: ${TORCHRUN_BIN}" >&2; exit 2; }
[[ -d "${DATASET}" ]] || { echo "ERROR: ArabicDataset not found: ${DATASET}" >&2; exit 2; }
[[ -f "${PRETRAINED_WEIGHTS}" ]] || { echo "ERROR: initial checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2; exit 2; }

# Use the architecture that produced the stronger fused Point-2 results.
export MODEL_BACKEND_VARIANT="resnet_token"
export JOB_NAME="${JOB_NAME:-res18_tinyvit_real_gray_full1024_sigreg_l020}"
export RESTORATION_EPOCHS="${RESTORATION_EPOCHS:-30}"
export RESTORATION_LEARNING_RATE="${RESTORATION_LEARNING_RATE:-2e-5}"
# 0 means use all manifest rows that pass the real-data label filters.
export RESTORATION_NUM_SAMPLES="${RESTORATION_NUM_SAMPLES:-0}"
# Keep the explicit target at 0 and derive it from the training split.
# Three copies means each source line is loaded three times per epoch, with a
# fresh random scan augmentation on every access.
export REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
export REAL_TRAIN_AUGMENT_COPIES="${REAL_TRAIN_AUGMENT_COPIES:-3}"

# ArabicDataset preprocessing for the new run:
# source XML text envelope -> crop ALL FOUR SIDES -> true grayscale ->
# training-only scan augmentation -> direct 1024x128 resize with NO artificial canvas.
export DATASET_TYPE=real
export REAL_SYNTHETIC_STYLE=0
export REAL_BINARIZE=0
export REAL_BINARIZE_AUTO_INVERT=0
export REAL_BINARIZE_AUTOCONTRAST=0
# Disable the old paired augmentor; REAL_SCAN_AUGMENT is the only augmentation.
export REAL_AUGMENT=0
export VISUAL_INPUT_CHANNELS=1
export VISUAL_GRAYSCALE=1
export REAL_GRAYSCALE=1
export REAL_BBOX_CROP=1
export REAL_BBOX_CROP_STRICT=1
export REAL_BBOX_MARGIN_RATIO="${REAL_BBOX_MARGIN_RATIO:-0.05}"
export REAL_BBOX_MIN_MARGIN_PX="${REAL_BBOX_MIN_MARGIN_PX:-2}"
# The XML crop already identifies text support. Do not run a second heuristic
# foreground crop on top of it.
export ZERO_SHOT_FOREGROUND_CROP=0

# Strong but non-geometric scan augmentation, training split only.
export REAL_SCAN_AUGMENT=1
export REAL_SCAN_AUGMENT_PROB="${REAL_SCAN_AUGMENT_PROB:-0.95}"
export REAL_SCAN_BRIGHTNESS_DELTA="${REAL_SCAN_BRIGHTNESS_DELTA:-0.12}"
export REAL_SCAN_CONTRAST_DELTA="${REAL_SCAN_CONTRAST_DELTA:-0.22}"
export REAL_SCAN_BLUR_PROB="${REAL_SCAN_BLUR_PROB:-0.55}"
export REAL_SCAN_BLUR_RADIUS_MIN="${REAL_SCAN_BLUR_RADIUS_MIN:-0.35}"
export REAL_SCAN_BLUR_RADIUS_MAX="${REAL_SCAN_BLUR_RADIUS_MAX:-1.40}"
export REAL_SCAN_GAUSSIAN_NOISE_PROB="${REAL_SCAN_GAUSSIAN_NOISE_PROB:-0.75}"
export REAL_SCAN_GAUSSIAN_NOISE_STD_MIN="${REAL_SCAN_GAUSSIAN_NOISE_STD_MIN:-5.0}"
export REAL_SCAN_GAUSSIAN_NOISE_STD_MAX="${REAL_SCAN_GAUSSIAN_NOISE_STD_MAX:-16.0}"
export REAL_SCAN_SPECKLE_PROB="${REAL_SCAN_SPECKLE_PROB:-0.30}"
export REAL_SCAN_SPECKLE_FRACTION="${REAL_SCAN_SPECKLE_FRACTION:-0.0010}"
export REAL_MANIFEST_NAME="dataset_manifest_full_pairs.jsonl"
export REAL_INDEPENDENT_LINES=1
# Critical for the small real dataset: scan every line image + its own
# transcript directly from page_pairs, including lines with no aligned partner.
# Pair labels/positive partners are not required by the positive image->text DTW.
export REAL_ALL_PAGE_LINES=1
export REAL_DATASET_LABELS="all"
export REAL_TEXT_KEY="${REAL_TEXT_KEY:-text_original_path}"
export REAL_MIN_TEXT_SCORE="${REAL_MIN_TEXT_SCORE:-0.0}"
# Group independent lines by their source page so lines from one page do not
# leak across train/validation/test.
export REAL_SPLIT_BY_PAIR_ID=1
export REAL_VALIDATE_PATHS="${REAL_VALIDATE_PATHS:-0}"
export REAL_FILTER_INFEASIBLE_SPAN_DTW=0
export ZERO_SHOT_PREPROCESS=1
export ZERO_SHOT_CROP_MODE=vertical_borders
# Fixed 1024px field gives 63 windows for width=32/stride=16. The XML-cropped
# line is resized directly to fill 1024x128, so there is NO artificial canvas
# and all 63 windows are valid for TinyViT, DTW and SIGReg.
export PACK_VALID_WINDOWS=0
export FULL_IMAGE_NO_PADDING=1
export ZERO_SHOT_PRESERVE_ASPECT=0
export LINE_GEOMETRY_MODE=xml-bbox-gray-full-resize
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"
# DIRECT_RESIZE ignores target ink-height ratio; retained only for checkpoint compatibility.

# Current positive letter-DTW objective.
export RESTORATION_TRAINING_STAGE=align
export RESTORATION_NUM_NEGATIVES="${RESTORATION_NUM_NEGATIVES:-0}"
export RESTORATION_CONTRASTIVE_WEIGHT="${RESTORATION_CONTRASTIVE_WEIGHT:-0.0}"
export RESTORATION_DTW_STRIDE_RATIO="${RESTORATION_DTW_STRIDE_RATIO:-0.50}"
export POSITIVE_LETTER_DTW_WEIGHT="${POSITIVE_LETTER_DTW_WEIGHT:-1.0}"
# Continue from the already fine-tuned real checkpoint without reopening the
# soft-DTW schedule. This isolates SIGReg as the new training change.
export POSITIVE_LETTER_DTW_GAMMA_START="${POSITIVE_LETTER_DTW_GAMMA_START:-0.05}"
export POSITIVE_LETTER_DTW_GAMMA_END="${POSITIVE_LETTER_DTW_GAMMA_END:-0.05}"
export POSITIVE_LETTER_DTW_ANNEAL_EPOCHS="${POSITIVE_LETTER_DTW_ANNEAL_EPOCHS:-1}"
export POSITIVE_LETTER_DTW_VERTICAL_PENALTY="${POSITIVE_LETTER_DTW_VERTICAL_PENALTY:-0.05}"
export POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY="${POSITIVE_LETTER_DTW_HORIZONTAL_PENALTY:-0.30}"
export POSITIVE_LETTER_DTW_POSITION_PRIOR="${POSITIVE_LETTER_DTW_POSITION_PRIOR:-0.15}"
export POSITIVE_LETTER_DTW_DISABLE_HORIZONTAL_WHEN_FEASIBLE=1
export POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE="${POSITIVE_LETTER_DTW_COMPETITION_TEMPERATURE:-0.10}"
export POSITIVE_LETTER_DTW_COST_MODE="${POSITIVE_LETTER_DTW_COST_MODE:-full_alphabet_nll}"
export LETTER_CODEBOOK_SEED="${LETTER_CODEBOOK_SEED:-1234}"

# Strong SIGReg on post-vision-norm, pre-L2 fused valid-window embeddings.
# DTW still operates on the normalized fused vectors exactly as before.
export SIGREG_LAMBDA="${SIGREG_LAMBDA:-0.20}"
export SIGREG_SKETCH_DIM="${SIGREG_SKETCH_DIM:-1024}"
export SIGREG_NUM_KNOTS="${SIGREG_NUM_KNOTS:-17}"
export SIGREG_T_MIN="${SIGREG_T_MIN:-0.0}"
export SIGREG_T_MAX="${SIGREG_T_MAX:-3.0}"
export SIGREG_MIN_SAMPLES="${SIGREG_MIN_SAMPLES:-32}"
export SIGREG_SLICE_CHUNK="${SIGREG_SLICE_CHUNK:-128}"

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
# Keep neural dropout unchanged so SIGReg is the only new representation
# regularizer in this continuation run.
export TINY_VIT_DROPOUT="${TINY_VIT_DROPOUT:-0.0}"
export FUSION_DROPOUT="${FUSION_DROPOUT:-0.0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_VISUAL=0
export GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_DROP_LAST=1
export MODEL_WEIGHTS_EVERY_N_EPOCHS="${MODEL_WEIGHTS_EVERY_N_EPOCHS:-5}"
# Longitudinal one-sample evaluation during training.  Rank 0 evaluates the
# same held-out sample at epoch 1 and then every 5 epochs using both the
# existing NW/Point-2 evaluator and the hard-DTW/Point-3 evaluator.
export TRAIN_SAMPLE_EVAL="${TRAIN_SAMPLE_EVAL:-1}"
export TRAIN_SAMPLE_EVAL_EPOCHS="${TRAIN_SAMPLE_EVAL_EPOCHS:-1,5,10,15,20,25,30}"
export TRAIN_SAMPLE_EVAL_START_INDEX="${TRAIN_SAMPLE_EVAL_START_INDEX:-1}"
export TRAIN_SAMPLE_EVAL_DEVICE="${TRAIN_SAMPLE_EVAL_DEVICE:-cpu}"
export TRAIN_SAMPLE_EVAL_THRESHOLD="${TRAIN_SAMPLE_EVAL_THRESHOLD:-0.10}"
# BF16 keeps mixed-precision throughput/memory while avoiding FP16 overflow
# from the strong SIGReg backward signal. GradScaler is disabled automatically
# when AMP_DTYPE=bf16.
export USE_AMP=1
export AMP_DTYPE="${AMP_DTYPE:-bf16}"

"${PYTHON_BIN}" -m py_compile \
  train.py trainer_core.py training_optimizations.py training_stability.py \
  model_backend.py vlm_restoration_positive_dtw.py \
  zero_shot_preprocessing.py RealDataSet.py AugmentedRealDataLoader.py \
  real_line_bbox_crop.py real_scan_augmentation.py \
  Evaluation/eval_point2.py Evaluation/point2_runtime.py \
  Evaluation/eval_yelda.py Evaluation/yelda_geometry.py \
  Evaluation/eval_point3_hard_paths.py

bash -n scripts/eval_point2_fused_visual_20_1x4090.sbatch

"${PYTHON_BIN}" - <<'PY'
from Evaluation import eval_point3_hard_paths as point3

required = (
    "_transcript_path_for_line",
    "_hard_letter_dtw_path",
    "_letter_dtw_side",
    "save_letter_dtw_overview",
)
missing = [name for name in required if not callable(getattr(point3, name, None))]
if missing:
    raise SystemExit(f"Point-3 DTW evaluator preflight missing callables: {missing}")
print("Point-3 letter-DTW evaluator preflight: OK")
PY

"${PYTHON_BIN}" - <<'PY'
import torch
print("BF16 preflight:")
print("  cuda_available =", torch.cuda.is_available())
print("  device_count   =", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
for index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(index)
    supported = bool(torch.cuda.is_bf16_supported())
    print(f"  device {index}: {name}, bf16_supported={supported}")
    if not supported:
        raise SystemExit(f"BF16 unsupported on CUDA device {index}: {name}")
print("  result         = OK")
PY

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
        "Expected a ResNet-token TinyViT checkpoint, but this checkpoint "
        "is the physical-window ablation."
    )

expected_geometry = {
    "line_width": 1024,
    "line_height": 128,
    "window_size": 32,
    "stride": 16,
    "vit_position_base_tokens": 63,
}
for key, expected in expected_geometry.items():
    actual = config.get(key)
    print(f"  {key:24s} = {actual}")
    if actual is not None and int(actual) != int(expected):
        raise SystemExit(
            f"Initial checkpoint geometry mismatch for {key}: "
            f"expected {expected}, got {actual}"
        )

if bool(config.get("synthetic_binarize", False)):
    raise SystemExit(
        "This continuation launcher accepts the previous RGB checkpoint and migrates "
        "(synthetic_binarize=False)."
    )
if bool(config.get("pack_valid_windows", False)):
    raise SystemExit(
        "Initial checkpoint reports packed/rebased windows, but this launcher "
        "is configured for the fixed 63-position Point-2 grid."
    )

checkpoint_vit_dropout = float(config.get("vit_dropout", 0.0))
requested_vit_dropout = float(os.environ.get("TINY_VIT_DROPOUT", "0.0"))
print("  checkpoint vit_dropout   =", checkpoint_vit_dropout)
print("  requested vit_dropout    =", requested_vit_dropout)
if abs(checkpoint_vit_dropout - requested_vit_dropout) > 1e-9:
    raise SystemExit(
        "TinyViT dropout mismatch between initial checkpoint and continuation run: "
        f"checkpoint={checkpoint_vit_dropout} real={requested_vit_dropout}"
    )

print("  window semantics         = fixed 63-position grid; outer padding masked")
print("  last partial train batch = dropped (synthetic gradient-spike fix)")
PY

"${PYTHON_BIN}" -m pytest -q \
  tests/test_real_rgb_crop_preprocessing.py \
  tests/test_real_line_bbox_crop.py \
  tests/test_real_scan_augmentation.py \
  tests/test_real_all_page_gray_pipeline.py \
  tests/test_real_independent_lines.py \
  tests/test_resnet18_packed_windows.py \
  tests/test_restoration_positive_dtw.py \
  tests/test_point3_letter_dtw.py

DATASET="${DATASET}" "${PYTHON_BIN}" - <<'PY'
import math
import os
import random
from collections import defaultdict
from pathlib import Path

from RealDataSet import ArabicAllPageLinesDataset

root = Path(os.environ["DATASET"])
dataset = ArabicAllPageLinesDataset(
    root,
    transform=None,
    text_key="text_original_path",
    validate_paths=False,
)

groups = defaultdict(list)
for index, sample in enumerate(dataset.samples):
    groups[str(sample["pair_id"])].append(index)
group_ids = list(groups)
random.Random(42).shuffle(group_ids)
train_target = int(0.6 * len(dataset))
valid_target = int(0.2 * len(dataset))
train = []
valid = []
test = []
for group_id in group_ids:
    if len(train) < train_target:
        train.extend(groups[group_id])
    elif len(valid) < valid_target:
        valid.extend(groups[group_id])
    else:
        test.extend(groups[group_id])

world_size = 2
batch_size_per_rank = 32
augment_copies = max(1, int(os.environ.get("REAL_TRAIN_AUGMENT_COPIES", "1")))
explicit_target = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
train_views = (
    explicit_target
    if explicit_target > len(train)
    else len(train) * augment_copies
)
# DistributedSampler(drop_last=True) gives floor(N/world_size) samples/rank;
# DataLoader(drop_last=True) then gives only complete per-rank batches.
samples_per_rank = train_views // world_size
expected_batches = samples_per_rank // batch_size_per_rank

print("All-page-line dataset census:")
print("  unique lines       =", len(dataset))
print("  source pages       =", len(groups))
print("  scan stats         =", dataset.scan_stats)
print("  train/valid/test   =", len(train), len(valid), len(test))
print("  augmentation copies=", augment_copies)
print("  train views/epoch  =", train_views)
print("  batch size / rank  =", batch_size_per_rank)
print("  DDP world size     =", world_size)
print("  expected train batches / rank =", expected_batches)
print("  drop last partial batch =", True)
print("  objective          = each line vs its own transcript; pair labels unused")
PY

# Save the exact four-sided crop and visibly corrupted grayscale training
# examples before spending GPU time.
"${PYTHON_BIN}" scripts/diagnose_real_bbox_crop.py \
  --dataset "${DATASET}" \
  --output-dir Results/Diagnostics/RealBBoxCrop \
  --samples 10 \
  --margin-ratio "${REAL_BBOX_MARGIN_RATIO}"

"${PYTHON_BIN}" scripts/diagnose_real_scan_augmentation.py \
  --dataset "${DATASET}" \
  --output-dir Results/Diagnostics/RealScanAugmentation \
  --samples 10

echo "============================================================"
echo "Real ArabicDataset fine-tune"
echo "dataset              = ${DATASET}"
echo "mode                 = FINE-TUNE (never train from zero)"
echo "initial weights      = ${PRETRAINED_WEIGHTS}"
echo "approved pipeline     = XML 4-side crop -> grayscale -> scan augmentation -> DIRECT 1024x128 -> 1ch ResNet18"
echo "bbox crop             = ${REAL_BBOX_CROP} margin_ratio=${REAL_BBOX_MARGIN_RATIO} min_margin=${REAL_BBOX_MIN_MARGIN_PX}px"
echo "visual channels       = ${VISUAL_INPUT_CHANNELS} (true grayscale=${VISUAL_GRAYSCALE})"
echo "scan augmentation     = ${REAL_SCAN_AUGMENT} prob=${REAL_SCAN_AUGMENT_PROB}"
echo "augmentation copies   = ${REAL_TRAIN_AUGMENT_COPIES} views/source-line/epoch"
echo "blur prob/radius      = ${REAL_SCAN_BLUR_PROB} / ${REAL_SCAN_BLUR_RADIUS_MIN}..${REAL_SCAN_BLUR_RADIUS_MAX}"
echo "gaussian noise        = prob=${REAL_SCAN_GAUSSIAN_NOISE_PROB} std=${REAL_SCAN_GAUSSIAN_NOISE_STD_MIN}..${REAL_SCAN_GAUSSIAN_NOISE_STD_MAX}"
echo "speckle               = prob=${REAL_SCAN_SPECKLE_PROB} fraction=${REAL_SCAN_SPECKLE_FRACTION}"
echo "last partial batch    = DROP (${TRAIN_DROP_LAST})"
echo "AMP dtype             = ${AMP_DTYPE} (GradScaler off for bf16)"
echo "continuation change  = Strong SIGReg on pre-L2 fused valid windows"
echo "SIGReg lambda        = ${SIGREG_LAMBDA}"
echo "SIGReg sketch/knots  = ${SIGREG_SKETCH_DIM}/${SIGREG_NUM_KNOTS}"
echo "SIGReg t-range       = ${SIGREG_T_MIN}..${SIGREG_T_MAX}"
echo "SIGReg slice chunk   = ${SIGREG_SLICE_CHUNK}"
echo "backend              = resnet_token (ResNet18 -> TinyViT -> fused)"
echo "epochs               = ${RESTORATION_EPOCHS}"
echo "learning rate        = ${RESTORATION_LEARNING_RATE}"
echo "preprocessing        = XML four-sided crop -> TRUE GRAYSCALE -> direct 1024x128 fill"
echo "bbox margin          = ratio ${REAL_BBOX_MARGIN_RATIO}, min ${REAL_BBOX_MIN_MARGIN_PX}px"
echo "output geometry      = 1024x128 DIRECT RESIZE, no artificial white padding"
echo "all windows valid    = ${FULL_IMAGE_NO_PADDING} (63/63 participate)"
echo "crop source          = original.xml PartOfWord boxes; left/right/top/bottom"
echo "window sequence      = fixed 63-window grid; ALL 63 windows valid (no artificial canvas)"
echo "online augmentation  = scan-only: blur/noise/speckle/brightness/contrast"
echo "augmentation geom    = NO rotation, NO translation, NO scaling, NO warp"
echo "scan blur radius     = ${REAL_SCAN_BLUR_RADIUS_MIN}..${REAL_SCAN_BLUR_RADIUS_MAX}"
echo "scan gaussian std    = ${REAL_SCAN_GAUSSIAN_NOISE_STD_MIN}..${REAL_SCAN_GAUSSIAN_NOISE_STD_MAX}"
echo "training view        = ALL real page lines + own transcript (aligned or unaligned)"
echo "dataset source       = direct page-side scan; pair manifest is NOT the sample filter"
echo "duplicate copies     = deduplicated by source-page image fingerprint + line number"
echo "pair labels          = ignored; positive partner line NOT required"
echo "TinyViT dropout      = ${TINY_VIT_DROPOUT}"
echo "visual channels      = ${VISUAL_INPUT_CHANNELS} (grayscale=${VISUAL_GRAYSCALE})"
echo "bbox margin          = ratio ${REAL_BBOX_MARGIN_RATIO}, min ${REAL_BBOX_MIN_MARGIN_PX}px"
echo "scan augmentation    = enabled=${REAL_SCAN_AUGMENT}, prob=${REAL_SCAN_AUGMENT_PROB}"
echo "fusion dropout       = ${FUSION_DROPOUT}"
echo "DTW gamma            = ${POSITIVE_LETTER_DTW_GAMMA_START} -> ${POSITIVE_LETTER_DTW_GAMMA_END}"
echo "drop last batch      = ${TRAIN_DROP_LAST} (match synthetic gradient fix)"
echo "epoch snapshots       = every ${MODEL_WEIGHTS_EVERY_N_EPOCHS} epochs"
echo "sample eval epochs    = ${TRAIN_SAMPLE_EVAL_EPOCHS}"
echo "sample eval index     = ${TRAIN_SAMPLE_EVAL_START_INDEX}"
echo "sample eval methods   = NW fused image-image + hard letter-DTW fused image-text (RTL windows)"
echo "sample eval device    = ${TRAIN_SAMPLE_EVAL_DEVICE}"
echo "============================================================"

"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --dataset "${DATASET}" \
  --weights "${PRETRAINED_WEIGHTS}"
