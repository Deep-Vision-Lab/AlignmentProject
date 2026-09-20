#!/usr/bin/env bash
#SBATCH --job-name=point2_fused20
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/manucripts_align/bin/python}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/ArabicDataset}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_point2/model_latest.pth}"
N_SAMPLES="${N_SAMPLES:-20}"
START_INDEX="${START_INDEX:-1}"
THRESHOLD="${THRESHOLD:-0.10}"
CROP_MODE="${CROP_MODE:-vertical_borders}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-training}"
SIDE_PADDING_PX="${SIDE_PADDING_PX:-144}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/Point2/fused_visual_20_${TAG}}"
EPOCH_SERIES="${EPOCH_SERIES:-0}"
EPOCH_LIST="${EPOCH_LIST:-1,5,10,15,20,25,30}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$(dirname "${WEIGHTS}")}"
SERIES_OUTPUT_DIR="${SERIES_OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/Point2/epoch_series_${TAG}}"

cd "${PROJECT_DIR}"
mkdir -p out

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 2; }
if [[ "${EPOCH_SERIES}" == "1" ]]; then
  [[ -d "${WEIGHTS_DIR}" ]] || { echo "ERROR: weights directory not found: ${WEIGHTS_DIR}" >&2; exit 2; }
else
  [[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }
fi
[[ -d "${DATASET}" || -f "${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }

# Preserve Slurm's CUDA_VISIBLE_DEVICES exactly. The cluster exposes the
# allocated physical RTX 4090 as job-local cuda:0.
echo "GPU environment:"
echo "  host=$(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
nvidia-smi -L || true

set +e
"${PYTHON_BIN}" - <<'PY'
import os
import sys
import traceback
import torch

print("CUDA preflight:")
print("  host                   =", os.uname().nodename)
print("  CUDA_VISIBLE_DEVICES   =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("  NVIDIA_VISIBLE_DEVICES =", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("  torch version          =", torch.__version__)
print("  torch CUDA version     =", torch.version.cuda)

try:
    count = torch.cuda.device_count()
    print("  device_count           =", count)
    print("  is_available           =", torch.cuda.is_available())
    torch.cuda.init()
    probe = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    print("  device 0               =", torch.cuda.get_device_name(0))
    print("  allocation probe       = OK", tuple(probe.shape))
except Exception as exc:
    print("  CUDA exception type    =", type(exc).__name__, file=sys.stderr)
    print("  CUDA exception         =", repr(exc), file=sys.stderr)
    traceback.print_exc()
    sys.exit(86)
PY
CUDA_RC=$?
set -e

if [[ "${CUDA_RC}" -ne 0 ]]; then
  BAD_NODE="${SLURMD_NODENAME:-$(hostname -s)}"
  echo >&2
  echo "ERROR: CUDA runtime initialization failed on node ${BAD_NODE}." >&2
  echo "The job preserved Slurm GPU visibility; failure occurred before model loading." >&2
  echo >&2
  echo "Device-file diagnostics:" >&2
  ls -l /dev/nvidia* 2>&1 || true
  echo >&2
  echo "Recommended resubmission excluding this node:" >&2
  echo "  sbatch --exclude=${BAD_NODE} --export=ALL,DATASET=${DATASET},WEIGHTS=${WEIGHTS},N_SAMPLES=${N_SAMPLES},START_INDEX=${START_INDEX},THRESHOLD=${THRESHOLD},CROP_MODE=${CROP_MODE},IMAGE_PREPROCESSING=${IMAGE_PREPROCESSING},OUTPUT_DIR=${OUTPUT_DIR} scripts/eval_point2_fused_visual_20_1x4090.sbatch" >&2
  exit "${CUDA_RC}"
fi

# Evaluation must match the deterministic part of the real training pipeline:
# XML PartOfWord bbox crop on all four sides -> grayscale -> aspect-preserving
# 1024x128 canvas -> 1-channel ResNet18.  Stochastic scan augmentation is
# deliberately disabled so the same held-out sample is identical at every epoch.
export EVAL_SIDE_PADDING_PX="${SIDE_PADDING_PX}"
export ZERO_SHOT_CROP_MODE="${CROP_MODE}"
export DATASET_TYPE=real
export LINE_HEIGHT=128
export LINE_WIDTH=1024
export PACK_VALID_WINDOWS=0
export REAL_SYNTHETIC_STYLE=0
export REAL_BINARIZE=0
export REAL_BINARIZE_AUTO_INVERT=0
export REAL_BINARIZE_AUTOCONTRAST=0
export REAL_AUGMENT=0
export REAL_SCAN_AUGMENT=0
export VISUAL_INPUT_CHANNELS=1
export VISUAL_GRAYSCALE=1
export REAL_GRAYSCALE=1
export REAL_BBOX_CROP=1
export REAL_BBOX_CROP_STRICT=1
export REAL_BBOX_MARGIN_RATIO="${REAL_BBOX_MARGIN_RATIO:-0.05}"
export REAL_BBOX_MIN_MARGIN_PX="${REAL_BBOX_MIN_MARGIN_PX:-2}"
export ZERO_SHOT_PREPROCESS=1
export ZERO_SHOT_FOREGROUND_CROP=0
export ZERO_SHOT_PRESERVE_ASPECT=1
export ZERO_SHOT_SOURCE_GEOMETRY=0
export ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-0.72}"
export LINE_GEOMETRY_MODE=xml-bbox-gray-aspect-preserving
export EVAL_COMPACT_OUTPUT=1

echo "============================================================"
echo "Point-2 fused-only visual inspection"
echo "dataset           = ${DATASET}"
echo "samples           = ${N_SAMPLES}"
echo "start_index       = ${START_INDEX}"
echo "representation    = fused"
echo "preprocessing     = ${IMAGE_PREPROCESSING}"
echo "input pipeline    = XML 4-side bbox -> grayscale -> 1024x128 aspect-preserved"
echo "visual channels   = ${VISUAL_INPUT_CHANNELS}"
echo "bbox crop         = ${REAL_BBOX_CROP} margin_ratio=${REAL_BBOX_MARGIN_RATIO} min=${REAL_BBOX_MIN_MARGIN_PX}px"
echo "scan augmentation = ${REAL_SCAN_AUGMENT} (must stay 0 for deterministic eval)"
echo "geometry mode     = ${LINE_GEOMETRY_MODE}"
echo "side padding px   = ${SIDE_PADDING_PX}"
echo "crop mode         = ${ZERO_SHOT_CROP_MODE}"
echo "cosine threshold  = ${THRESHOLD}"
echo "epoch series      = ${EPOCH_SERIES}"
if [[ "${EPOCH_SERIES}" == "1" ]]; then
  echo "weights dir       = ${WEIGHTS_DIR}"
  echo "epochs            = ${EPOCH_LIST}"
  echo "series output     = ${SERIES_OUTPUT_DIR}"
else
  echo "weights           = ${WEIGHTS}"
  echo "output            = ${OUTPUT_DIR}"
fi
echo "============================================================"

run_one_eval() {
  local weights_path="$1"
  local output_path="$2"
  local epoch_label="$3"

  [[ -f "${weights_path}" ]] || {
    echo "ERROR: checkpoint missing for ${epoch_label}: ${weights_path}" >&2
    return 2
  }
  if [[ -d "${output_path}" ]] && [[ -n "$(find "${output_path}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: output is not empty for ${epoch_label}: ${output_path}" >&2
    return 2
  fi

  echo
  echo ">>> Point-2 evaluation ${epoch_label}"
  echo "    weights = ${weights_path}"
  echo "    output  = ${output_path}"

  "${PYTHON_BIN}" -u -m Evaluation.eval_point2 \
    --point2-representation fused \
    --dataset "${DATASET}" \
    --weights "${weights_path}" \
    --branch restoration \
    --alignment-unit window \
    --word-support-floor 0.0 \
    --min-aligned-windows 5 \
    --image-preprocessing "${IMAGE_PREPROCESSING}" \
    --split test \
    --training-samples 6000 \
    --split-seed 42 \
    --n-samples "${N_SAMPLES}" \
    --start-index "${START_INDEX}" \
    --device cuda \
    --score-mode raw \
    --threshold "${THRESHOLD}" \
    --gap -0.30 \
    --output-dir "${output_path}"
}

if [[ "${EPOCH_SERIES}" == "1" ]]; then
  mkdir -p "${SERIES_OUTPUT_DIR}"
  IFS=',' read -ra EPOCH_VALUES <<< "${EPOCH_LIST}"
  for raw_epoch in "${EPOCH_VALUES[@]}"; do
    epoch="$((10#${raw_epoch}))"
    epoch_pad="$(printf '%03d' "${epoch}")"
    weights_path="${WEIGHTS_DIR}/model_epoch_${epoch_pad}.pth"
    output_path="${SERIES_OUTPUT_DIR}/epoch_${epoch_pad}"
    run_one_eval "${weights_path}" "${output_path}" "epoch=${epoch}"
  done
  echo
  echo "Finished longitudinal Point-2 evaluation:"
  echo "  ${SERIES_OUTPUT_DIR}"
  echo "Compare epoch_001, epoch_005, epoch_010, ... / pair_XXXXX/cosine_similarity_values.png"
else
  run_one_eval "${WEIGHTS}" "${OUTPUT_DIR}" "single checkpoint"
  echo
  echo "Finished fused-only visual set:"
  echo "  ${OUTPUT_DIR}"
  echo "Inspect each pair_XXXXX/cosine_similarity_values.png"
  echo "and pair_XXXXX/line1_pred_mask.png + line2_pred_mask.png"
fi
