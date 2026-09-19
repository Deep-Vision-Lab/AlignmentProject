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
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/res18_tinyvit_point2/model_latest.pth}"
N_SAMPLES="${N_SAMPLES:-20}"
START_INDEX="${START_INDEX:-1}"
THRESHOLD="${THRESHOLD:-0.10}"
CROP_MODE="${CROP_MODE:-vertical_borders}"
SIDE_PADDING_PX="${SIDE_PADDING_PX:-144}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Evaluation/Point2/fused_visual_20_${TAG}}"

cd "${PROJECT_DIR}"
mkdir -p out

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }
[[ -d "${DATASET}" || -f "${DATASET}" ]] || { echo "ERROR: dataset not found: ${DATASET}" >&2; exit 2; }

# Preserve Slurm's CUDA_VISIBLE_DEVICES exactly. The cluster exposes the
# allocated physical RTX 4090 as job-local cuda:0.
echo "GPU environment:"
echo "  host=$(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
nvidia-smi -L || true

"${PYTHON_BIN}" - <<'PY'
import torch
print("CUDA preflight:")
print("  available =", torch.cuda.is_available())
print("  count     =", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
torch.cuda.init()
x = torch.empty(1, device="cuda:0")
torch.cuda.synchronize()
print("  device    =", torch.cuda.get_device_name(0))
print("  probe     = OK")
PY

export EVAL_SIDE_PADDING_PX="${SIDE_PADDING_PX}"
export ZERO_SHOT_CROP_MODE="${CROP_MODE}"

echo "============================================================"
echo "Point-2 fused-only visual inspection"
echo "weights           = ${WEIGHTS}"
echo "dataset           = ${DATASET}"
echo "samples           = ${N_SAMPLES}"
echo "start_index       = ${START_INDEX}"
echo "representation    = fused"
echo "preprocessing     = wide_side_padding"
echo "side padding px   = ${SIDE_PADDING_PX}"
echo "crop mode         = ${ZERO_SHOT_CROP_MODE}"
echo "cosine threshold  = ${THRESHOLD}"
echo "output            = ${OUTPUT_DIR}"
echo "============================================================"

"${PYTHON_BIN}" -u -m Evaluation.eval_point2 \
  --point2-representation fused \
  --dataset "${DATASET}" \
  --weights "${WEIGHTS}" \
  --branch restoration \
  --alignment-unit window \
  --word-support-floor 0.0 \
  --min-aligned-windows 5 \
  --image-preprocessing wide_side_padding \
  --split test \
  --training-samples 6000 \
  --split-seed 42 \
  --n-samples "${N_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --device cuda \
  --score-mode raw \
  --threshold "${THRESHOLD}" \
  --gap -0.30 \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Finished fused-only visual set:"
echo "  ${OUTPUT_DIR}"
echo "Inspect each pair_XXXXX/cosine_similarity_values.png"
echo "and pair_XXXXX/line1_pred_mask.png + line2_pred_mask.png"
