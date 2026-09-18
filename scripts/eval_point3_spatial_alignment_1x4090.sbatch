#!/usr/bin/env bash
#SBATCH --job-name=point3_spatial
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
N_SAMPLES="${N_SAMPLES:-100}"
START_INDEX="${START_INDEX:-1}"
MIN_ALIGNED_WINDOWS="${MIN_ALIGNED_WINDOWS:-5}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT="${ROOT_OUT:-${PROJECT_DIR}/Results/Evaluation/Point3/spatial_${TAG}}"
EVAL_OUT="${ROOT_OUT}/fused"
METRICS_OUT="${ROOT_OUT}/spatial_metrics"

cd "${PROJECT_DIR}"
mkdir -p out "${ROOT_OUT}"

echo "============================================================"
echo "Point 3: spatial alignment validation"
echo "architecture       = old ResNet18 -> TinyViT -> Fusion"
echo "weights            = ${WEIGHTS}"
echo "dataset            = ${DATASET}"
echo "samples            = ${N_SAMPLES}"
echo "physical window    = 128x32"
echo "stride             = 16"
echo "min word support   = ${MIN_ALIGNED_WINDOWS} consecutive windows"
echo "output             = ${ROOT_OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
echo "============================================================"

"${PYTHON_BIN}" -m py_compile \
  Evaluation/eval_point2.py \
  Evaluation/point2_runtime.py \
  Evaluation/point3_spatial_metrics.py

"${PYTHON_BIN}" - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

[[ -f "${WEIGHTS}" ]] || { echo "Missing checkpoint: ${WEIGHTS}" >&2; exit 2; }

"${PYTHON_BIN}" -u -m Evaluation.eval_point2 \
  --point2-representation fused \
  --dataset "${DATASET}" \
  --weights "${WEIGHTS}" \
  --branch restoration \
  --alignment-unit window \
  --word-support-floor 0.0 \
  --min-aligned-windows "${MIN_ALIGNED_WINDOWS}" \
  --image-preprocessing original \
  --split test \
  --training-samples 6000 \
  --split-seed 42 \
  --n-samples "${N_SAMPLES}" \
  --start-index "${START_INDEX}" \
  --device cuda \
  --score-mode raw \
  --threshold 0.0 \
  --gap -0.30 \
  --output-dir "${EVAL_OUT}"

"${PYTHON_BIN}" -u Evaluation/point3_spatial_metrics.py \
  --eval-root "${EVAL_OUT}" \
  --output-dir "${METRICS_OUT}" \
  --min-consecutive-windows "${MIN_ALIGNED_WINDOWS}"

echo
echo "Point-3 outputs:"
echo "  ${EVAL_OUT}/summary.json"
echo "  ${METRICS_OUT}/point3_pair_metrics.csv"
echo "  ${METRICS_OUT}/point3_summary.json"
