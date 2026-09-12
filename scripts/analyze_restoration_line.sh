#!/usr/bin/env bash
# Inspect every internal window representation for one restoration-model line.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS="${WEIGHTS:-${PROJECT_DIR}/Weights/vit_restore_dtw_s16/model_best.pth}"
DATASET="${DATASET:-${PROJECT_DIR}/DataSet/Synthetic63}"
INDEX="${INDEX:-132}"
SIDE="${SIDE:-1}"
DEVICE="${DEVICE:-cuda}"
IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
TAG="${TAG:-line${INDEX}_side${SIDE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/Results/Diagnostics/restoration_windows/${TAG}}"

cd "${PROJECT_DIR}"
export MPLBACKEND=Agg

"${PYTHON_BIN}" -m py_compile Evaluation/analyze_restoration_line.py
echo "Diagnostic syntax preflight: OK"

echo "CUDA environment:"
echo "  host=$(hostname)"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<none>}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ "${DEVICE}" == cuda* ]]; then
  if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print("  torch.cuda.is_available() =", torch.cuda.is_available())
print("  torch.cuda.device_count() =", torch.cuda.device_count())
if ok:
    print("  cuda:0 =", torch.cuda.get_device_name(0))
sys.exit(0 if ok else 1)
PY
  then
    echo "ERROR: DEVICE=${DEVICE} requested, but this shell has no usable CUDA device." >&2
    echo "Run inside an active SLURM GPU allocation, or use DEVICE=cpu." >&2
    exit 3
  fi
fi

echo "============================================================"
echo "Restoration line feature diagnostic"
echo "weights       = ${WEIGHTS}"
echo "dataset       = ${DATASET}"
echo "index         = ${INDEX}"
echo "side          = ${SIDE}"
echo "device        = ${DEVICE}"
echo "preprocessing = ${IMAGE_PREPROCESSING}"
echo "output        = ${OUTPUT_DIR}"
echo "============================================================"

exec "${PYTHON_BIN}" -u -m Evaluation.analyze_restoration_line \
  --weights "${WEIGHTS}" \
  --dataset "${DATASET}" \
  --index "${INDEX}" \
  --side "${SIDE}" \
  --device "${DEVICE}" \
  --image-preprocessing "${IMAGE_PREPROCESSING}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
