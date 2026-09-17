#!/usr/bin/env bash
#SBATCH --job-name=point2_eval
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

cd "${PROJECT_DIR}"
mkdir -p out

[[ -x "${PYTHON_BIN}" ]] || {
  echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2
  exit 2
}

# Slurm should own GPU visibility inside the batch job. A stale
# CUDA_VISIBLE_DEVICES inherited from the submit shell can make CUDA count a
# device through NVML but fail runtime initialization. Record the incoming
# state, then prefer Slurm's explicit single-GPU mapping before Python imports
# torch.
echo "GPU environment before normalization:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
echo "  SLURM_STEP_GPUS=${SLURM_STEP_GPUS-<unset>}"
echo "  SLURM_GPU_INDEX=${SLURM_GPU_INDEX-<unset>}"

_slurm_gpu=""
for _candidate in "${SLURM_STEP_GPUS-}" "${SLURM_JOB_GPUS-}" "${SLURM_GPU_INDEX-}"; do
  if [[ "${_candidate}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    _slurm_gpu="${_candidate%%,*}"
    break
  fi
done
if [[ -n "${_slurm_gpu}" ]]; then
  export CUDA_VISIBLE_DEVICES="${_slurm_gpu}"
fi

echo "GPU environment after normalization:"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi -L:"
  nvidia-smi -L || true
  echo "nvidia-smi allocation view:"
  nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi is not available on this node"
fi

# Fail early with useful diagnostics before starting the eight evaluation runs.
"${PYTHON_BIN}" - <<'PY'
import os
import torch
print("CUDA preflight:")
print("  CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("  torch version        =", torch.__version__)
print("  torch CUDA version   =", torch.version.cuda)
print("  device_count         =", torch.cuda.device_count())
print("  is_available         =", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA preflight failed. If nvidia-smi above also failed, this is a node/driver allocation issue. "
        "If nvidia-smi worked, inspect CUDA_VISIBLE_DEVICES and the PyTorch CUDA runtime."
    )
print("  device 0             =", torch.cuda.get_device_name(0))
PY

export PROJECT_DIR
export PYTHON_BIN
export DEVICE="${DEVICE:-cuda}"
export N_SAMPLES="${N_SAMPLES:-100}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export START_INDEX="${START_INDEX:-1}"
export IMAGE_PREPROCESSING="${IMAGE_PREPROCESSING:-original}"
export MIN_ALIGNED_WINDOWS="${MIN_ALIGNED_WINDOWS:-5}"
export SCORE_MODE="${SCORE_MODE:-raw}"

bash scripts/eval_point2_both_architectures.sh
