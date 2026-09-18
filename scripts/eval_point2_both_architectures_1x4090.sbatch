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

# IMPORTANT: Do not remap CUDA_VISIBLE_DEVICES from SLURM_JOB_GPUS or
# SLURM_STEP_GPUS on this cluster.
#
# Slurm/cgroup GPU isolation already exposes the allocated physical GPU as a
# job-local logical device, normally CUDA device 0.  For example, a job may
# report SLURM_JOB_GPUS=2 while CUDA_VISIBLE_DEVICES=0 and nvidia-smi -L shows
# exactly one GPU as GPU 0.  Replacing CUDA_VISIBLE_DEVICES with the physical or
# step index would hide the only device from CUDA.
#
# Therefore preserve the CUDA_VISIBLE_DEVICES value Slurm installed for the
# batch job exactly as-is.
echo "GPU environment (preserved from Slurm):"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
echo "  SLURM_STEP_GPUS=${SLURM_STEP_GPUS-<unset>}"
echo "  SLURM_GPU_INDEX=${SLURM_GPU_INDEX-<unset>}"

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
        "CUDA preflight failed. Slurm GPU visibility was preserved exactly. "
        "If nvidia-smi above sees the allocated GPU but torch still cannot initialize CUDA, "
        "this is a node/runtime issue rather than a remapping issue."
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
