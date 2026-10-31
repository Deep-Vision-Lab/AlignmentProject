#!/usr/bin/env bash
#SBATCH --job-name=mask_optuna
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=out/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" ]] || {
  echo 'Refusing to execute outside an allocated SLURM compute node. Use scripts/eval/tune_masks.sh.' >&2; exit 2;
}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
assigned_cuda="${CUDA_VISIBLE_DEVICES-}"
cuda_was_set="${CUDA_VISIBLE_DEVICES+x}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${MASK_CONDA_ENV:-manucripts_align}"
if [[ -n "$cuda_was_set" ]]; then export CUDA_VISIBLE_DEVICES="$assigned_cuda"; fi
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1 WANDB_MODE=offline WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# PyTorch's deterministic CUDA operations need this set before Python starts.
case "${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" in
  :4096:8|:16:8) export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" ;;
  *) echo 'CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA evaluation' >&2; exit 2 ;;
esac
if [[ -n "${MASK_OUTPUT_DIR:-}" ]]; then
  export MPLCONFIGDIR="$MASK_OUTPUT_DIR/cache/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
fi
echo "Job: $SLURM_JOB_ID / ${SLURM_JOB_NAME:-mask_optuna}; node: $(hostname)"
echo "Conda: $CONDA_DEFAULT_ENV; Python: $(command -v python); GPUs: ${CUDA_VISIBLE_DEVICES:-scheduler-managed}"
echo "CuBLAS deterministic workspace: $CUBLAS_WORKSPACE_CONFIG"
echo "Partition: ${SLURM_JOB_PARTITION:-site-default}; account: ${SLURM_JOB_ACCOUNT:-site-default}; QoS: ${SLURM_JOB_QOS:-site-default}"
exec python -u scripts/eval/mask_batch.py "$@"
