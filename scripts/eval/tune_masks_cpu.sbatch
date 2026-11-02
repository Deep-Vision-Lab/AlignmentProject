#!/usr/bin/env bash
#SBATCH --job-name=mask_optuna_cpu
#SBATCH --partition=cpu
#SBATCH --gpus=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@120
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" ]] || {
  echo 'Submit this worker through scripts/eval/tune_masks_cpu.sh' >&2; exit 2;
}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'CPU tuner received visible GPUs; refusing an idle GPU reservation' >&2; exit 2
fi
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${MASK_CONDA_ENV:-manucripts_align}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1 WANDB_MODE=offline WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="${MASK_OUTPUT_DIR:?}/cpu_cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
echo "CPU mask study: job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-unknown} GPUs=0 threads=$OMP_NUM_THREADS"
exec python -u scripts/eval/cpu_resume_masks.py --output "$MASK_OUTPUT_DIR" --threads "$OMP_NUM_THREADS" "$@"
