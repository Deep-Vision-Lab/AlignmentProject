#!/usr/bin/env bash

#SBATCH --job-name=alignment_train
#SBATCH --partition=rtx4090
#SBATCH --gpus=rtx_4090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03-00:00:00
#SBATCH --output=out/%x_%J.out
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

RUN_NAME="${SLURM_JOB_NAME}"
FUSION_MODE="${FUSION_MODE:-concat}"
USE_GATED_FUSION="${USE_GATED_FUSION:-0}"
GPUS="${GPUS:-2}"

echo "Job name: $SLURM_JOB_NAME"
echo "Run name: $RUN_NAME"
echo "Fusion mode: $FUSION_MODE"
echo "Gated fusion: $USE_GATED_FUSION"

# A preempted SLURM job restarts this script with the same arguments. Resume
# the last complete (atomically written) epoch instead of attempting a fresh
# run in the existing output directory.
TRAIN_ARGS=("$@")
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
if [[ ! "$RESTART_COUNT" =~ ^[0-9]+$ ]]; then
    echo "Invalid SLURM_RESTART_COUNT: $RESTART_COUNT" >&2
    exit 2
fi
echo "SLURM restart count: $RESTART_COUNT"
if (( RESTART_COUNT > 0 )); then
    HAS_RESUME=0
    HAS_FINETUNE=0
    for ARG in "${TRAIN_ARGS[@]}"; do
        case "${ARG%%=*}" in
            --resume) HAS_RESUME=1 ;;
            --finetune) HAS_FINETUNE=1 ;;
        esac
    done
    if (( ! HAS_RESUME )); then
        if (( HAS_FINETUNE )); then
            echo "Preempted fine-tuning requires an explicit resume command; refusing to restart from source weights." >&2
            exit 2
        fi
        LATEST="Weights/$RUN_NAME/checkpoint_latest.pt"
        if [[ -s "$LATEST" ]]; then
            TRAIN_ARGS+=(--resume "$LATEST")
            echo "SLURM restart: resuming $LATEST"
        else
            TRAIN_ARGS+=(--restart-after-preemption)
            echo "SLURM restart: no complete checkpoint; retrying the first epoch"
        fi
    fi
fi

# --------------------------------------------------
# Preserve the GPU assignment made by SLURM
# --------------------------------------------------

# SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# echo "CUDA_VISIBLE_DEVICES before conda: ${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-NOT_SET}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-NOT_SET}"

# --------------------------------------------------
# Conda
# --------------------------------------------------

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate manucripts_align

# # Conda must NOT override SLURM's GPU assignment.
# if [[ -n "$SLURM_CUDA_VISIBLE_DEVICES" ]]; then
#     export CUDA_VISIBLE_DEVICES="$SLURM_CUDA_VISIBLE_DEVICES"
# fi

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
# echo "CUDA_VISIBLE_DEVICES after conda: ${CUDA_VISIBLE_DEVICES:-NOT_SET}"

# --------------------------------------------------
# GPU diagnostic
# --------------------------------------------------

echo "========================================"
echo "GPU PRE-FLIGHT"
echo "========================================"

nvidia-smi -L

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"Logical GPU {i}: {torch.cuda.get_device_name(i)}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable before torchrun")

expected = int(os.environ.get('GPUS', '2'))
if torch.cuda.device_count() != expected:
    raise RuntimeError(
        f"Expected exactly {expected} visible GPUs, got {torch.cuda.device_count()}"
    )
PY

echo "GPU pre-flight: PASSED"

# --------------------------------------------------
# Training
# --------------------------------------------------

exec python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$GPUS" \
    train.py \
    --run-name "$RUN_NAME" \
    --fusion-mode "$FUSION_MODE" \
    --use-gated-fusion "$USE_GATED_FUSION" \
    "${TRAIN_ARGS[@]}"
