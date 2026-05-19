#!/bin/bash

# Script to clean directories and run training.
#
# Inputs can be passed either as positional args OR as env vars (env vars win
# when set, which is how sbatch --export=... feeds values in).
#
# Usage: ./run_train.sh [job_id] [data_dir] [pretrained_weights]
#
# Env vars (all optional):
#   JOB_ID              - identifier for this run (default: localRun)
#   DATA_DIR            - dataset root override
#   PRETRAINED_WEIGHTS  - path to a .pth file (weights-only load, fresh optimizer)
#   FINETUNE            - "1"/"true" -> pass --finetune to train.py
#                         (uses finetune_data_dir/lr/epochs from Parameters.py
#                          unless DATA_DIR is also set)
#   RESUME              - path to checkpoint .pth (restores model+optimizer+epoch)

JOB_ID=${JOB_ID:-${1:-localRun}}
DATA_DIR=${DATA_DIR:-${2:-}}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-${3:-}}
FINETUNE=${FINETUNE:-}
RESUME=${RESUME:-}

# Normalize FINETUNE truthiness
case "${FINETUNE,,}" in
    1|true|t|yes|y|on) FINETUNE_FLAG=1 ;;
    *)                 FINETUNE_FLAG=0 ;;
esac

echo "====================================="
echo "Parameters"
echo "====================================="
echo "Job ID:             ${JOB_ID}"
echo "Data Dir:           ${DATA_DIR:-<default>}"
echo "Pretrained Weights: ${PRETRAINED_WEIGHTS:-<none>}"
echo "Finetune:           ${FINETUNE_FLAG}"
echo "Resume:             ${RESUME:-<none>}"
echo ""

# Check if debug mode is enabled
DEBUG_MODE=$(python3 -c "import Parameters; print(Parameters.debug)")

if [ "${DEBUG_MODE}" = "True" ]; then
    echo "====================================="
    echo "Cleaning training directories..."
    echo "====================================="

    bash ./Clean/clean_dirs_train.sh "${JOB_ID}"

    # Never wipe weights when resuming — we need the checkpoint we're resuming from.
    if [ -n "${RESUME}" ]; then
        echo ""
        echo "Resume mode: skipping weights cleanup to preserve checkpoint."
    else
        echo ""
        echo "====================================="
        echo "Cleaning weights..."
        echo "====================================="
        bash ./Clean/clean_weights.sh "${JOB_ID}"
    fi

    echo ""
    echo "Cleaning completed!"
else
    echo "Debug mode is disabled, skipping cleaning steps."
fi

echo ""
echo "====================================="
echo "Starting training..."
echo "====================================="
echo ""

# Build optional extra args
EXTRA_ARGS=""
if [ -n "${DATA_DIR}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --data_dir \"${DATA_DIR}\""
fi
if [ -n "${PRETRAINED_WEIGHTS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --pretrained_weights \"${PRETRAINED_WEIGHTS}\""
fi
if [ "${FINETUNE_FLAG}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --finetune"
fi
if [ -n "${RESUME}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --resume \"${RESUME}\""
fi

eval python3 train.py --job_id "${JOB_ID}" ${EXTRA_ARGS}

echo ""
echo "====================================="
echo "Training completed!"
echo "====================================="
