#!/bin/bash

# Script to clean directories and run training
# Usage: ./run_train.sh [job_id] [data_dir] [pretrained_weights]
#   job_id            - identifier for this run (default: localRun)
#   data_dir          - path to a different dataset root for finetuning (optional)
#   pretrained_weights - path to a .pth file to load before training (optional)

JOB_ID=${1:-localRun}
DATA_DIR=${2:-}
PRETRAINED_WEIGHTS=${3:-}


echo "====================================="
echo "Parameters"
echo "====================================="
echo "Job ID: ${JOB_ID}"
echo ""

# Check if debug mode is enabled
DEBUG_MODE=$(python3 -c "import Parameters; print(Parameters.debug)")

if [ "${DEBUG_MODE}" = "True" ]; then
    echo "====================================="
    echo "Cleaning training directories..."
    echo "====================================="

    # Run clean_dirs_train.sh
    bash ./Clean/clean_dirs_train.sh "${JOB_ID}"

    echo ""
    echo "====================================="
    echo "Cleaning weights..."
    echo "====================================="

    # Run clean_weights.sh
    bash ./Clean/clean_weights.sh "${JOB_ID}"

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

# Run the training script
eval python3 train.py --job_id "${JOB_ID}" ${EXTRA_ARGS}

echo ""
echo "====================================="
echo "Training completed!"
echo "====================================="
