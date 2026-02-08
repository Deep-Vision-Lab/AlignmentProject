#!/bin/bash

# Script to clean directories and run training
# Usage: ./run_train.sh <loss_type> <job_id>

# Check if both parameters are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: loss_type and job_id are required"
    echo "Usage: ./run_train.sh <loss_type> <job_id>"
    exit 1
fi

LOSS_TYPE=$1
JOB_ID=$2

echo "====================================="
echo "Parameters"
echo "====================================="
echo "Loss Type: ${LOSS_TYPE}"
echo "Job ID: ${JOB_ID}"
echo ""

# Check if debug mode is enabled
DEBUG_MODE=$(python3 -c "import Parameters; print(Parameters.debug)")

if [ "${DEBUG_MODE}" = "True" ]; then
    echo "====================================="
    echo "Cleaning training directories..."
    echo "====================================="

    # Run clean_dirs_train.sh
    bash ./Clean/clean_dirs_train.sh "${LOSS_TYPE}"

    echo ""
    echo "====================================="
    echo "Cleaning weights..."
    echo "====================================="

    # Run clean_weights.sh
    bash ./Clean/clean_weights.sh "${LOSS_TYPE}"

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

# Run the training script with job_id
python3 train.py --job_id "${JOB_ID}"

echo ""
echo "====================================="
echo "Training completed!"
echo "====================================="
