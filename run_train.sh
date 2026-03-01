#!/bin/bash

# Script to clean directories and run training
# Usage: ./run_train.sh [job_id]
# If job_id is not provided, defaults to "localRun"

JOB_ID=${1:-localRun}


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

# Run the training script with job_id
python3 train.py --job_id "${JOB_ID}"

echo ""
echo "====================================="
echo "Training completed!"
echo "====================================="
