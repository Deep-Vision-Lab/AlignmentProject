#!/bin/bash

# Script to clean directories and run training
# Reads parameters from Parameters.py

echo "====================================="
echo "Reading parameters from Parameters.py"
echo "====================================="

# Extract loss_type and model_arch from Parameters.py
LOSS_TYPE=$(python3 -c "import Parameters; print(Parameters.loss_type)")

echo "Loss Type: ${LOSS_TYPE}"
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

# Run the training script
python3 train.py

echo ""
echo "====================================="
echo "Training completed!"
echo "====================================="
