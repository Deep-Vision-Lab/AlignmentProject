#!/bin/bash

# Script to clean directories and run training
# Reads parameters from Parameters.py

echo "====================================="
echo "Reading parameters from Parameters.py"
echo "====================================="

# Extract loss_type and model_arch from Parameters.py
LOSS_TYPE=$(python3 -c "import Parameters; print(Parameters.loss_type)")
MODEL_ARCH=$(python3 -c "import Parameters; print(Parameters.model_arch)")

echo "Loss Type: ${LOSS_TYPE}"
echo "Model Architecture: ${MODEL_ARCH}"
echo ""

echo "====================================="
echo "Cleaning training directories..."
echo "====================================="

# Run clean_dirs_train.sh
bash ./Clean/clean_dirs_train.sh "${LOSS_TYPE}" "${MODEL_ARCH}"

echo ""
echo "====================================="
echo "Cleaning weights..."
echo "====================================="

# Run clean_weights.sh
bash ./Clean/clean_weights.sh "${LOSS_TYPE}" "${MODEL_ARCH}"

echo ""
echo "Cleaning completed!"

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
