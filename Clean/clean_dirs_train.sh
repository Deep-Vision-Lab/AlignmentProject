#!/bin/bash

# Check if loss_type argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <loss_type>"
  exit 1
fi

LOSS_TYPE="$1"

# List of directories to clean (relative to Results/{LOSS_TYPE}/)
DIRS_TO_CLEAN=(
  "InputImages"
  "SimilarityMatricesPerEpoch"
)

echo "Cleaning the following directories under Results/${LOSS_TYPE}:"
for dir in "${DIRS_TO_CLEAN[@]}"; do
  echo " - TrainResults/${LOSS_TYPE}/${dir}"
done

for dir in "${DIRS_TO_CLEAN[@]}"; do
  full_path="TrainResults/${LOSS_TYPE}/${dir}"
  if [ -d "$full_path" ]; then
    echo "Cleaning $full_path..."
    rm -rf "$full_path"/*
    rm -rf "$full_path"/.??* 2>/dev/null  # Clean hidden files
  else
    echo "Warning: $full_path does not exist."
  fi
done

echo "Cleanup completed."