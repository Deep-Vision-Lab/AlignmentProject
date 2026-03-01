#!/bin/bash

# Check if JOB_ID argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <JOB_ID>"
  exit 1
fi

JOB_ID="$1"

# List of directories to clean (relative to Results/{JOB_ID}/)
DIRS_TO_CLEAN=(
  "InputImages"
  "SimilarityMatricesPerEpoch"
)

echo "Cleaning the following directories under Results/${JOB_ID}:"
for dir in "${DIRS_TO_CLEAN[@]}"; do
  echo " - TrainResults/${JOB_ID}/${dir}"
done

for dir in "${DIRS_TO_CLEAN[@]}"; do
  full_path="TrainResults/${JOB_ID}/${dir}"
  if [ -d "$full_path" ]; then
    echo "Cleaning $full_path..."
    rm -rf "$full_path"/*
    rm -rf "$full_path"/.??* 2>/dev/null  # Clean hidden files
  else
    echo "Warning: $full_path does not exist."
  fi
done

echo "Cleanup completed."