#!/bin/bash

# Check if loss_type argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <loss_type>"
  exit 1
fi

LOSS_TYPE="$1"

# List of directories to clean (relative to Results/{LOSS_TYPE}/)
DIRS_TO_CLEAN=(
  "Elements_Vectors"
  "Vectors_similarity"
  "AlignedTwoimages"
  "Lines_plots"
  "Matrices_plots"
  "Vectors_plots"
  "Vectors_similarity"
  "VectorsSub_plots"
)

echo "Cleaning the following directories under Results:"
for dir in "${DIRS_TO_CLEAN[@]}"; do
  echo " - Results/${dir}"
done

read -p "Are you sure you want to delete all contents in these directories? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
  echo "Aborted."
  exit 1
fi

for dir in "${DIRS_TO_CLEAN[@]}"; do
  full_path="Results/${dir}"
  if [ -d "$full_path" ]; then
    echo "Cleaning $full_path..."
    rm -rf "$full_path"/*
    rm -rf "$full_path"/.??* 2>/dev/null  # Clean hidden files
  else
    echo "Warning: $full_path does not exist."
  fi
done

echo "Cleanup completed."