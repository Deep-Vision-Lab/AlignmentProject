#!/bin/bash
# Check if loss_type argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <loss_type> <architecture>"
  exit 1
fi
LOSS_TYPE="$1"
ARCHITECTURE="$2"


read -p "Are you sure you want to delete all contents in these directories? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
  echo "Aborted."
  exit 1
fi


echo " "
echo "Cleaning the following directories under Weights/${LOSS_TYPE}/${ARCHITECTURE}:"


# Pattern to match files
PATTERN="model_epoch_*.pth"
# Find and delete matching files
find "Weights/${LOSS_TYPE}/${ARCHITECTURE}" -type f -name "${PATTERN}" -exec rm -v {} \;