#!/bin/bash
# Check if JOB_ID argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <JOB_ID>"
  exit 1
fi
JOB_ID="$1"

echo " "
echo "Cleaning the following directories under Weights/${JOB_ID}:"


# Pattern to match files
PATTERN="model_epoch_*.pth"
# Find and delete matching files
find "Weights/${JOB_ID}" -type f -name "${PATTERN}" -exec rm -v {} \;