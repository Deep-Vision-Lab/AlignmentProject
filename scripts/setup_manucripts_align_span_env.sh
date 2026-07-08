#!/usr/bin/env bash
#
# One-time setup for the offline span-D3TW sbatch job.
# Run this on a login node with network access before submitting the offline job:
#   bash scripts/setup_manucripts_align_span_env.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -m pip install transformers

python -c "from transformers import AutoModel, AutoTokenizer; import os; model=os.environ['ARABIC_TEXT_MODEL_NAME']; AutoTokenizer.from_pretrained(model); AutoModel.from_pretrained(model); print(f'Cached {model}')"

echo "Environment '${CONDA_ENV}' is ready for offline span-D3TW jobs."
