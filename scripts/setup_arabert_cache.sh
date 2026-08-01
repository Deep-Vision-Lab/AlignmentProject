#!/usr/bin/env bash
# One-time setup for the offline AraBERT cache used by training and evaluation.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.hf_cache}"

export ARABIC_TEXT_MODEL_NAME HF_HOME
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -m pip install --upgrade \
  "huggingface-hub>=0.23.2,<1.0" \
  "tokenizers>=0.19,<0.20" \
  "safetensors>=0.4.1" \
  "transformers==4.44.2"

python - <<'PY'
import os
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

model = os.environ["ARABIC_TEXT_MODEL_NAME"]
cache = os.environ["HF_HOME"]
print(f"torch={torch.__version__} transformers={transformers.__version__}")
AutoTokenizer.from_pretrained(model, cache_dir=cache)
AutoModel.from_pretrained(model, cache_dir=cache)
print(f"Cached {model} in {cache}")
PY

echo "Environment '${CONDA_ENV}' is ready; AraBERT cache: ${HF_HOME}"
