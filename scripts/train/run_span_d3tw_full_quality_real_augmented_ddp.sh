#!/usr/bin/env bash
# Submit the multi-GPU DDP full-quality real-data augmentation job.
#
# Default: two RTX 4090 GPUs with BATCH_SIZE=8 per GPU (global batch 16).
# Override NUM_GPUS, GPU_RESOURCE, PARTITION, BATCH_SIZE, and other training
# settings through environment variables.
#
# Examples:
#   bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
#   NUM_GPUS=4 BATCH_SIZE=4 bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh"
  echo "Override settings through environment variables, not command-line flags."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_DIR}"
mkdir -p out logs

NUM_GPUS="${NUM_GPUS:-2}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
PARTITION="${PARTITION:-rtx4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-03-00:00:00}"

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got ${NUM_GPUS}" >&2
  exit 2
fi

# Resolve the frozen Arabic backbone before submitting the job. This searches the
# current checkout, the sibling AlignmentProject_clone checkout, the user's
# standard Hugging Face cache, and any explicitly supplied HF_HOME.
export ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME:-aubmindlab/bert-base-arabertv02}"
export HF_HOME="$(
  bash "${SCRIPT_DIR}/resolve_hf_cache_home.sh" \
    "${PROJECT_DIR}" \
    "${ARABIC_TEXT_MODEL_NAME}"
)"

# The corrected foreground mask measures contrast from the local background and
# is therefore valid for both black-on-white and white-on-black line images.
export NUM_GPUS
export INK_CONTRAST_THRESHOLD="${INK_CONTRAST_THRESHOLD:-0.15}"

GLOBAL_BATCH_NOTE="BATCH_SIZE is per GPU; global batch = BATCH_SIZE x ${NUM_GPUS}."
echo "Submitting full-quality DDP training"
echo "  project            = ${PROJECT_DIR}"
echo "  partition          = ${PARTITION}"
echo "  GPU request        = ${GPU_RESOURCE}:${NUM_GPUS}"
echo "  CPUs               = ${CPUS_PER_TASK}"
echo "  memory             = ${MEMORY}"
echo "  time limit         = ${TIME_LIMIT}"
echo "  text model         = ${ARABIC_TEXT_MODEL_NAME}"
echo "  Hugging Face cache = ${HF_HOME}"
echo "  ink threshold      = ${INK_CONTRAST_THRESHOLD}"
echo "  ${GLOBAL_BATCH_NOTE}"

sbatch \
  --partition="${PARTITION}" \
  --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",NUM_GPUS="${NUM_GPUS}",HF_HOME="${HF_HOME}",ARABIC_TEXT_MODEL_NAME="${ARABIC_TEXT_MODEL_NAME}",INK_CONTRAST_THRESHOLD="${INK_CONTRAST_THRESHOLD}" \
  "${SCRIPT_DIR}/sbatch_span_d3tw_full_quality_real_augmented_ddp.sbatch"
