#!/usr/bin/env bash
#SBATCH --job-name=gen_three_font_cpu
#SBATCH --partition=jelsana
#SBATCH --qos=jelsana
#SBATCH --account=jelsana
#SBATCH --time=01-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=out/%x_%J.out
#SBATCH --error=out/%x_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
OUTPUT_DIR="${PROJECT_DIR}/DataSet/Synthetic_Arabic_Three_Font_Augmented"

mkdir -p "${PROJECT_DIR}/out"
cd "${PROJECT_DIR}"

printf '%s\n' \
  "Started:       $(date)" \
  "Job ID:        ${SLURM_JOB_ID:-not-set}" \
  "Node:          ${SLURM_JOB_NODELIST:-not-set}" \
  "Project:       ${PROJECT_DIR}" \
  "Conda env:     ${CONDA_ENV}" \
  "CPU cores:     ${SLURM_CPUS_PER_TASK:-4}" \
  "Output:        ${OUTPUT_DIR}"

# This is a CPU-only generation job. No GPU is requested or exposed.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found after loading the Anaconda module." >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: output directory is not empty: ${OUTPUT_DIR}" >&2
  echo "Remove it, rename it, or submit with a different output directory before retrying." >&2
  exit 2
fi

python generateDataArabicThreeFontsCompatible.py \
  --font-dir Fonts \
  --font-count 3 \
  --samples-per-font 3000 \
  --output-dir DataSet/Synthetic_Arabic_Three_Font_Augmented \
  --original-ratio 0.15 \
  --cross-injection-ratio 0.50 \
  --aligned-unaligned-ratio 0.35 \
  --mixed-font-injection-prob 0.50 \
  --segment-gap-min 2 \
  --segment-gap-max 6 \
  --target-fill-ratio 0.96

printf '%s\n' \
  "Finished:      $(date)" \
  "Dataset:       ${OUTPUT_DIR}"
