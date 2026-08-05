#!/usr/bin/env bash
#SBATCH --job-name=gen_three_font_cpu
#SBATCH --partition=jelsana
#SBATCH --qos=jelsana
#SBATCH --account=jelsana
#SBATCH --time=01-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=out/%x_%J.out
#SBATCH --error=out/%x_%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

# Slurm executes a temporary copy of this file from /var/spool. Therefore,
# BASH_SOURCE must not be used to locate the repository. SLURM_SUBMIT_DIR is
# the directory from which `sbatch` was executed.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"

# Also support submission while standing inside scripts/data.
if [[ ! -f "${PROJECT_DIR}/generateDataArabicThreeFontsCompatible.py" ]] \
  && [[ -f "${PROJECT_DIR}/../../generateDataArabicThreeFontsCompatible.py" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}/../..")"
fi

if [[ ! -f "${PROJECT_DIR}/generateDataArabicThreeFontsCompatible.py" ]]; then
  echo "ERROR: generator was not found under PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Submit this job from the AlignmentProject root, or set PROJECT_DIR explicitly." >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic_Three_Font_Augmented}"
REQUIREMENTS_FILE="${PROJECT_DIR}/scripts/data/requirements_three_font_generator.txt"

mkdir -p "${PROJECT_DIR}/out"
cd "${PROJECT_DIR}"

printf '%s\n' \
  "Started:       $(date)" \
  "Job ID:        ${SLURM_JOB_ID:-not-set}" \
  "Node:          ${SLURM_JOB_NODELIST:-not-set}" \
  "Submit dir:    ${SLURM_SUBMIT_DIR:-not-set}" \
  "Project:       ${PROJECT_DIR}" \
  "Conda env:     ${CONDA_ENV}" \
  "CPU cores:     ${SLURM_CPUS_PER_TASK:-4}" \
  "Output:        ${OUTPUT_DIR}" \
  "Fonts:         Amiri-Regular.ttf, Arslan_Wessam_B.ttf, DejaVuSans-Bold.ttf"

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

# Validate rendering dependencies before creating any dataset files.
if ! python - <<'PY'
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
import numpy
from tqdm import tqdm
print("Generator dependencies are available.")
PY
then
  echo "ERROR: required Arabic rendering packages are missing from ${CONDA_ENV}." >&2
  echo "Install them once from the login node, then resubmit:" >&2
  echo "  conda activate ${CONDA_ENV}" >&2
  echo "  python -m pip install -r ${REQUIREMENTS_FILE}" >&2
  exit 2
fi

for font in \
  Fonts/Amiri-Regular.ttf \
  Fonts/Arslan_Wessam_B.ttf \
  Fonts/DejaVuSans-Bold.ttf; do
  if [[ ! -f "${font}" ]]; then
    echo "ERROR: required font is missing: ${PROJECT_DIR}/${font}" >&2
    exit 2
  fi
done

if [[ -d "${OUTPUT_DIR}" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: output directory is not empty: ${OUTPUT_DIR}" >&2
  echo "Remove it, rename it, or set OUTPUT_DIR to a different path before retrying." >&2
  exit 2
fi

python generateDataArabicThreeFontsCompatible.py \
  --font-dir Fonts \
  --fonts \
    Amiri-Regular.ttf \
    Arslan_Wessam_B.ttf \
    DejaVuSans-Bold.ttf \
  --font-count 3 \
  --samples-per-font 3000 \
  --output-dir "${OUTPUT_DIR}" \
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
