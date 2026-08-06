#!/usr/bin/env bash
#SBATCH --job-name=gen_27k_three_font_63_cpu
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=out/%x_%J.out

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmedmas@post.bgu.ac.il

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"

if [[ ! -f "${PROJECT_DIR}/generateDataArabicThreeFontsFixed63.py" ]] \
  && [[ -f "${PROJECT_DIR}/../../generateDataArabicThreeFontsFixed63.py" ]]; then
  PROJECT_DIR="$(readlink -f "${PROJECT_DIR}/../..")"
fi

if [[ ! -f "${PROJECT_DIR}/generateDataArabicThreeFontsFixed63.py" ]]; then
  echo "ERROR: fixed-63 generator was not found under PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "Submit from the AlignmentProject root, or set PROJECT_DIR explicitly." >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
SAMPLES_PER_FONT="${SAMPLES_PER_FONT:-9000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/DataSet/AugmentedArabicDataset63}"
REQUIREMENTS_FILE="${PROJECT_DIR}/scripts/data/requirements_three_font_generator.txt"
TOTAL_PAIRS=$((3 * SAMPLES_PER_FONT))
TOTAL_IMAGES=$((2 * TOTAL_PAIRS))
TOTAL_TEXTS=$((2 * TOTAL_PAIRS))

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
  "Per font:      ${SAMPLES_PER_FONT} pairs" \
  "Expected:      ${TOTAL_PAIRS} pairs / ${TOTAL_IMAGES} images / ${TOTAL_TEXTS} texts" \
  "Text length:   exactly 63 characters per line" \
  "Noise:         Gaussian + salt/pepper + optional blur" \
  "Fonts:         randomized from Amiri-Regular.ttf, Arslan_Wessam_B.ttf, DejaVuSans-Bold.ttf" \
  "Modes:         original, cross_injection, aligned_unaligned, two_aligned_parts, three_aligned_parts"

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

if ! python - <<'PY'
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFilter, ImageFont
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

SAMPLES_PER_FONT="${SAMPLES_PER_FONT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FONT_SIZE=84 \
MIN_TEXT_CHARS=63 \
MAX_TEXT_CHARS=63 \
NOISE_PROB=0.75 \
ORIGINAL_NOISE_SCALE=0.20 \
GAUSSIAN_NOISE_PROB=0.90 \
GAUSSIAN_NOISE_STD_MIN=2.0 \
GAUSSIAN_NOISE_STD_MAX=10.0 \
SALT_PEPPER_NOISE_PROB=0.55 \
SALT_PEPPER_DENSITY_MIN=0.0005 \
SALT_PEPPER_DENSITY_MAX=0.0035 \
BLUR_PROB=0.25 \
BLUR_RADIUS_MIN=0.15 \
BLUR_RADIUS_MAX=0.75 \
TARGET_FILL_RATIO=0.98 \
bash scripts/data/run_generate_three_font_synthetic.sh

printf '%s\n' \
  "Finished:      $(date)" \
  "Dataset:       ${OUTPUT_DIR}" \
  "Pairs:         ${TOTAL_PAIRS}" \
  "Images:        ${TOTAL_IMAGES}" \
  "Texts:         ${TOTAL_TEXTS}, all validated at exactly 63 characters"
