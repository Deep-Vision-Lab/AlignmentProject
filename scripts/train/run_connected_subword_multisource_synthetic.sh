#!/usr/bin/env bash
# Submit connected-subword CNN training with 3,000 samples from each of
# Synthetic_Arabic_1, Synthetic_Arabic_2, Synthetic_Arabic_3, and
# Synthetic_Arabic_4. The delegated launcher submits the Slurm job itself.
set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: JOB_ID=<name> bash scripts/train/run_connected_subword_multisource_synthetic.sh" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"

: "${JOB_ID:?Set JOB_ID to the output weights-folder name.}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DataSet}"
SYNTHETIC_SAMPLES_PER_DIR="${SYNTHETIC_SAMPLES_PER_DIR:-3000}"
SYNTHETIC_REQUIRE_FULL_PER_DIR="${SYNTHETIC_REQUIRE_FULL_PER_DIR:-1}"

DEFAULT_DIRS=(
  "${DATA_ROOT}/Synthetic_Arabic_1"
  "${DATA_ROOT}/Synthetic_Arabic_2"
  "${DATA_ROOT}/Synthetic_Arabic_3"
  "${DATA_ROOT}/Synthetic_Arabic_4"
)

if [[ -z "${SYNTHETIC_DATA_DIRS:-}" ]]; then
  SYNTHETIC_DATA_DIRS="$(IFS=,; printf '%s' "${DEFAULT_DIRS[*]}")"
fi

IFS=',' read -r -a SOURCE_DIRS <<< "${SYNTHETIC_DATA_DIRS}"
if [[ "${#SOURCE_DIRS[@]}" -ne 4 ]]; then
  echo "ERROR: SYNTHETIC_DATA_DIRS must contain exactly four comma-separated directories." >&2
  exit 2
fi

for source_dir in "${SOURCE_DIRS[@]}"; do
  source_dir="${source_dir//[[:space:]]/}"
  [[ -d "${source_dir}/images" && -d "${source_dir}/texts" ]] || {
    echo "ERROR: source must contain images/ and texts/: ${source_dir}" >&2
    exit 2
  }
done

# The original self-submitting launcher validates one ordinary synthetic root.
# DATA_DIR therefore points at the first source while SYNTHETIC_DATA_DIRS tells
# the patched DataLoader to combine all four roots.
DATA_DIR="${DATA_DIR:-${SOURCE_DIRS[0]//[[:space:]]/}}"
NUM_SAMPLES="${NUM_SAMPLES:-$((SYNTHETIC_SAMPLES_PER_DIR * ${#SOURCE_DIRS[@]}))}"
SYNTHETIC_MANUSCRIPT_AUGMENT="${SYNTHETIC_MANUSCRIPT_AUGMENT:-1}"

# Direct no-DTW supervision uses fixed renderer subword intervals. Its
# augmentation must preserve pixel geometry, so use the box-safe profile rather
# than the crop/rotate/resize zero-shot profile.
if [[ "${DIRECT_SUBWORD_SUPERVISION:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  ZERO_SHOT_PROFILE=0
  DIRECT_SUBWORD_BOX_SAFE_AUGMENT="${DIRECT_SUBWORD_BOX_SAFE_AUGMENT:-1}"
  if [[ -n "${DIRECT_SUBWORD_BOX_DIR:-}" ]]; then
    echo "WARNING: DIRECT_SUBWORD_BOX_DIR is shared across sources." >&2
    echo "         Leave it unset to use each source's own subword_boxes/ directory." >&2
  fi
else
  ZERO_SHOT_PROFILE="${ZERO_SHOT_PROFILE:-1}"
fi

export PROJECT_DIR DATA_ROOT DATA_DIR NUM_SAMPLES
export SYNTHETIC_DATA_DIRS SYNTHETIC_SAMPLES_PER_DIR
export SYNTHETIC_REQUIRE_FULL_PER_DIR SYNTHETIC_MANUSCRIPT_AUGMENT
export ZERO_SHOT_PROFILE DIRECT_SUBWORD_BOX_SAFE_AUGMENT

printf '%s\n' \
  "Balanced four-source synthetic training" \
  "  branch             = $(git branch --show-current)" \
  "  samples per source = ${SYNTHETIC_SAMPLES_PER_DIR}" \
  "  total samples      = ${NUM_SAMPLES}" \
  "  sources            = ${SYNTHETIC_DATA_DIRS}" \
  "  direct supervision = ${DIRECT_SUBWORD_SUPERVISION:-0}" \
  "  zero-shot profile  = ${ZERO_SHOT_PROFILE}" \
  "  box-safe augment   = ${DIRECT_SUBWORD_BOX_SAFE_AUGMENT:-0}"

exec bash scripts/train/run_connected_subword_synthetic.sh
