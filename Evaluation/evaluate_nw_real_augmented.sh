#!/usr/bin/env bash
# Evaluate a trained CNN+BiLSTM or ViT checkpoint with component-aware
# Needleman-Wunsch on the explicit test split of ArabicDatasetRealAug10K.
#
# Real localization metrics use the generated pair-specific alignment masks, the
# same way the synthetic augmented evaluator uses its committed binary masks.
set -euo pipefail
set -a

if [[ "$#" -ne 0 ]]; then
  echo "Usage: WEIGHTS=<checkpoint> bash Evaluation/evaluate_nw_real_augmented.sh" >&2
  echo "Optional overrides: REAL_DATA_DIR, TEST_MANIFEST, REAL_MASK_MANIFEST, N_SAMPLES, LABELS, RESULTS_ROOT." >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to the trained model_latest.pth/model_best.pth checkpoint.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

NW_ENTRYPOINT="${PROJECT_DIR}/Evaluation/eval_img_align_nw.py"
[[ -f "${NW_ENTRYPOINT}" ]] || {
  echo "ERROR: NW evaluator is not available on this branch: ${NW_ENTRYPOINT}" >&2
  echo "Use agent/training-speed-optimization for the component-aware NW evaluator." >&2
  exit 2
}

REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDatasetRealAug10K}"
REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR}")"
[[ -d "${REAL_DATA_DIR}" ]] || {
  echo "ERROR: real augmented dataset not found: ${REAL_DATA_DIR}" >&2
  exit 2
}

# Prefer the companion manifest written by build_real_alignment_masks.py.  The
# original explicit test manifest remains untouched and is used only when the
# caller explicitly disables mask evaluation.
REAL_MASK_EVAL="${REAL_MASK_EVAL:-1}"
REAL_REQUIRE_ALIGNMENT_MASKS="${REAL_REQUIRE_ALIGNMENT_MASKS:-1}"
REAL_BOX_EVAL="${REAL_BOX_EVAL:-0}"
case "${REAL_MASK_EVAL}" in 0|1) ;; *) echo "ERROR: REAL_MASK_EVAL must be 0 or 1." >&2; exit 2 ;; esac
case "${REAL_REQUIRE_ALIGNMENT_MASKS}" in 0|1) ;; *) echo "ERROR: REAL_REQUIRE_ALIGNMENT_MASKS must be 0 or 1." >&2; exit 2 ;; esac
case "${REAL_BOX_EVAL}" in 0|1) ;; *) echo "ERROR: REAL_BOX_EVAL must be 0 or 1." >&2; exit 2 ;; esac

if [[ -z "${TEST_MANIFEST:-}" ]]; then
  if [[ "${REAL_MASK_EVAL}" == "1" && -f "${REAL_DATA_DIR}/test_manifest_with_masks.jsonl" ]]; then
    TEST_MANIFEST="${REAL_DATA_DIR}/test_manifest_with_masks.jsonl"
  else
    TEST_MANIFEST="${REAL_DATA_DIR}/test_manifest.jsonl"
  fi
fi
TEST_MANIFEST="$(readlink -f "${TEST_MANIFEST}")"
[[ -f "${TEST_MANIFEST}" ]] || {
  echo "ERROR: explicit test manifest not found: ${TEST_MANIFEST}" >&2
  exit 2
}

if [[ -z "${REAL_MASK_MANIFEST:-}" ]]; then
  if [[ "${TEST_MANIFEST}" == *_with_masks.jsonl ]]; then
    REAL_MASK_MANIFEST="${TEST_MANIFEST}"
  else
    MASK_CANDIDATE="${TEST_MANIFEST%.jsonl}_with_masks.jsonl"
    if [[ -f "${MASK_CANDIDATE}" ]]; then
      REAL_MASK_MANIFEST="${MASK_CANDIDATE}"
    else
      REAL_MASK_MANIFEST="${TEST_MANIFEST}"
    fi
  fi
fi
REAL_MASK_MANIFEST="$(readlink -f "${REAL_MASK_MANIFEST}")"

TEST_ROWS="$(grep -cve '^[[:space:]]*$' "${TEST_MANIFEST}" || true)"
[[ "${TEST_ROWS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: test manifest contains no rows: ${TEST_MANIFEST}" >&2
  exit 2
}

# Fail before requesting a GPU when mask ground truth is incomplete.  Paths are
# resolved exactly as the real dataset manifests resolve them: relative to the
# manifest/dataset root first, then absolute paths are accepted unchanged.
if [[ "${REAL_MASK_EVAL}" == "1" ]]; then
  [[ -f "${REAL_MASK_MANIFEST}" ]] || {
    echo "ERROR: real alignment-mask manifest not found: ${REAL_MASK_MANIFEST}" >&2
    echo "Run scripts/data/build_real_alignment_masks.py --all-manifests --write-manifests first." >&2
    exit 2
  }
  python - "${REAL_MASK_MANIFEST}" "${REAL_DATA_DIR}" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1]).expanduser().resolve()
root = pathlib.Path(sys.argv[2]).expanduser().resolve()
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
missing_fields = []
missing_files = []
usable = 0


def resolve(value):
    path = pathlib.Path(str(value).replace("\\", "/")).expanduser()
    if path.is_absolute():
        return path
    for candidate in (manifest.parent / path, root / path, root.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (manifest.parent / path).resolve()

for index, row in enumerate(rows, start=1):
    a, b = row.get("A"), row.get("B")
    if not isinstance(a, dict) or not isinstance(b, dict):
        missing_fields.append((index, "A/B"))
        continue
    ma, mb = a.get("alignment_mask_path"), b.get("alignment_mask_path")
    if not ma or not mb:
        missing_fields.append((index, "alignment_mask_path"))
        continue
    pa, pb = resolve(ma), resolve(mb)
    absent = [str(path) for path in (pa, pb) if not path.is_file()]
    if absent:
        missing_files.append((index, absent))
        continue
    usable += 1

print(f"mask_ground_truth manifest={manifest} rows={len(rows)} usable={usable}")
if missing_fields or missing_files:
    print(
        f"ERROR: incomplete real alignment-mask ground truth: "
        f"missing_fields={len(missing_fields)} missing_files={len(missing_files)}",
        file=sys.stderr,
    )
    for item in missing_fields[:5]:
        print(f"  missing field row={item[0]} field={item[1]}", file=sys.stderr)
    for item in missing_files[:5]:
        print(f"  missing mask file row={item[0]} paths={item[1]}", file=sys.stderr)
    raise SystemExit(2)
if usable == 0:
    raise SystemExit("ERROR: mask manifest contains no usable A/B alignment masks")
PY
fi

N_SAMPLES="${N_SAMPLES:-${TEST_ROWS}}"
START_INDEX="${START_INDEX:-1}"
LABELS="${LABELS:-all}"
FEATURE="${FEATURE:-contextual}"
SCORE_MODE="${SCORE_MODE:-auto}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/NW/RealAugmented/${RUN_TAG}}"

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: N_SAMPLES must be a positive integer." >&2
  exit 2
}
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: START_INDEX must be a positive integer." >&2
  exit 2
}
if (( START_INDEX > TEST_ROWS )); then
  echo "ERROR: START_INDEX=${START_INDEX} exceeds test rows=${TEST_ROWS}." >&2
  exit 2
fi
AVAILABLE=$((TEST_ROWS - START_INDEX + 1))
if (( N_SAMPLES > AVAILABLE )); then
  N_SAMPLES="${AVAILABLE}"
fi

# Real image preprocessing: same geometry/binarization used by training/eval.
LINE_HEIGHT="${LINE_HEIGHT:-128}"
LINE_WIDTH="${LINE_WIDTH:-1024}"
TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-${TARGET_INK_HEIGHT_RATIO}}"
ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"
REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
SW_INK_AWARE="${SW_INK_AWARE:-1}"
SW_MIN_INK="${SW_MIN_INK:-0.02}"
SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"
SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"

# Same component-aware interpretation used by the augmented synthetic NW
# evaluator. Ground-truth masks affect reporting only; they never select runs.
NW_COMPONENT_MAX_COMPONENTS="${NW_COMPONENT_MAX_COMPONENTS:-3}"
NW_COMPONENT_MIN_MATCHES="${NW_COMPONENT_MIN_MATCHES:-7}"
NW_COMPONENT_MIN_SPAN_WINDOWS="${NW_COMPONENT_MIN_SPAN_WINDOWS:-7}"
NW_COMPONENT_MIN_SPAN_FRACTION="${NW_COMPONENT_MIN_SPAN_FRACTION:-0.13}"
NW_COMPONENT_WEAK_GLOBAL_SCORE="${NW_COMPONENT_WEAK_GLOBAL_SCORE:--1000000.0}"

# Legacy bbox evaluation remains available only as an explicit ablation:
# REAL_MASK_EVAL=0 REAL_BOX_EVAL=1.
REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-0}"
REAL_BOX_IN_MASK_RULE="${REAL_BOX_IN_MASK_RULE:-center}"
REAL_BOX_MIN_COVERAGE="${REAL_BOX_MIN_COVERAGE:-0.50}"
REAL_BOX_COORDINATE_SPACE="${REAL_BOX_COORDINATE_SPACE:-original}"
REAL_BOX_BBOX_FORMAT="${REAL_BOX_BBOX_FORMAT:-auto}"
REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-${REAL_DATA_DIR}}"
REAL_BOX_JSON="${REAL_BOX_JSON:-}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-nw_real_aug_${RUN_TAG}}"

export PROJECT_DIR WEIGHTS REAL_DATA_DIR TEST_MANIFEST TEST_ROWS
export REAL_MASK_EVAL REAL_MASK_MANIFEST REAL_REQUIRE_ALIGNMENT_MASKS REAL_BOX_EVAL
export N_SAMPLES START_INDEX LABELS FEATURE SCORE_MODE SCORE_CLIP THRESHOLD GAP
export HEATMAP_SOURCE RUN_TAG RESULTS_ROOT
export LINE_HEIGHT LINE_WIDTH TARGET_INK_HEIGHT_RATIO ZERO_SHOT_TARGET_INK_HEIGHT_RATIO
export ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP
export ZERO_SHOT_SOURCE_GEOMETRY REAL_BINARIZE REAL_BINARIZE_METHOD
export REAL_BINARIZE_AUTO_INVERT REAL_BINARIZE_AUTOCONTRAST
export SW_INK_AWARE SW_MIN_INK SW_BLANK_BLANK_SCORE SW_BLANK_INK_SCORE
export NW_COMPONENT_MAX_COMPONENTS NW_COMPONENT_MIN_MATCHES
export NW_COMPONENT_MIN_SPAN_WINDOWS NW_COMPONENT_MIN_SPAN_FRACTION
export NW_COMPONENT_WEAK_GLOBAL_SCORE
export REAL_REQUIRE_BOX_ANNOTATIONS REAL_BOX_IN_MASK_RULE REAL_BOX_MIN_COVERAGE
export REAL_BOX_COORDINATE_SPACE REAL_BOX_BBOX_FORMAT REAL_BOX_ANNOTATIONS_ROOT REAL_BOX_JSON
export CONDA_ENV PARTITION GPU_RESOURCE CPUS_PER_TASK MEMORY TIME_LIMIT
export MAIL_USER EVAL_JOB_NAME
set +a

print_config() {
  printf '%s\n' \
    "Component-aware NW evaluation on real augmented data" \
    "  branch=$(git branch --show-current)" \
    "  checkpoint=${WEIGHTS}" \
    "  dataset=${REAL_DATA_DIR}" \
    "  manifest=${TEST_MANIFEST}" \
    "  split=explicit test manifest (no re-split)" \
    "  manifest rows=${TEST_ROWS}" \
    "  start=${START_INDEX}" \
    "  samples=${N_SAMPLES}" \
    "  labels=${LABELS}" \
    "  algorithm=Needleman-Wunsch, up to ${NW_COMPONENT_MAX_COMPONENTS} components" \
    "  mask scoring=${REAL_MASK_EVAL}" \
    "  mask manifest=${REAL_MASK_MANIFEST}" \
    "  bbox fallback=${REAL_BOX_EVAL}" \
    "  metrics=same component-union mask IoU/start/end errors as synthetic augmented NW" \
    "  results=${RESULTS_ROOT}" \
    "  GPU=${GPU_RESOURCE}:1" \
    "  time limit=${TIME_LIMIT}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  print_config
  sbatch \
    --job-name="${EVAL_JOB_NAME}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" \
    --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --mail-type=ALL \
    --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" \
    "${SCRIPT_PATH}"
  exit 0
fi

resolve_env_prefix() {
  local candidate
  for candidate in \
    "${CONDA_PREFIX:-}" \
    "${HOME}/.conda/envs/${CONDA_ENV}" \
    "${HOME}/miniconda3/envs/${CONDA_ENV}" \
    "${HOME}/anaconda3/envs/${CONDA_ENV}"; do
    [[ -n "${candidate}" ]] || continue
    [[ -x "${candidate}/bin/python" ]] || continue
    if "${candidate}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
import numpy
from PIL import Image
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ENV_PREFIX="$(resolve_env_prefix)" || {
  echo "ERROR: could not find usable conda environment '${CONDA_ENV}'." >&2
  exit 2
}
TRAIN_PYTHON="${ENV_PREFIX}/bin/python"
export PATH="${ENV_PREFIX}/bin:${PATH}"
hash -r

mkdir -p "${RESULTS_ROOT}"
print_config
printf '%s\n' "  python=${TRAIN_PYTHON}"
"${TRAIN_PYTHON}" -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
nvidia-smi -L || true

"${TRAIN_PYTHON}" -m Evaluation.eval_img_align_nw \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${REAL_DATA_DIR}" \
  --arabic-manifest "${TEST_MANIFEST}" \
  --dataset-type real \
  --batch \
  --real-split all \
  --real-labels "${LABELS}" \
  --real-text-key text_original_path \
  --real-min-text-score 0.0 \
  --start-index "${START_INDEX}" \
  --n-samples "${N_SAMPLES}" \
  --feature "${FEATURE}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${THRESHOLD}" \
  --gap "${GAP}" \
  --heatmap-source "${HEATMAP_SOURCE}" \
  --no-annotate-heatmap-values \
  --no-save-binarized-images \
  --output-dir "${RESULTS_ROOT}"
