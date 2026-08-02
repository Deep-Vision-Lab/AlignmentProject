#!/usr/bin/env bash
# Calibrate SW threshold on one synthetic subset, then evaluate a disjoint holdout.
set -euo pipefail
set -a

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

: "${WEIGHTS:?Set WEIGHTS to the direct synthetic checkpoint path.}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/DataSet/Synthetic_Arabic}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")")_threshold_sweep}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/${RUN_TAG}}"
THRESHOLDS="${THRESHOLDS:-0.60,0.70,0.80,0.85,0.90}"
CALIBRATION_START_INDEX="${CALIBRATION_START_INDEX:-1}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-100}"
HOLDOUT_START_INDEX="${HOLDOUT_START_INDEX:-101}"
HOLDOUT_SAMPLES="${HOLDOUT_SAMPLES:-100}"
FEATURE="${FEATURE:-local}"
SCORE_MODE="${SCORE_MODE:-raw}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-dp-score}"

SPAN_TOKENIZATION_MODE=connected_subword
SPAN_USE_BLANK_TRANSITIONS=1
MAX_WINDOWS_PER_SPAN="${MAX_WINDOWS_PER_SPAN:-16}"
SPAN_CONNECTED_WINDOWS_PER_CHAR="${SPAN_CONNECTED_WINDOWS_PER_CHAR:-3}"
SPAN_CONNECTED_EXTRA_WINDOWS="${SPAN_CONNECTED_EXTRA_WINDOWS:-1}"
SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS="${SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS:-2}"
SPAN_SPACE_MAX_WINDOWS="${SPAN_SPACE_MAX_WINDOWS:-3}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-eval_connected_direct_sweep}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
set +a

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  [[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
    echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
    exit 2
  }
  if [[ -z "${DEPENDENCY_JOB_ID}" && ! -f "${WEIGHTS}" ]]; then
    echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
    exit 2
  fi

  SBATCH_ARGS=(
    --parsable
    --job-name="${EVAL_JOB_NAME}"
    --output="${PROJECT_DIR}/out/%x_%J.out"
    --chdir="${PROJECT_DIR}"
    --partition="${PARTITION}"
    --gpus="${GPU_RESOURCE}:1"
    --tasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --mail-type=ALL
    --mail-user="${MAIL_USER}"
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}"
  )
  if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
  fi

  printf '%s\n' \
    "Direct connected-subword synthetic calibration + holdout evaluation" \
    "  checkpoint          = ${WEIGHTS}" \
    "  feature             = ${FEATURE}" \
    "  thresholds          = ${THRESHOLDS}" \
    "  calibration samples = ${CALIBRATION_SAMPLES} from ${CALIBRATION_START_INDEX}" \
    "  holdout samples     = ${HOLDOUT_SAMPLES} from ${HOLDOUT_START_INDEX}" \
    "  results             = ${RESULTS_ROOT}" \
    "  dependency          = ${DEPENDENCY_JOB_ID:-none}"
  sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_PATH}"
  exit 0
fi

[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: dependent training completed but checkpoint was not found: ${WEIGHTS}" >&2
  exit 2
}
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || {
  echo "ERROR: synthetic dataset must contain images/ and texts/: ${DATA_DIR}" >&2
  exit 2
}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
CALIBRATION_ROOT="${RESULTS_ROOT}/calibration"
HOLDOUT_ROOT="${RESULTS_ROOT}/holdout"
mkdir -p "${CALIBRATION_ROOT}" "${HOLDOUT_ROOT}"

IFS=',' read -r -a THRESHOLD_ARRAY <<< "${THRESHOLDS}"
for THRESHOLD in "${THRESHOLD_ARRAY[@]}"; do
  THRESHOLD="${THRESHOLD//[[:space:]]/}"
  [[ -n "${THRESHOLD}" ]] || continue
  LABEL="threshold_${THRESHOLD//./p}"
  OUTPUT_DIR="${CALIBRATION_ROOT}/${LABEL}"
  mkdir -p "${OUTPUT_DIR}"

  echo "Calibrating threshold=${THRESHOLD} -> ${OUTPUT_DIR}"
  python -m Evaluation.eval_connected_subword \
    --weights "${WEIGHTS}" \
    --device cuda \
    --data-dir "${DATA_DIR}" \
    --dataset-type synthetic \
    --batch \
    --start-index "${CALIBRATION_START_INDEX}" \
    --n-samples "${CALIBRATION_SAMPLES}" \
    --feature "${FEATURE}" \
    --score-mode "${SCORE_MODE}" \
    --score-clip "${SCORE_CLIP}" \
    --threshold "${THRESHOLD}" \
    --gap "${GAP}" \
    --heatmap-source "${HEATMAP_SOURCE}" \
    --output-dir "${OUTPUT_DIR}"
done

python - "${CALIBRATION_ROOT}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("threshold_*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = summary_path.parent.name.removeprefix("threshold_").replace("p", ".")
    rows.append(
        {
            "threshold": float(threshold),
            "successful": data.get("successful"),
            "mean_region_iou": data.get("mean_region_iou"),
            "mean_start_error_px": data.get("mean_start_error_px"),
            "mean_end_error_px": data.get("mean_end_error_px"),
            "mean_path_cosine": data.get("mean_path_cosine"),
            "mean_line1_matched_fraction": data.get("mean_line1_matched_fraction"),
            "mean_line2_matched_fraction": data.get("mean_line2_matched_fraction"),
        }
    )
if not rows:
    raise SystemExit("No calibration summaries were produced")
with (root / "threshold_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
valid = [row for row in rows if row["mean_region_iou"] is not None]
if not valid:
    raise SystemExit("No valid calibration mean_region_iou was produced")
best = max(valid, key=lambda row: float(row["mean_region_iou"]))
(root / "best_threshold.json").write_text(
    json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
(root / "best_threshold.txt").write_text(str(best["threshold"]) + "\n", encoding="utf-8")
print("Calibration winner:", json.dumps(best, ensure_ascii=False))
PY

BEST_THRESHOLD="$(tr -d '[:space:]' < "${CALIBRATION_ROOT}/best_threshold.txt")"
HOLDOUT_OUTPUT="${HOLDOUT_ROOT}/threshold_${BEST_THRESHOLD//./p}"
mkdir -p "${HOLDOUT_OUTPUT}"

echo "Evaluating calibrated threshold=${BEST_THRESHOLD} on the disjoint holdout subset"
python -m Evaluation.eval_connected_subword \
  --weights "${WEIGHTS}" \
  --device cuda \
  --data-dir "${DATA_DIR}" \
  --dataset-type synthetic \
  --batch \
  --start-index "${HOLDOUT_START_INDEX}" \
  --n-samples "${HOLDOUT_SAMPLES}" \
  --feature "${FEATURE}" \
  --score-mode "${SCORE_MODE}" \
  --score-clip "${SCORE_CLIP}" \
  --threshold "${BEST_THRESHOLD}" \
  --gap "${GAP}" \
  --heatmap-source "${HEATMAP_SOURCE}" \
  --output-dir "${HOLDOUT_OUTPUT}"

python - "${CALIBRATION_ROOT}/best_threshold.json" "${HOLDOUT_OUTPUT}/summary.json" "${RESULTS_ROOT}/final_summary.json" <<'PY'
import json
import sys
from pathlib import Path

calibration = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
holdout = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
result = {
    "threshold_selected_on": "synthetic_calibration_subset",
    "selected_threshold": calibration["threshold"],
    "calibration": calibration,
    "holdout": holdout,
}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
