#!/usr/bin/env bash
# Evaluate RealSyntheticBridge V2 before/after bridge fine-tuning.
set -euo pipefail
set -a
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out
: "${CHECKPOINT:?Set CHECKPOINT to the checkpoint to evaluate.}"
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing ${CHECKPOINT}" >&2; exit 2; }
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
MANIFEST="${BRIDGE_DATA_DIR}/dataset_manifest.jsonl"
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing bridge manifest ${MANIFEST}" >&2; exit 2; }
RUN_TAG="${RUN_TAG:-bridge_eval_$(basename "$(dirname "${CHECKPOINT}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_TAG}}"
QUAL_SAMPLES="${QUAL_SAMPLES:-6}"
QUANT_SAMPLES="${QUANT_SAMPLES:-400}"
THRESHOLD="${THRESHOLD:-0.50}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  DEP_ARGS=(); [[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS+=(--dependency="${DEPENDENCY}")
  sbatch --job-name="bridgeeval_${RUN_TAG}" --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}" \
    --mail-type=ALL --mail-user="${MAIL_USER}" "${DEP_ARGS[@]}" --export=ALL "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python scripts/data/smoke_test_real_synthetic_bridge.py --data-dir "${BRIDGE_DATA_DIR}"
mkdir -p "${RESULTS_ROOT}"
export REAL_BINARIZE=1 REAL_BINARIZE_METHOD=otsu REAL_BOX_EVAL=0 REAL_REQUIRE_BOX_ANNOTATIONS=0

# Qualitative positive and negative examples.
for LABEL in medium_match no_shared_content; do
  python -m Evaluation.eval_img_align_sw \
    --weights "${CHECKPOINT}" --device cuda \
    --data-dir "${BRIDGE_DATA_DIR}" --arabic-manifest "${MANIFEST}" \
    --dataset-type real --batch --real-split all --real-labels "${LABEL}" \
    --n-samples "${QUAL_SAMPLES}" --feature local --score-mode raw \
    --threshold "${THRESHOLD}" --gap -0.30 --heatmap-source cosine \
    --no-save-binarized-images --output-dir "${RESULTS_ROOT}/qualitative/${LABEL}"
done

# Metrics-only bridge distributions.
for LABEL in medium_match no_shared_content; do
  python -m Evaluation.eval_img_align_sw_no_png \
    --weights "${CHECKPOINT}" --device cuda \
    --data-dir "${BRIDGE_DATA_DIR}" --arabic-manifest "${MANIFEST}" \
    --dataset-type real --batch --real-split all --real-labels "${LABEL}" \
    --n-samples "${QUANT_SAMPLES}" --feature local --score-mode raw \
    --threshold "${THRESHOLD}" --gap -0.30 --heatmap-source cosine \
    --no-save-binarized-images --output-dir "${RESULTS_ROOT}/quantitative/${LABEL}"
done

python - "${RESULTS_ROOT}" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])

def rows(label):
    path = root / "quantitative" / label / "samples.csv"
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "ok"]

def auc(pos, neg):
    values = [(float(x), 1) for x in pos] + [(float(x), 0) for x in neg]
    if not pos or not neg: return None
    values.sort(key=lambda x: x[0])
    ranks = [0.0] * len(values); i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[j][0] == values[i][0]: j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j): ranks[k] = rank
        i = j
    rank_sum = sum(r for r, v in zip(ranks, values) if v[1] == 1)
    return (rank_sum - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

p, n = rows("medium_match"), rows("no_shared_content")
summary = {"positive_samples": len(p), "negative_samples": len(n)}
for key in ("score", "path_steps", "line1_matched_fraction", "line2_matched_fraction", "mean_path_cosine"):
    pv = [float(r[key]) for r in p if r.get(key) not in (None, "")]
    nv = [float(r[key]) for r in n if r.get(key) not in (None, "")]
    summary[key + "_auc"] = auc(pv, nv)
    summary[key + "_positive_mean"] = sum(pv)/len(pv) if pv else None
    summary[key + "_negative_mean"] = sum(nv)/len(nv) if nv else None
(root / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "BRIDGE_EVAL_RESULTS=${RESULTS_ROOT}"
