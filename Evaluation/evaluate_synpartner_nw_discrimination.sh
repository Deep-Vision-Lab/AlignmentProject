#!/usr/bin/env bash
# Same-protocol NW discrimination on held-out real positives vs no-shared controls.
# Designed for the clean synthetic-partner CNN+BiLSTM treatment branch.
set -euo pipefail
set -a

[[ "$#" -eq 0 ]] || {
  echo "Usage: WEIGHTS=<checkpoint> [RUN_TAG=<tag>] bash Evaluation/evaluate_synpartner_nw_discrimination.sh" >&2
  exit 2
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

EXPECTED_BRANCH="agent/cnn-bilstm-partial-overlap"
CURRENT_BRANCH="$(git branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] || {
  echo "ERROR: expected ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH:-<detached>}." >&2
  exit 2
}

: "${WEIGHTS:?Set WEIGHTS to the trained treatment checkpoint.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || {
  echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2
  exit 2
}

REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}")"
SOURCE_MANIFEST="$(readlink -f "${SOURCE_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}")"
[[ -f "${SOURCE_MANIFEST}" ]] || {
  echo "ERROR: manifest not found: ${SOURCE_MANIFEST}" >&2
  exit 2
}

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${RUN_TAG}}"
MANIFEST_DIR="${RESULTS_ROOT}/manifests"
POSITIVE_MANIFEST="${MANIFEST_DIR}/positive_test.jsonl"
NEGATIVE_MANIFEST="${MANIFEST_DIR}/no_shared_controls.jsonl"

N_SAMPLES="${N_SAMPLES:-100000}"
START_INDEX="${START_INDEX:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
FEATURE="${FEATURE:-local}"
SCORE_MODE="${SCORE_MODE:-mutual-z}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-match-score}"

[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: N_SAMPLES must be positive." >&2; exit 2; }
[[ "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: START_INDEX must be positive." >&2; exit 2; }
[[ "${FEATURE}" == "local" ]] || { echo "ERROR: discrimination protocol requires FEATURE=local." >&2; exit 2; }
[[ "${SCORE_MODE}" == "mutual-z" ]] || { echo "ERROR: discrimination protocol requires SCORE_MODE=mutual-z." >&2; exit 2; }

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
EVAL_JOB_NAME="${EVAL_JOB_NAME:-disc_synpartner_v1}"

# Keep real inference/preprocessing identical to the canonical NW evaluation.
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
REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"
# Discrimination itself must not fail just because a control lacks a bbox file.
REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-0}"
unset EVAL_WINDOW_STRIDE

export PROJECT_DIR WEIGHTS REAL_DATA_DIR SOURCE_MANIFEST RUN_TAG RESULTS_ROOT MANIFEST_DIR
export POSITIVE_MANIFEST NEGATIVE_MANIFEST N_SAMPLES START_INDEX DATASET_SPLIT_SEED
export FEATURE SCORE_MODE SCORE_CLIP THRESHOLD GAP HEATMAP_SOURCE
export CONDA_ENV PARTITION GPU_RESOURCE CPUS_PER_TASK MEMORY TIME_LIMIT MAIL_USER EVAL_JOB_NAME
export LINE_HEIGHT LINE_WIDTH TARGET_INK_HEIGHT_RATIO ZERO_SHOT_TARGET_INK_HEIGHT_RATIO
export ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP ZERO_SHOT_SOURCE_GEOMETRY
export REAL_BINARIZE REAL_BINARIZE_METHOD REAL_BINARIZE_AUTO_INVERT REAL_BINARIZE_AUTOCONTRAST
export SW_INK_AWARE SW_MIN_INK SW_BLANK_BLANK_SCORE SW_BLANK_INK_SCORE REAL_BOX_EVAL REAL_REQUIRE_BOX_ANNOTATIONS
set +a

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${RESULTS_ROOT}" "${MANIFEST_DIR}"

# Build the same canonical 60/20/20 positive test split as training, then select
# leakage-aware no_shared controls. Prefer rows touching held-out eval pages;
# otherwise use non-training pair IDs before falling back to masked training IDs.
python - "${SOURCE_MANIFEST}" "${POSITIVE_MANIFEST}" "${NEGATIVE_MANIFEST}" "${START_INDEX}" "${N_SAMPLES}" <<'PY'
import json, sys
from collections import OrderedDict
import DataLoader as DL
from RealDataSet import ArabicManifestLinePairDataset

src, posout, negout, start, n = sys.argv[1:]
start, n = int(start) - 1, int(n)

def ds(labels):
    return ArabicManifestLinePairDataset(
        src, allowed_labels=labels, max_samples=None, paired=True, min_text_score=0.0
    )

def indices(subset):
    return [int(i) for i in subset.indices]

def round_robin(rows):
    groups = OrderedDict()
    for i, row in enumerate(rows):
        groups.setdefault(str(row.get("pair_id", i)), []).append(row)
    positions = {key: 0 for key in groups}
    out = []
    while True:
        added = False
        for key, values in groups.items():
            p = positions[key]
            if p < len(values):
                out.append(values[p])
                positions[key] += 1
                added = True
        if not added:
            return out

def dump(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

positive = ds(("high_match", "medium_match"))
train, valid, test = DL._group_split_real_dataset(positive)
train_i, valid_i, test_i = indices(train), indices(valid), indices(test)
positive_rows = round_robin([positive.samples[i] for i in test_i])
if start >= len(positive_rows):
    raise SystemExit(f"ERROR: START_INDEX exceeds positive test rows={len(positive_rows)}")
selected_positive = positive_rows[start:start+n]
if not selected_positive:
    raise SystemExit("ERROR: no held-out positive rows selected")
dump(posout, selected_positive)

train_ids = {str(positive.samples[i].get("pair_id", i)) for i in train_i}
eval_pages = set()
for i in valid_i + test_i:
    row = positive.samples[i]
    for key in ("A_page_id", "B_page_id"):
        value = row.get(key)
        if value is not None:
            eval_pages.add(str(value))

def pair_key(row):
    return tuple(str((row.get(side) or {}).get("line_image_path", "")) for side in ("A", "B"))

train_positive_pairs = {pair_key(positive.samples[i]) for i in train_i}
negative = ds(("no_shared_content",))
tiers = {0: [], 1: [], 2: []}
for i, row in enumerate(negative.samples):
    if pair_key(row) in train_positive_pairs:
        continue
    pair_id = str(row.get("pair_id", i))
    pages = {
        str(value)
        for value in (row.get("A_page_id"), row.get("B_page_id"))
        if value is not None
    }
    tier = 0 if pages & eval_pages else 1 if pair_id not in train_ids else 2
    tiers[tier].append(row)

selected_negative = []
tier_counts = []
for tier in (0, 1, 2):
    before = len(selected_negative)
    for row in round_robin(tiers[tier]):
        if len(selected_negative) >= len(selected_positive):
            break
        selected_negative.append(row)
    tier_counts.append(len(selected_negative) - before)
if not selected_negative:
    raise SystemExit("ERROR: no no_shared_content controls available")
dump(negout, selected_negative)

print(
    f"discrimination manifests: positive_all={len(positive)} "
    f"train={len(train_i)} valid={len(valid_i)} test={len(test_i)} "
    f"selected_positive={len(selected_positive)} selected_negative={len(selected_negative)} "
    f"negative_tiers(eval_page/nontrain/masked)={tier_counts}",
    flush=True,
)
PY

POSITIVE_COUNT="$(grep -cve '^[[:space:]]*$' "${POSITIVE_MANIFEST}" || true)"
NEGATIVE_COUNT="$(grep -cve '^[[:space:]]*$' "${NEGATIVE_MANIFEST}" || true)"
[[ "${POSITIVE_COUNT}" -gt 0 && "${NEGATIVE_COUNT}" -gt 0 ]] || {
  echo "ERROR: empty discrimination manifest." >&2
  exit 2
}

COMMON=(
  --weights "${WEIGHTS}"
  --device cuda
  --data-dir "${REAL_DATA_DIR}"
  --dataset-type real
  --batch
  --real-split all
  --real-labels all
  --real-text-key text_original_path
  --real-min-text-score 0.0
  --start-index 1
  --feature "${FEATURE}"
  --score-mode "${SCORE_MODE}"
  --score-clip "${SCORE_CLIP}"
  --threshold "${THRESHOLD}"
  --gap "${GAP}"
  --heatmap-source "${HEATMAP_SOURCE}"
  --no-annotate-heatmap-values
  --no-save-binarized-images
)

POSITIVE_DIR="${RESULTS_ROOT}/positive_nw_local_mutualz"
NEGATIVE_DIR="${RESULTS_ROOT}/no_shared_nw_local_mutualz"

printf '%s\n' \
  "=== NW DISCRIMINATION: POSITIVE ===" \
  "weights=${WEIGHTS}" \
  "feature=${FEATURE}" \
  "score_mode=${SCORE_MODE}" \
  "samples=${POSITIVE_COUNT}"
python -m Evaluation.eval_img_align_nw_diagnostic \
  "${COMMON[@]}" \
  --arabic-manifest "${POSITIVE_MANIFEST}" \
  --n-samples "${POSITIVE_COUNT}" \
  --output-dir "${POSITIVE_DIR}"

printf '%s\n' \
  "=== NW DISCRIMINATION: NO-SHARED ===" \
  "samples=${NEGATIVE_COUNT}"
python -m Evaluation.eval_img_align_nw_diagnostic \
  "${COMMON[@]}" \
  --arabic-manifest "${NEGATIVE_MANIFEST}" \
  --n-samples "${NEGATIVE_COUNT}" \
  --output-dir "${NEGATIVE_DIR}"

# Produce the two primary ranking diagnostics used in this project plus several
# secondary scores. AUC is P(score_positive > score_negative), ties count 0.5.
python - "${POSITIVE_DIR}/samples.csv" "${NEGATIVE_DIR}/samples.csv" "${RESULTS_ROOT}" <<'PY'
import csv, json, math, sys
from pathlib import Path

pos_csv, neg_csv, root = map(Path, sys.argv[1:])

def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]

def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def matched(row):
    values = [number(row.get("line1_matched_fraction")), number(row.get("line2_matched_fraction"))]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None

def values(rows, metric):
    if metric == "matched_fraction":
        result = [matched(row) for row in rows]
    else:
        result = [number(row.get(metric)) for row in rows]
    return [value for value in result if value is not None]

def mean(items):
    return sum(items) / len(items) if items else None

def auc(positives, negatives):
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))

positive_rows = read(pos_csv)
negative_rows = read(neg_csv)
metrics = [
    "path_steps",
    "matched_fraction",
    "region_score",
    "mean_region_cosine",
    "normalized_score",
]
summary = {
    "protocol": {
        "algorithm": "Needleman-Wunsch",
        "feature": "local",
        "score_mode": "mutual-z",
        "positive_control": "canonical held-out high/medium test rows",
        "negative_control": "leakage-aware no_shared_content rows",
    },
    "positive_samples": len(positive_rows),
    "negative_samples": len(negative_rows),
    "metrics": {},
}
for metric in metrics:
    pos = values(positive_rows, metric)
    neg = values(negative_rows, metric)
    summary["metrics"][metric] = {
        "positive_mean": mean(pos),
        "negative_mean": mean(neg),
        "gap": (mean(pos) - mean(neg)) if pos and neg else None,
        "auc": auc(pos, neg),
        "positive_n": len(pos),
        "negative_n": len(neg),
    }

root.mkdir(parents=True, exist_ok=True)
(root / "discrimination_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (root / "discrimination_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["metric", "positive_mean", "negative_mean", "gap", "auc", "positive_n", "negative_n"],
    )
    writer.writeheader()
    for metric, row in summary["metrics"].items():
        writer.writerow({"metric": metric, **row})

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"summary={root / 'discrimination_summary.json'}")
PY

echo "=== NW DISCRIMINATION PASS ==="
echo "results=${RESULTS_ROOT}"
