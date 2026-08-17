#!/usr/bin/env bash
# Final frozen evaluation over every row in the canonical real manifest.
set -euo pipefail
set -a
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out
: "${CHECKPOINT:?Set CHECKPOINT to the final checkpoint.}"
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing ${CHECKPOINT}" >&2; exit 2; }
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
MANIFEST="${REAL_DATA_DIR}/dataset_manifest.jsonl"
[[ -s "${MANIFEST}" ]] || { echo "ERROR: missing ${MANIFEST}" >&2; exit 2; }
RUN_TAG="${RUN_TAG:-final_all_real_$(basename "$(dirname "${CHECKPOINT}")") }"
RUN_TAG="${RUN_TAG% }"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_TAG}}"
FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}"

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
PARTITION="${PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  DEP_ARGS=(); [[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS+=(--dependency="${DEPENDENCY}")
  sbatch --job-name="final_${RUN_TAG}" --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" \
    --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}" \
    --mail-type=ALL --mail-user="${MAIL_USER}" "${DEP_ARGS[@]}" --export=ALL "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export REAL_BINARIZE=1 REAL_BINARIZE_METHOD=otsu REAL_BOX_EVAL=0 REAL_REQUIRE_BOX_ANNOTATIONS=0
export ZERO_SHOT_PREPROCESS=1 ZERO_SHOT_PRESERVE_ASPECT=1 ZERO_SHOT_FOREGROUND_CROP=1 ZERO_SHOT_SOURCE_GEOMETRY=1
mkdir -p "${RESULTS_ROOT}"

# Compute exact full-manifest counts so no label is sampled or truncated.
eval "$(python - "${MANIFEST}" <<'PY'
import json, sys
from collections import Counter
c = Counter()
with open(sys.argv[1], encoding='utf-8') as f:
    for line in f:
        if line.strip(): c[json.loads(line).get('label_type','')] += 1
for label in ('high_match','medium_match','low_match','no_shared_content'):
    print(f"COUNT_{label.upper()}={c[label]}")
PY
)"

declare -A COUNTS=(
  [high_match]="${COUNT_HIGH_MATCH}"
  [medium_match]="${COUNT_MEDIUM_MATCH}"
  [low_match]="${COUNT_LOW_MATCH}"
  [no_shared_content]="${COUNT_NO_SHARED_CONTENT}"
)

for LABEL in high_match medium_match low_match no_shared_content; do
  N="${COUNTS[$LABEL]}"
  (( N > 0 )) || continue
  echo "=== FINAL ALL-REAL: ${LABEL} n=${N} ==="
  python -m Evaluation.eval_img_align_sw_no_png \
    --weights "${CHECKPOINT}" --device cuda \
    --data-dir "${REAL_DATA_DIR}" --arabic-manifest "${MANIFEST}" \
    --dataset-type real --batch --real-split all --real-labels "${LABEL}" \
    --start-index 1 --n-samples "${N}" --feature local --score-mode raw \
    --threshold "${FINAL_THRESHOLD}" --gap -0.30 --heatmap-source cosine \
    --no-save-binarized-images --output-dir "${RESULTS_ROOT}/${LABEL}"
done

python - "${RESULTS_ROOT}" "${FINAL_THRESHOLD}" <<'PY'
import csv, json, math, sys
from pathlib import Path
root = Path(sys.argv[1]); threshold = float(sys.argv[2])
labels = ['high_match','medium_match','low_match','no_shared_content']

def load(label):
    p = root / label / 'samples.csv'
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r.get('status') == 'ok']

def vals(rows, key):
    out=[]
    for r in rows:
        try:
            x=float(r[key])
            if math.isfinite(x): out.append(x)
        except Exception: pass
    return out

def mean(v): return sum(v)/len(v) if v else None

def std(v):
    if not v: return None
    m=mean(v); return (sum((x-m)**2 for x in v)/len(v))**0.5

def auc(pos, neg):
    data=[(x,1) for x in pos]+[(x,0) for x in neg]
    if not pos or not neg: return None
    data.sort(key=lambda z:z[0]); ranks=[0.0]*len(data); i=0
    while i<len(data):
        j=i+1
        while j<len(data) and data[j][0]==data[i][0]: j+=1
        r=(i+1+j)/2.0
        for k in range(i,j): ranks[k]=r
        i=j
    rs=sum(r for r,z in zip(ranks,data) if z[1]==1)
    return (rs-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def ap(pos, neg):
    data=sorted([(x,1) for x in pos]+[(x,0) for x in neg], key=lambda z:z[0], reverse=True)
    if not pos: return None
    tp=0; total=0; s=0.0
    for _,y in data:
        total+=1
        if y:
            tp+=1; s += tp/total
    return s/len(pos)

rows={label:load(label) for label in labels}
summary={'final_threshold':threshold,'labels':{}}
for label, rr in rows.items():
    item={'samples':len(rr)}
    for key in ('score','path_steps','line1_matched_fraction','line2_matched_fraction','mean_path_cosine'):
        v=vals(rr,key); item[key+'_mean']=mean(v); item[key+'_std']=std(v)
    summary['labels'][label]=item

positive=rows['high_match']+rows['medium_match']; negative=rows['no_shared_content']
summary['binary_positive_definition']='high_match + medium_match'
summary['binary_negative_definition']='no_shared_content'
summary['low_match_policy']='reported separately; not forced into binary positive/negative labels'
summary['binary']={}
for key in ('score','path_steps','line1_matched_fraction','line2_matched_fraction','mean_path_cosine'):
    p=vals(positive,key); n=vals(negative,key)
    summary['binary'][key]={'roc_auc':auc(p,n),'average_precision':ap(p,n),'positive_mean':mean(p),'negative_mean':mean(n),'positive_n':len(p),'negative_n':len(n)}
(root/'final_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
PY

echo "FINAL_REAL_RESULTS=${RESULTS_ROOT}"
