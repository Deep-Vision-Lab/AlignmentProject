#!/usr/bin/env bash
# Phase-3 real Arabic alignment diagnostic suite. No training is performed.
set -euo pipefail
set -a

[[ "$#" -eq 0 ]] || { echo "Usage: WEIGHTS=<checkpoint> [RUN_TAG=<tag>] bash Evaluation/evaluate_real_alignment_diagnostics.sh" >&2; exit 2; }
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"
cd "${PROJECT_DIR}"
mkdir -p out

EXPECTED_BRANCH="agent/use-extra-real-lines-cnn"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] || { echo "ERROR: run from ${EXPECTED_BRANCH}." >&2; exit 2; }
: "${WEIGHTS:?Set WEIGHTS to the trained Phase-3 checkpoint.}"
WEIGHTS="$(readlink -f "${WEIGHTS}")"
[[ -f "${WEIGHTS}" ]] || { echo "ERROR: checkpoint not found: ${WEIGHTS}" >&2; exit 2; }
REAL_DATA_DIR="$(readlink -f "${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}")"
SOURCE_MANIFEST="$(readlink -f "${SOURCE_MANIFEST:-${REAL_DATA_DIR}/dataset_manifest.jsonl}")"
[[ -f "${SOURCE_MANIFEST}" ]] || { echo "ERROR: manifest not found: ${SOURCE_MANIFEST}" >&2; exit 2; }

RUN_TAG="${RUN_TAG:-$(basename "$(dirname "${WEIGHTS}")")_alignment_diagnostics}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/Results/Evaluation/Alignment_Diagnostics/${RUN_TAG}}"
MANIFEST_DIR="${RESULTS_ROOT}/manifests"
CANONICAL_TEST_MANIFEST="${MANIFEST_DIR}/canonical_positive_test.jsonl"
POSITIVE_MANIFEST="${MANIFEST_DIR}/diagnostic_positive_rows.jsonl"
NEGATIVE_MANIFEST="${MANIFEST_DIR}/diagnostic_no_shared_rows.jsonl"
N_SAMPLES="${N_SAMPLES:-10}"
START_INDEX="${START_INDEX:-1}"
DATASET_SPLIT_SEED="${DATASET_SPLIT_SEED:-42}"
SCORE_MODE="${SCORE_MODE:-mutual-z}"
SCORE_CLIP="${SCORE_CLIP:-4.0}"
THRESHOLD="${THRESHOLD:-0.45}"
GAP="${GAP:--0.30}"
HEATMAP_SOURCE="${HEATMAP_SOURCE:-match-score}"
[[ "${N_SAMPLES}" =~ ^[1-9][0-9]*$ && "${START_INDEX}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: N_SAMPLES/START_INDEX must be positive integers." >&2; exit 2; }
[[ "${SCORE_MODE}" == "mutual-z" && "${HEATMAP_SOURCE}" == "match-score" ]] || { echo "ERROR: diagnostics require SCORE_MODE=mutual-z and HEATMAP_SOURCE=match-score." >&2; exit 2; }

LINE_HEIGHT="${LINE_HEIGHT:-128}"; LINE_WIDTH="${LINE_WIDTH:-1024}"
TARGET_INK_HEIGHT_RATIO="${TARGET_INK_HEIGHT_RATIO:-0.72}"
ZERO_SHOT_TARGET_INK_HEIGHT_RATIO="${ZERO_SHOT_TARGET_INK_HEIGHT_RATIO:-${TARGET_INK_HEIGHT_RATIO}}"
ZERO_SHOT_PREPROCESS="${ZERO_SHOT_PREPROCESS:-1}"
ZERO_SHOT_PRESERVE_ASPECT="${ZERO_SHOT_PRESERVE_ASPECT:-1}"
ZERO_SHOT_FOREGROUND_CROP="${ZERO_SHOT_FOREGROUND_CROP:-1}"
ZERO_SHOT_SOURCE_GEOMETRY="${ZERO_SHOT_SOURCE_GEOMETRY:-1}"
REAL_BINARIZE="${REAL_BINARIZE:-1}"; REAL_BINARIZE_METHOD="${REAL_BINARIZE_METHOD:-otsu}"
REAL_BINARIZE_THRESHOLD="${REAL_BINARIZE_THRESHOLD:-180}"
REAL_BINARIZE_AUTO_INVERT="${REAL_BINARIZE_AUTO_INVERT:-1}"
REAL_BINARIZE_AUTOCONTRAST="${REAL_BINARIZE_AUTOCONTRAST:-1}"
SW_INK_AWARE="${SW_INK_AWARE:-1}"; SW_MIN_INK="${SW_MIN_INK:-0.02}"
SW_BLANK_BLANK_SCORE="${SW_BLANK_BLANK_SCORE:--0.20}"; SW_BLANK_INK_SCORE="${SW_BLANK_INK_SCORE:--0.50}"
REAL_BOX_ANNOTATIONS_ROOT="${REAL_BOX_ANNOTATIONS_ROOT:-${REAL_DATA_DIR}}"
REAL_BOX_EVAL="${REAL_BOX_EVAL:-1}"; REAL_REQUIRE_BOX_ANNOTATIONS="${REAL_REQUIRE_BOX_ANNOTATIONS:-1}"
REAL_EVAL_BALANCED=0
unset EVAL_WINDOW_STRIDE

CONDA_ENV="${CONDA_ENV:-manucripts_align}"; PARTITION="${PARTITION:-rtx4090}"; GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"; MEMORY="${MEMORY:-32G}"; TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"; EVAL_JOB_NAME="${EVAL_JOB_NAME:-phase3_alignment_diag}"
export PROJECT_DIR WEIGHTS REAL_DATA_DIR SOURCE_MANIFEST RUN_TAG RESULTS_ROOT MANIFEST_DIR CANONICAL_TEST_MANIFEST POSITIVE_MANIFEST NEGATIVE_MANIFEST
export N_SAMPLES START_INDEX DATASET_SPLIT_SEED SCORE_MODE SCORE_CLIP THRESHOLD GAP HEATMAP_SOURCE
export LINE_HEIGHT LINE_WIDTH TARGET_INK_HEIGHT_RATIO ZERO_SHOT_TARGET_INK_HEIGHT_RATIO ZERO_SHOT_PREPROCESS ZERO_SHOT_PRESERVE_ASPECT ZERO_SHOT_FOREGROUND_CROP ZERO_SHOT_SOURCE_GEOMETRY
export REAL_BINARIZE REAL_BINARIZE_METHOD REAL_BINARIZE_THRESHOLD REAL_BINARIZE_AUTO_INVERT REAL_BINARIZE_AUTOCONTRAST SW_INK_AWARE SW_MIN_INK SW_BLANK_BLANK_SCORE SW_BLANK_INK_SCORE
export REAL_BOX_ANNOTATIONS_ROOT REAL_BOX_EVAL REAL_REQUIRE_BOX_ANNOTATIONS REAL_EVAL_BALANCED CONDA_ENV PARTITION GPU_RESOURCE CPUS_PER_TASK MEMORY TIME_LIMIT MAIL_USER EVAL_JOB_NAME
set +a

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submitting Phase-3 alignment diagnostics: ${RESULTS_ROOT}"
  sbatch --job-name="${EVAL_JOB_NAME}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" \
    --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:1" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEMORY}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}" "${SCRIPT_PATH}"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${RESULTS_ROOT}" "${MANIFEST_DIR}"

read -r CHECKPOINT_WINDOW CHECKPOINT_STRIDE < <(python - "${WEIGHTS}" <<'PY'
import sys, torch
try: ckpt=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
except TypeError: ckpt=torch.load(sys.argv[1],map_location="cpu")
cfg=ckpt.get("model_config",{}) if isinstance(ckpt,dict) else {}; w=int(cfg.get("window_size",32))
if "stride" in cfg: s=max(1,int(cfg["stride"]))
else:
 m=str(cfg.get("window_overlap_mode","custom")).lower(); r=float(cfg.get("stride_ratio",.5))
 s=w if m=="no_overlap" else max(1,w//2) if m=="light_overlap" else max(1,w//4) if m=="dense_overlap" else max(1,int(w*r))
print(w,s)
PY
)
[[ "${CHECKPOINT_WINDOW}" -eq 32 ]] || { echo "ERROR: checkpoint window=${CHECKPOINT_WINDOW}; expected 32." >&2; exit 2; }
echo "Checkpoint geometry: window=${CHECKPOINT_WINDOW} stride=${CHECKPOINT_STRIDE} (no override)"

python - "${SOURCE_MANIFEST}" "${CANONICAL_TEST_MANIFEST}" "${POSITIVE_MANIFEST}" "${NEGATIVE_MANIFEST}" "${START_INDEX}" "${N_SAMPLES}" <<'PY'
import json,sys
from collections import OrderedDict
import DataLoader as DL
from RealDataSet import ArabicManifestLinePairDataset
src,canonical,posout,negout,start,n=sys.argv[1:]; start=int(start)-1; n=int(n)
def ds(labels): return ArabicManifestLinePairDataset(src,allowed_labels=labels,max_samples=None,paired=True,min_text_score=0.0)
pos=ds(("high_match","medium_match")); train,valid,test=DL._group_split_real_dataset(pos)
def indices(sub): return [int(i) for i in sub.indices]
tr,va,te=indices(train),indices(valid),indices(test)
def dump(path,rows):
 with open(path,"w",encoding="utf-8") as f:
  for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
def rr(rows):
 g=OrderedDict()
 for i,row in enumerate(rows): g.setdefault(str(row.get("pair_id",i)),[]).append(row)
 out=[]; p={k:0 for k in g}
 while True:
  added=False
  for k,v in g.items():
   if p[k]<len(v): out.append(v[p[k]]); p[k]+=1; added=True
  if not added:return out
canonical_rows=[pos.samples[i] for i in te]; dump(canonical,canonical_rows)
ordered=rr(canonical_rows)
if start>=len(ordered): raise SystemExit(f"ERROR: START_INDEX exceeds test rows={len(ordered)}")
selected=ordered[start:start+n]
if not selected: raise SystemExit("ERROR: no positive diagnostic rows selected")
dump(posout,selected)
train_ids={str(pos.samples[i].get("pair_id",i)) for i in tr}; eval_pages=set()
for i in va+te:
 for k in ("A_page_id","B_page_id"):
  v=pos.samples[i].get(k)
  if v is not None: eval_pages.add(str(v))
def pairkey(row): return tuple(str((row.get(s) or {}).get("line_image_path", "")) for s in ("A","B"))
train_pairs={pairkey(pos.samples[i]) for i in tr}; neg=ds(("no_shared_content",)); tiers={0:[],1:[],2:[]}
for i,row in enumerate(neg.samples):
 if pairkey(row) in train_pairs: continue
 pid=str(row.get("pair_id",i)); pages={str(v) for v in (row.get("A_page_id"),row.get("B_page_id")) if v is not None}
 tier=0 if pages&eval_pages else 1 if pid not in train_ids else 2; tiers[tier].append(row)
chosen=[]; counts=[]
for tier in (0,1,2):
 before=len(chosen)
 for row in rr(tiers[tier]):
  if len(chosen)>=len(selected): break
  chosen.append(row)
 counts.append(len(chosen)-before)
if not chosen: raise SystemExit("ERROR: no no_shared_content controls available")
dump(negout,chosen)
print(f"split positive={len(pos)} train={len(tr)} valid={len(va)} test={len(te)} selected={len(selected)}; negative tiers heldout/nontrain/masked={counts}")
PY

POSITIVE_COUNT="$(grep -cve '^[[:space:]]*$' "${POSITIVE_MANIFEST}" || true)"
NEGATIVE_COUNT="$(grep -cve '^[[:space:]]*$' "${NEGATIVE_MANIFEST}" || true)"
[[ "${POSITIVE_COUNT}" -gt 0 && "${NEGATIVE_COUNT}" -gt 0 ]] || { echo "ERROR: diagnostic manifest selection failed." >&2; exit 2; }
[[ "${NEGATIVE_COUNT}" -ge "${POSITIVE_COUNT}" ]] || echo "WARNING: ${NEGATIVE_COUNT} negatives for ${POSITIVE_COUNT} positives." >&2

COMMON=(--weights "${WEIGHTS}" --device cuda --data-dir "${REAL_DATA_DIR}" --dataset-type real --batch --real-split all --real-labels all \
 --real-text-key text_original_path --real-min-text-score 0.0 --start-index 1 --score-mode "${SCORE_MODE}" --score-clip "${SCORE_CLIP}" \
 --threshold "${THRESHOLD}" --gap "${GAP}" --heatmap-source "${HEATMAP_SOURCE}" --no-annotate-heatmap-values --no-save-binarized-images)
run_pos(){ local module="$1" name="$2" feature="$3"; echo "=== ${name} ==="; python -m "${module}" "${COMMON[@]}" --arabic-manifest "${POSITIVE_MANIFEST}" --n-samples "${POSITIVE_COUNT}" --feature "${feature}" --output-dir "${RESULTS_ROOT}/${name}"; }
run_pos Evaluation.eval_img_align_nw_diagnostic A_nw_local_mutualz local
run_pos Evaluation.eval_img_align_sw B_sw_local_mutualz local
run_pos Evaluation.eval_img_align_sw C_sw_grouped_mutualz grouped
run_pos Evaluation.eval_img_align_sw D_sw_contextual_mutualz contextual
echo "=== E_sw_local_mutualz_no_shared ==="
python -m Evaluation.eval_img_align_sw "${COMMON[@]}" --arabic-manifest "${NEGATIVE_MANIFEST}" --n-samples "${NEGATIVE_COUNT}" --feature local --output-dir "${RESULTS_ROOT}/E_sw_local_mutualz_no_shared"

python - "${RESULTS_ROOT}" <<'PY'
import csv,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); specs=[("A","A_nw_local_mutualz","Needleman-Wunsch","local","positive"),("B","B_sw_local_mutualz","Smith-Waterman","local","positive"),("C","C_sw_grouped_mutualz","Smith-Waterman","grouped","positive"),("D","D_sw_contextual_mutualz","Smith-Waterman","contextual","positive"),("E","E_sw_local_mutualz_no_shared","Smith-Waterman","local","no_shared_content")]
def num(v):
 try: x=float(v); return x if math.isfinite(x) else None
 except (TypeError,ValueError): return None
def mean(v): v=[x for x in v if x is not None]; return sum(v)/len(v) if v else None
out=[]
for key,d,alg,feat,control in specs:
 p=root/d
 with open(p/"samples.csv",newline="",encoding="utf-8") as f: rows=[r for r in csv.DictReader(f) if r.get("status")=="ok"]
 agg=json.loads((p/"summary.json").read_text(encoding="utf-8")); matched=[]
 for r in rows: matched.append(mean([num(r.get("line1_matched_fraction")),num(r.get("line2_matched_fraction"))]))
 out.append({"experiment":key,"configuration":d,"algorithm":alg,"feature":feat,"control":control,"score_mode":"mutual-z","samples":len(rows),"mean_path_cosine":mean([num(r.get("mean_path_cosine")) for r in rows]),"matched_fraction":mean(matched),"bbox_precision":agg.get("box_micro_precision"),"bbox_recall":agg.get("box_micro_recall"),"bbox_f1":agg.get("box_micro_f1"),"bbox_specificity":agg.get("box_micro_specificity"),"mean_bbox_interval_iou":agg.get("mean_box_interval_iou"),"mean_bbox_pixel_iou":agg.get("mean_box_pixel_iou")})
with open(root/"diagnostic_summary.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(root/"diagnostic_summary.csv")
PY

echo "Done: ${RESULTS_ROOT}/diagnostic_summary.csv"
