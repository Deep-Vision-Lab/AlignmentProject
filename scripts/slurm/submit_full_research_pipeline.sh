#!/usr/bin/env bash
# Submit the model curriculum with automatic tracking and afterok dependencies.
# Normally Bridge must already exist. With DEFER_BRIDGE_VALIDATION=1, S1-S3 are
# submitted immediately and the first Bridge-dependent stage waits for both S3 and
# BRIDGE_READY_JOB_ID (typically the D0 dataset job).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs/experiments

BRANCH="$(git branch --show-current)"
COMMIT="$(git rev-parse HEAD)"
BACKEND="$(python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
)"
case "${BRANCH}" in
  agent/training-speed-optimization|agent/use-vit-encoder|agent/use-dinov3-convnext) ;;
  *) echo "ERROR: run this pipeline only from a canonical architecture branch; got ${BRANCH}" >&2; exit 2 ;;
esac
if [[ "${BACKEND}" == "dinov3_convnext" ]]; then
  : "${DINOV3_REPO_DIR:?DINOv3 pipeline requires DINOV3_REPO_DIR.}"
  : "${DINOV3_WEIGHTS:?DINOv3 synthetic S1 requires DINOV3_WEIGHTS for the initial pretrained backbone.}"
fi

SYNTH_DATA_DIR="${SYNTH_DATA_DIR:-${PROJECT_DIR}/DataSet/AugmentedArabicDataset63}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
DEFER_BRIDGE_VALIDATION="${DEFER_BRIDGE_VALIDATION:-0}"
BRIDGE_READY_JOB_ID="${BRIDGE_READY_JOB_ID:-}"

[[ -d "${SYNTH_DATA_DIR}/images" && -d "${SYNTH_DATA_DIR}/texts" ]] || { echo "ERROR: missing synthetic data ${SYNTH_DATA_DIR}" >&2; exit 2; }
[[ -s "${REAL_DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing real manifest ${REAL_DATA_DIR}/dataset_manifest.jsonl" >&2; exit 2; }

if [[ "${DEFER_BRIDGE_VALIDATION}" == "1" ]]; then
  [[ "${BRIDGE_READY_JOB_ID}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: DEFER_BRIDGE_VALIDATION=1 requires numeric BRIDGE_READY_JOB_ID" >&2
    exit 2
  }
  echo "Bridge validation deferred to D0 job ${BRIDGE_READY_JOB_ID}."
  echo "S1-S3 will be submitted immediately; S4 will wait for S3 AND D0."
else
  [[ -s "${BRIDGE_DATA_DIR}/dataset_manifest.jsonl" && -s "${BRIDGE_DATA_DIR}/metadata.json" ]] || {
    echo "ERROR: Bridge V2 must be created before Bridge-dependent model stages: ${BRIDGE_DATA_DIR}" >&2
    echo "Run: bash scripts/slurm/submit_bridge_v2_dataset.sh" >&2
    exit 2
  }
  echo "Validating frozen Bridge V2 before submitting GPU jobs..."
  python scripts/data/smoke_test_real_synthetic_bridge.py --data-dir "${BRIDGE_DATA_DIR}"
fi

RUN_PREFIX="${RUN_PREFIX:-${BACKEND}_research_$(date +%Y%m%d_%H%M%S)}"
SYNTH_JOB_ID="${RUN_PREFIX}_synth"
BRIDGE_JOB_ID="${RUN_PREFIX}_bridge_v2"
SYNTH_CKPT="${PROJECT_DIR}/Weights/${SYNTH_JOB_ID}/checkpoint_latest.pth"
BRIDGE_BEST="${PROJECT_DIR}/Weights/${BRIDGE_JOB_ID}/checkpoint_best_val.pth"
RESULTS_BASE="${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_PREFIX}"
LEDGER="${PROJECT_DIR}/logs/research_pipeline_${RUN_PREFIX}.jobs"
TRACKER_JSON="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.json"
TRACKER_MD="${PROJECT_DIR}/logs/experiments/${RUN_PREFIX}.md"
TRACKER_TOOL="${PROJECT_DIR}/scripts/pipeline/experiment_tracker.py"
TRACKED_WRAPPER="${PROJECT_DIR}/scripts/pipeline/run_tracked_stage.sh"

if [[ "${ALLOW_EXISTING_WEIGHTS:-0}" != "1" ]]; then
  for d in "${PROJECT_DIR}/Weights/${SYNTH_JOB_ID}" "${PROJECT_DIR}/Weights/${BRIDGE_JOB_ID}"; do
    [[ ! -e "${d}" ]] || { echo "ERROR: ${d} already exists. Choose another RUN_PREFIX or set ALLOW_EXISTING_WEIGHTS=1." >&2; exit 2; }
  done
fi

TRAIN_GPUS="${TRAIN_GPUS:-2}"
TRAIN_PARTITION="${TRAIN_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
TRAIN_CPUS="${TRAIN_CPUS:-16}"
TRAIN_MEM="${TRAIN_MEM:-96G}"
EVAL_PARTITION="${EVAL_PARTITION:-rtx4090}"
EVAL_CPUS="${EVAL_CPUS:-4}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SYNTH_EPOCHS="${SYNTH_EPOCHS:-20}"
BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-25}"
BRIDGE_LR="${BRIDGE_LR:-1e-6}"
FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}"

python "${TRACKER_TOOL}" init \
  --tracker "${TRACKER_JSON}" --run-prefix "${RUN_PREFIX}" \
  --branch "${BRANCH}" --commit "${COMMIT}" --backend "${BACKEND}" \
  --bridge-dataset "${BRIDGE_DATA_DIR}" --results-root "${RESULTS_BASE}" >/dev/null

jobid() { cut -d';' -f1 <<<"$1"; }
register_stage() {
  local stage="$1" desc="$2" kind="$3" jid="$4" name="$5" dep="$6" log="$7" artifact="$8" checkpoint="$9" results="${10}"
  python "${TRACKER_TOOL}" register \
    --tracker "${TRACKER_JSON}" --stage "${stage}" --description "${desc}" --kind "${kind}" \
    --job-id "${jid}" --job-name "${name}" --dependency "${dep}" --log-path "${log}" \
    --artifact "${artifact}" --checkpoint "${checkpoint}" --result-root "${results}" >/dev/null
}
submit_gpu_train() {
  local stage="$1" dep="$2" name="$3" time="$4" exports="$5" script="$6" desc="$7" artifact="$8" checkpoint="$9" results="${10}"
  register_stage "${stage}" "${desc}" train "" "${name}" "${dep}" "" "${artifact}" "${checkpoint}" "${results}"
  local args=(--parsable --partition="${TRAIN_PARTITION}" --job-name="${name}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --gpus="${GPU_RESOURCE}:${TRAIN_GPUS}" --ntasks=1 --cpus-per-task="${TRAIN_CPUS}" --mem="${TRAIN_MEM}" --time="${time}" --mail-type=ALL --mail-user="${MAIL_USER}")
  [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
  args+=(--export="ALL,${exports},PIPELINE_TRACKER_JSON=${TRACKER_JSON},PIPELINE_STAGE=${stage},TRACKED_STAGE_SCRIPT=${script}")
  local jid; jid="$(jobid "$(sbatch "${args[@]}" "${TRACKED_WRAPPER}")")"
  register_stage "${stage}" "${desc}" train "${jid}" "${name}" "${dep}" "${PROJECT_DIR}/out/${name}_${jid}.out" "${artifact}" "${checkpoint}" "${results}"
  printf '%s' "${jid}"
}
submit_gpu_eval() {
  local stage="$1" dep="$2" name="$3" time="$4" exports="$5" script="$6" kind="$7" desc="$8" artifact="$9" checkpoint="${10}" results="${11}"
  register_stage "${stage}" "${desc}" "${kind}" "" "${name}" "${dep}" "" "${artifact}" "${checkpoint}" "${results}"
  local args=(--parsable --partition="${EVAL_PARTITION}" --job-name="${name}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --gpus="${GPU_RESOURCE}:1" --ntasks=1 --cpus-per-task="${EVAL_CPUS}" --time="${time}" --mail-type=ALL --mail-user="${MAIL_USER}")
  [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
  args+=(--export="ALL,${exports},PIPELINE_TRACKER_JSON=${TRACKER_JSON},PIPELINE_STAGE=${stage},TRACKED_STAGE_SCRIPT=${script}")
  local jid; jid="$(jobid "$(sbatch "${args[@]}" "${TRACKED_WRAPPER}")")"
  register_stage "${stage}" "${desc}" "${kind}" "${jid}" "${name}" "${dep}" "${PROJECT_DIR}/out/${name}_${jid}.out" "${artifact}" "${checkpoint}" "${results}"
  printf '%s' "${jid}"
}
record() { printf '%s\t%s\n' "$1" "$2" >> "${LEDGER}"; }
: > "${LEDGER}"
printf '# branch=%s\n# commit=%s\n# backend=%s\n# run_prefix=%s\n# tracker=%s\n# bridge_ready_job=%s\n' "${BRANCH}" "${COMMIT}" "${BACKEND}" "${RUN_PREFIX}" "${TRACKER_MD}" "${BRIDGE_READY_JOB_ID}" >> "${LEDGER}"

echo "Submitting tracked pipeline: branch=${BRANCH} backend=${BACKEND} commit=${COMMIT} run=${RUN_PREFIX}"

# S1 starts immediately. It is intentionally independent of Bridge D0.
J1=$(submit_gpu_train S1 "" "${RUN_PREFIX}_S1_synth" "2-00:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},JOB_ID=${SYNTH_JOB_ID},DATA_DIR=${SYNTH_DATA_DIR},EPOCHS=${SYNTH_EPOCHS},LEARNING_RATE=1e-4,NUM_NEGATIVES=10,SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4,NUM_GPUS=${TRAIN_GPUS}" \
  "${PROJECT_DIR}/scripts/train/run_branch_fixed63_synthetic.sh" \
  "Synthetic pretraining" "${PROJECT_DIR}/Weights/${SYNTH_JOB_ID}" "${SYNTH_CKPT}" "")
record S1_SYNTH "$J1"

J2=$(submit_gpu_eval S2 "$J1" "${RUN_PREFIX}_S2_synth_qual" "04:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},WEIGHTS=${SYNTH_CKPT},RUN_TAG=${RUN_PREFIX}/s2_synth_qualitative,RESULTS_ROOT=${RESULTS_BASE}/s2_synth_qualitative" \
  "${PROJECT_DIR}/scripts/eval/run_stage_qualitative.sh" qualitative \
  "Synthetic checkpoint qualitative real evaluation" "" "${SYNTH_CKPT}" "${RESULTS_BASE}/s2_synth_qualitative")
record S2_SYNTH_QUAL "$J2"

J3=$(submit_gpu_eval S3 "$J2" "${RUN_PREFIX}_S3_synth_quant" "10:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${SYNTH_CKPT},RUN_TAG=${RUN_PREFIX}_s3_synth_quantitative,RESULTS_ROOT=${RESULTS_BASE}/s3_synth_quantitative" \
  "${PROJECT_DIR}/scripts/eval/run_stage_quantitative.sh" quantitative \
  "Synthetic checkpoint quantitative zero-shot real evaluation" "" "${SYNTH_CKPT}" "${RESULTS_BASE}/s3_synth_quantitative")
record S3_SYNTH_QUANT "$J3"

# S4 is the join point. In concurrent D0 mode, Slurm requires BOTH S3 and D0 to
# finish successfully: --dependency=afterok:<S3>:<D0>.
J4_DEP="$J3"
if [[ -n "${BRIDGE_READY_JOB_ID}" ]]; then
  J4_DEP="${J3}:${BRIDGE_READY_JOB_ID}"
fi
J4=$(submit_gpu_eval S4 "$J4_DEP" "${RUN_PREFIX}_S4_bridge_pre" "08:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${SYNTH_CKPT},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},RUN_TAG=${RUN_PREFIX}_s4_bridge_pretrain,RESULTS_ROOT=${RESULTS_BASE}/s4_bridge_pretrain" \
  "${PROJECT_DIR}/scripts/eval/run_stage_bridge_eval.sh" bridge_eval \
  "Bridge V2 evaluation before bridge fine-tuning" "${BRIDGE_DATA_DIR}" "${SYNTH_CKPT}" "${RESULTS_BASE}/s4_bridge_pretrain")
record S4_BRIDGE_PRE_EVAL "$J4"

J5=$(submit_gpu_train S5 "$J4" "${RUN_PREFIX}_S5_bridge_train" "2-00:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},JOB_ID=${BRIDGE_JOB_ID},DATA_DIR=${BRIDGE_DATA_DIR},PRETRAINED_WEIGHTS=${SYNTH_CKPT},EPOCHS=${BRIDGE_EPOCHS},LEARNING_RATE=${BRIDGE_LR},NUM_NEGATIVES=10,SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4,NUM_GPUS=${TRAIN_GPUS}" \
  "${PROJECT_DIR}/scripts/train/run_real_synthetic_bridge.sh" \
  "Direct Bridge V2 real-domain fine-tuning" "${PROJECT_DIR}/Weights/${BRIDGE_JOB_ID}" "${BRIDGE_BEST}" "")
record S5_BRIDGE_TRAIN "$J5"

J6=$(submit_gpu_eval S6 "$J5" "${RUN_PREFIX}_S6_post_qual" "04:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},WEIGHTS=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}/s6_post_bridge_qualitative,RESULTS_ROOT=${RESULTS_BASE}/s6_post_bridge_qualitative" \
  "${PROJECT_DIR}/scripts/eval/run_stage_qualitative.sh" qualitative \
  "Post-Bridge qualitative real evaluation" "" "${BRIDGE_BEST}" "${RESULTS_BASE}/s6_post_bridge_qualitative")
record S6_POST_BRIDGE_QUAL "$J6"

J7=$(submit_gpu_eval S7A "$J6" "${RUN_PREFIX}_S7_post_quant" "10:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}_s7_post_bridge_quantitative,RESULTS_ROOT=${RESULTS_BASE}/s7_post_bridge_quantitative" \
  "${PROJECT_DIR}/scripts/eval/run_stage_quantitative.sh" quantitative \
  "Post-Bridge quantitative real evaluation" "" "${BRIDGE_BEST}" "${RESULTS_BASE}/s7_post_bridge_quantitative")
record S7_POST_BRIDGE_QUANT "$J7"

J8=$(submit_gpu_eval S7B "$J7" "${RUN_PREFIX}_S7_bridge_post" "08:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},RUN_TAG=${RUN_PREFIX}_s7_bridge_posttrain,RESULTS_ROOT=${RESULTS_BASE}/s7_bridge_posttrain" \
  "${PROJECT_DIR}/scripts/eval/run_stage_bridge_eval.sh" bridge_eval \
  "Bridge V2 positive/negative evaluation after fine-tuning" "${BRIDGE_DATA_DIR}" "${BRIDGE_BEST}" "${RESULTS_BASE}/s7_bridge_posttrain")
record S7_BRIDGE_POST_EVAL "$J8"

J9=$(submit_gpu_eval S8 "$J8" "${RUN_PREFIX}_S8_final" "1-00:00:00" \
  "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}_s8_final_all_real,RESULTS_ROOT=${RESULTS_BASE}/s8_final_all_real,FINAL_THRESHOLD=${FINAL_THRESHOLD}" \
  "${PROJECT_DIR}/scripts/eval/run_stage_final_all_real.sh" final \
  "Final frozen evaluation on every real manifest relationship" "" "${BRIDGE_BEST}" "${RESULTS_BASE}/s8_final_all_real")
record S8_FINAL_ALL_REAL "$J9"

cat <<EOF
=== TRACKED MODEL PIPELINE SUBMITTED ===
branch=${BRANCH}
commit=${COMMIT}
backend=${BACKEND}
run_prefix=${RUN_PREFIX}
bridge_dataset=${BRIDGE_DATA_DIR}
bridge_ready_job=${BRIDGE_READY_JOB_ID:-already-built}
bridge_epochs_max=${BRIDGE_EPOCHS}
bridge_lr=${BRIDGE_LR}
tracker=${TRACKER_MD}
tracker_state=${TRACKER_JSON}
ledger=${LEDGER}
final_job=${J9}
results=${RESULTS_BASE}

Dependency graph:
  S1 -> S2 -> S3
  S3 + D0(${BRIDGE_READY_JOB_ID:-already-built}) -> S4 -> S5 -> S6 -> S7A -> S7B -> S8

Open this file whenever you want the current job/result map:
  ${TRACKER_MD}

Refresh from SLURM manually if a node/time-limit kill prevented an automatic update:
  python scripts/pipeline/experiment_tracker.py refresh --tracker ${TRACKER_JSON}

Dependencies use afterok. A failed prerequisite blocks its downstream jobs.
EOF
cat "${LEDGER}"
