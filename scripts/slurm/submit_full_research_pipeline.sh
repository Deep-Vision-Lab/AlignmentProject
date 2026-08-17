#!/usr/bin/env bash
# Submit the complete research curriculum as one strict afterok SLURM chain.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p out logs

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
fi

SYNTH_DATA_DIR="${SYNTH_DATA_DIR:-${PROJECT_DIR}/DataSet/AugmentedArabicDataset63}"
REAL_DATA_DIR="${REAL_DATA_DIR:-${PROJECT_DIR}/DataSet/ArabicDataset}"
BRIDGE_DATA_DIR="${BRIDGE_DATA_DIR:-${PROJECT_DIR}/DataSet/RealSyntheticBridge_v2}"
[[ -d "${SYNTH_DATA_DIR}/images" && -d "${SYNTH_DATA_DIR}/texts" ]] || { echo "ERROR: missing synthetic data ${SYNTH_DATA_DIR}" >&2; exit 2; }
[[ -s "${REAL_DATA_DIR}/dataset_manifest.jsonl" ]] || { echo "ERROR: missing real manifest ${REAL_DATA_DIR}/dataset_manifest.jsonl" >&2; exit 2; }

RUN_PREFIX="${RUN_PREFIX:-${BACKEND}_research_$(date +%Y%m%d_%H%M%S)}"
SYNTH_JOB_ID="${RUN_PREFIX}_synth"
REAL_JOB_ID="${RUN_PREFIX}_real"
BRIDGE_JOB_ID="${RUN_PREFIX}_bridge_v2"
SYNTH_CKPT="${PROJECT_DIR}/Weights/${SYNTH_JOB_ID}/checkpoint_latest.pth"
REAL_CKPT="${PROJECT_DIR}/Weights/${REAL_JOB_ID}/checkpoint_latest.pth"
BRIDGE_BEST="${PROJECT_DIR}/Weights/${BRIDGE_JOB_ID}/checkpoint_best_val.pth"
RESULTS_BASE="${PROJECT_DIR}/Results/Evaluation/ResearchPipeline/${RUN_PREFIX}"
LEDGER="${PROJECT_DIR}/logs/research_pipeline_${RUN_PREFIX}.jobs"

if [[ "${ALLOW_EXISTING_WEIGHTS:-0}" != "1" ]]; then
  for d in "${PROJECT_DIR}/Weights/${SYNTH_JOB_ID}" "${PROJECT_DIR}/Weights/${REAL_JOB_ID}" "${PROJECT_DIR}/Weights/${BRIDGE_JOB_ID}"; do
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
CPU_PARTITION="${CPU_PARTITION:-main}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
SYNTH_EPOCHS="${SYNTH_EPOCHS:-20}"
REAL_EPOCHS="${REAL_EPOCHS:-5}"
BRIDGE_EPOCHS="${BRIDGE_EPOCHS:-8}"
FINAL_THRESHOLD="${FINAL_THRESHOLD:-0.50}"
REBUILD_BRIDGE="${REBUILD_BRIDGE:-0}"

jobid() { cut -d';' -f1 <<<"$1"; }
submit_gpu_train() {
  local dep="$1" name="$2" time="$3" exports="$4" script="$5"
  local args=(--parsable --partition="${TRAIN_PARTITION}" --job-name="${name}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --gpus="${GPU_RESOURCE}:${TRAIN_GPUS}" --ntasks=1 --cpus-per-task="${TRAIN_CPUS}" --mem="${TRAIN_MEM}" --time="${time}" --mail-type=ALL --mail-user="${MAIL_USER}")
  [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
  args+=(--export="ALL,${exports}")
  jobid "$(sbatch "${args[@]}" "${script}")"
}
submit_gpu_eval() {
  local dep="$1" name="$2" time="$3" exports="$4" script="$5"
  local args=(--parsable --partition="${EVAL_PARTITION}" --job-name="${name}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --gpus="${GPU_RESOURCE}:1" --ntasks=1 --cpus-per-task="${EVAL_CPUS}" --time="${time}" --mail-type=ALL --mail-user="${MAIL_USER}")
  [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
  args+=(--export="ALL,${exports}")
  jobid "$(sbatch "${args[@]}" "${script}")"
}
submit_cpu() {
  local dep="$1" name="$2" time="$3" exports="$4" script="$5"
  local args=(--parsable --partition="${CPU_PARTITION}" --job-name="${name}" --output="${PROJECT_DIR}/out/%x_%J.out" --chdir="${PROJECT_DIR}" --ntasks=1 --cpus-per-task=4 --mem=16G --time="${time}" --mail-type=ALL --mail-user="${MAIL_USER}")
  [[ -n "${dep}" ]] && args+=(--dependency="afterok:${dep}")
  args+=(--export="ALL,${exports}")
  jobid "$(sbatch "${args[@]}" "${script}")"
}
record() { printf '%s\t%s\n' "$1" "$2" >> "${LEDGER}"; }
: > "${LEDGER}"
printf '# branch=%s\n# commit=%s\n# backend=%s\n# run_prefix=%s\n' "${BRANCH}" "${COMMIT}" "${BACKEND}" "${RUN_PREFIX}" >> "${LEDGER}"

echo "Submitting research pipeline: branch=${BRANCH} backend=${BACKEND} commit=${COMMIT} run=${RUN_PREFIX}"

J1=$(submit_gpu_train "" "${RUN_PREFIX}_S1_synth" "2-00:00:00" "PROJECT_DIR=${PROJECT_DIR},JOB_ID=${SYNTH_JOB_ID},DATA_DIR=${SYNTH_DATA_DIR},EPOCHS=${SYNTH_EPOCHS},LEARNING_RATE=1e-4,NUM_NEGATIVES=10,SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4,NUM_GPUS=${TRAIN_GPUS}" "${PROJECT_DIR}/scripts/train/run_branch_fixed63_synthetic.sh")
record S1_SYNTH "$J1"

J2=$(submit_gpu_eval "$J1" "${RUN_PREFIX}_S2_synth_qual" "04:00:00" "PROJECT_DIR=${PROJECT_DIR},WEIGHTS=${SYNTH_CKPT},RUN_TAG=${RUN_PREFIX}/s2_synth_qualitative,RESULTS_ROOT=${RESULTS_BASE}/s2_synth_qualitative" "${PROJECT_DIR}/scripts/eval/run_stage_qualitative.sh")
record S2_SYNTH_QUAL "$J2"

J3=$(submit_gpu_eval "$J2" "${RUN_PREFIX}_S3_synth_quant" "10:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${SYNTH_CKPT},RUN_TAG=${RUN_PREFIX}_s3_synth_quantitative,RESULTS_ROOT=${RESULTS_BASE}/s3_synth_quantitative" "${PROJECT_DIR}/scripts/eval/run_stage_quantitative.sh")
record S3_SYNTH_QUANT "$J3"

J4=$(submit_gpu_train "$J3" "${RUN_PREFIX}_S4_real" "1-00:00:00" "PROJECT_DIR=${PROJECT_DIR},JOB_ID=${REAL_JOB_ID},PRETRAINED_WEIGHTS=${SYNTH_CKPT},DATA_DIR=${REAL_DATA_DIR},EPOCHS=${REAL_EPOCHS},LEARNING_RATE=2e-6,NUM_GPUS=${TRAIN_GPUS}" "${PROJECT_DIR}/scripts/train/run_stage_real_finetune.sh")
record S4_REAL_TRAIN "$J4"

J5=$(submit_gpu_eval "$J4" "${RUN_PREFIX}_S5_real_qual" "04:00:00" "PROJECT_DIR=${PROJECT_DIR},WEIGHTS=${REAL_CKPT},RUN_TAG=${RUN_PREFIX}/s5_real_qualitative,RESULTS_ROOT=${RESULTS_BASE}/s5_real_qualitative" "${PROJECT_DIR}/scripts/eval/run_stage_qualitative.sh")
record S5_REAL_QUAL "$J5"

J6=$(submit_gpu_eval "$J5" "${RUN_PREFIX}_S6_real_quant" "10:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${REAL_CKPT},RUN_TAG=${RUN_PREFIX}_s6_real_quantitative,RESULTS_ROOT=${RESULTS_BASE}/s6_real_quantitative" "${PROJECT_DIR}/scripts/eval/run_stage_quantitative.sh")
record S6_REAL_QUANT "$J6"

J7=$(submit_cpu "$J6" "${RUN_PREFIX}_S7_bridge_build" "1-00:00:00" "PROJECT_DIR=${PROJECT_DIR},REAL_DATA_DIR=${REAL_DATA_DIR},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},REBUILD_BRIDGE=${REBUILD_BRIDGE}" "${PROJECT_DIR}/scripts/data/prepare_real_synthetic_bridge_v2.sh")
record S7_BRIDGE_BUILD "$J7"

J8=$(submit_gpu_eval "$J7" "${RUN_PREFIX}_S8_bridge_pre" "08:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${REAL_CKPT},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},RUN_TAG=${RUN_PREFIX}_s8_bridge_pretrain_eval,RESULTS_ROOT=${RESULTS_BASE}/s8_bridge_pretrain_eval" "${PROJECT_DIR}/scripts/eval/run_stage_bridge_eval.sh")
record S8_BRIDGE_PRE_EVAL "$J8"

J9=$(submit_gpu_train "$J8" "${RUN_PREFIX}_S9_bridge_train" "1-00:00:00" "PROJECT_DIR=${PROJECT_DIR},JOB_ID=${BRIDGE_JOB_ID},DATA_DIR=${BRIDGE_DATA_DIR},PRETRAINED_WEIGHTS=${REAL_CKPT},EPOCHS=${BRIDGE_EPOCHS},LEARNING_RATE=7.5e-7,NUM_GPUS=${TRAIN_GPUS}" "${PROJECT_DIR}/scripts/train/run_real_synthetic_bridge.sh")
record S9_BRIDGE_TRAIN "$J9"

J10=$(submit_gpu_eval "$J9" "${RUN_PREFIX}_S10_post_qual" "04:00:00" "PROJECT_DIR=${PROJECT_DIR},WEIGHTS=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}/s10_post_bridge_qualitative,RESULTS_ROOT=${RESULTS_BASE}/s10_post_bridge_qualitative" "${PROJECT_DIR}/scripts/eval/run_stage_qualitative.sh")
record S10_POST_BRIDGE_QUAL "$J10"

J11=$(submit_gpu_eval "$J10" "${RUN_PREFIX}_S10_post_quant" "10:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}_s10_post_bridge_quantitative,RESULTS_ROOT=${RESULTS_BASE}/s10_post_bridge_quantitative" "${PROJECT_DIR}/scripts/eval/run_stage_quantitative.sh")
record S10_POST_BRIDGE_QUANT "$J11"

J12=$(submit_gpu_eval "$J11" "${RUN_PREFIX}_S10_bridge_post" "08:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},BRIDGE_DATA_DIR=${BRIDGE_DATA_DIR},RUN_TAG=${RUN_PREFIX}_s10_bridge_posttrain,RESULTS_ROOT=${RESULTS_BASE}/s10_bridge_posttrain" "${PROJECT_DIR}/scripts/eval/run_stage_bridge_eval.sh")
record S10_BRIDGE_POST_EVAL "$J12"

J13=$(submit_gpu_eval "$J12" "${RUN_PREFIX}_S11_final" "1-00:00:00" "PROJECT_DIR=${PROJECT_DIR},CHECKPOINT=${BRIDGE_BEST},RUN_TAG=${RUN_PREFIX}_s11_final_all_real,RESULTS_ROOT=${RESULTS_BASE}/s11_final_all_real,FINAL_THRESHOLD=${FINAL_THRESHOLD}" "${PROJECT_DIR}/scripts/eval/run_stage_final_all_real.sh")
record S11_FINAL_ALL_REAL "$J13"

cat <<EOF
=== PIPELINE SUBMITTED ===
branch=${BRANCH}
commit=${COMMIT}
backend=${BACKEND}
run_prefix=${RUN_PREFIX}
ledger=${LEDGER}
final_job=${J13}
results=${RESULTS_BASE}

All dependencies use afterok. A failed stage blocks downstream jobs.
Keep this checkout on the same branch while the submitted chain is running.
EOF
cat "${LEDGER}"
