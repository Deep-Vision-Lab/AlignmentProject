#!/usr/bin/env bash
# Submit the full clean synthetic-partner pilot as one strict Slurm dependency chain:
#   preprocess -> train from Stage 1 -> canonical NW eval -> NW discrimination eval.
set -euo pipefail

ROOT="${PROJECT_DIR:-$HOME/BGU-Lab/AlignmentProject}"
ROOT="$(readlink -f "${ROOT}")"
cd "${ROOT}"
mkdir -p out

EXPECTED_BRANCH="agent/cnn-bilstm-partial-overlap"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expected branch ${EXPECTED_BRANCH}, got ${CURRENT_BRANCH:-<detached>}." >&2
  exit 2
fi

# Definitive names for this experiment.
TRAIN_JOB_ID="${TRAIN_JOB_ID:-cnn_real_clean_synpartner_from_stage1_v1}"
PREP_JOB_NAME="${PREP_JOB_NAME:-prep_synpartner_v1}"
NW_JOB_NAME="${NW_JOB_NAME:-nw_synpartner_v1}"
DISC_JOB_NAME="${DISC_JOB_NAME:-disc_synpartner_v1}"

DATA_DIR="${DATA_DIR:-${ROOT}/DataSet/ArabicDataset}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-${ROOT}/DataSet/ArabicDatasetSyntheticPartners}"
SYNTHETIC_MANIFEST="${SYNTHETIC_MANIFEST:-${SYNTHETIC_DIR}/dataset_manifest.jsonl}"
# This is the recorded Stage-1 CNN+BiLSTM synthetic parent.
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${ROOT}/Weights/cnn_bilstm/model_latest.pth}"
TRAINED_WEIGHTS="${ROOT}/Weights/${TRAIN_JOB_ID}/model_latest.pth"

[[ -f "${DATA_DIR}/dataset_manifest.jsonl" ]] || {
  echo "ERROR: missing canonical real manifest: ${DATA_DIR}/dataset_manifest.jsonl" >&2
  exit 2
}
[[ -f "${PRETRAINED_WEIGHTS}" ]] || {
  echo "ERROR: Stage-1 checkpoint not found: ${PRETRAINED_WEIGHTS}" >&2
  echo "Expected the Stage-1 CNN+BiLSTM parent under Weights/cnn_bilstm/model_latest.pth." >&2
  exit 2
}

# Guard against accidentally supplying a later real-stage checkpoint.
lower_weights="${PRETRAINED_WEIGHTS,,}"
case "${lower_weights}" in
  *phase3*|*stage2*|*from_r2*|*real_aug*|*joint_real*)
    echo "ERROR: PRETRAINED_WEIGHTS looks like a later real-data checkpoint:" >&2
    echo "  ${PRETRAINED_WEIGHTS}" >&2
    echo "This pipeline is explicitly Stage 1 -> clean real synthetic-partner training." >&2
    exit 2
    ;;
esac

CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

# Pilot training settings. Keep this short until the evaluation gate passes.
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_GPUS="${NUM_GPUS:-2}"
EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE:-64}"
REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH:-0}"
NUM_NEGATIVES="${NUM_NEGATIVES:-10}"
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE:-4}"

# Preprocessing settings.
PREP_PARTITION="${PREP_PARTITION:-main}"
PREP_CPUS="${PREP_CPUS:-8}"
PREP_MEMORY="${PREP_MEMORY:-64G}"
PREP_TIME="${PREP_TIME:-1-00:00:00}"
SEED="${SEED:-42}"
MIN_REGIONS="${MIN_REGIONS:-1}"
MAX_REGIONS="${MAX_REGIONS:-3}"
MAX_RUN_BOXES="${MAX_RUN_BOXES:-3}"
MIN_CHARS="${MIN_CHARS:-3}"
MAX_CHARS="${MAX_CHARS:-28}"
WIDTH_RATIO_MIN="${WIDTH_RATIO_MIN:-0.40}"
WIDTH_RATIO_MAX="${WIDTH_RATIO_MAX:-2.50}"
MULTI_REGION_PROB="${MULTI_REGION_PROB:-0.65}"
THREE_REGION_PROB="${THREE_REGION_PROB:-0.15}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-120}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-20}"

# GPU stages.
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
TRAIN_CPUS="${TRAIN_CPUS:-$((8 * NUM_GPUS))}"
TRAIN_MEMORY="${TRAIN_MEMORY:-96G}"
TRAIN_TIME="${TRAIN_TIME:-1-00:00:00}"
EVAL_CPUS="${EVAL_CPUS:-4}"
EVAL_MEMORY="${EVAL_MEMORY:-32G}"
NW_TIME="${NW_TIME:-12:00:00}"
DISC_TIME="${DISC_TIME:-12:00:00}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-100000}"

NW_RUN_TAG="${NW_RUN_TAG:-${TRAIN_JOB_ID}_local_mutualz}"
NW_RESULTS_ROOT="${NW_RESULTS_ROOT:-${ROOT}/Results/Evaluation/NW/Real/${NW_RUN_TAG}}"
DISC_RUN_TAG="${DISC_RUN_TAG:-${TRAIN_JOB_ID}}"
DISC_RESULTS_ROOT="${DISC_RESULTS_ROOT:-${ROOT}/Results/Evaluation/Representation_Diagnostics/${DISC_RUN_TAG}}"

for value_name in EPOCHS NUM_GPUS EFFECTIVE_GLOBAL_BATCH_SIZE NUM_NEGATIVES SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${value_name} must be a positive integer, got ${value}." >&2
    exit 2
  }
done

printf '%s\n' \
  "=== CLEAN SYNTHETIC-PARTNER STAGE-1 PIPELINE ===" \
  "branch=${CURRENT_BRANCH}" \
  "commit=$(git rev-parse --short HEAD)" \
  "Stage-1 parent=${PRETRAINED_WEIGHTS}" \
  "synthetic output=${SYNTHETIC_DIR}" \
  "training job/weights=${TRAIN_JOB_ID}" \
  "trained weights=${TRAINED_WEIGHTS}" \
  "epochs=${EPOCHS}" \
  "learning_rate=${LEARNING_RATE}" \
  "generic augmentation=OFF" \
  "natural train mixture=clean positives + synthetic partners + clean no-shared negatives" \
  "NW protocol=local + mutual-z + checkpoint stride (32px window / expected 8px stride)" \
  "dependency policy=afterok for every stage"

# 1) CPU preprocessing + smoke test.
PREP_SUBMIT="$(sbatch --parsable \
  --job-name="${PREP_JOB_NAME}" \
  --partition="${PREP_PARTITION}" \
  --ntasks=1 \
  --cpus-per-task="${PREP_CPUS}" \
  --mem="${PREP_MEMORY}" \
  --time="${PREP_TIME}" \
  --output="${ROOT}/out/%x_%j.out" \
  --mail-type=ALL \
  --mail-user="${MAIL_USER}" \
  --export=ALL,SYNTHETIC_PARTNER_OFFLINE_WORKER=1,PROJECT_DIR="${ROOT}",DATA_DIR="${DATA_DIR}",OUTPUT_DIR="${SYNTHETIC_DIR}",CONDA_ENV="${CONDA_ENV}",SEED="${SEED}",MIN_REGIONS="${MIN_REGIONS}",MAX_REGIONS="${MAX_REGIONS}",MAX_RUN_BOXES="${MAX_RUN_BOXES}",MIN_CHARS="${MIN_CHARS}",MAX_CHARS="${MAX_CHARS}",WIDTH_RATIO_MIN="${WIDTH_RATIO_MIN}",WIDTH_RATIO_MAX="${WIDTH_RATIO_MAX}",MULTI_REGION_PROB="${MULTI_REGION_PROB}",THREE_REGION_PROB="${THREE_REGION_PROB}",MAX_ATTEMPTS="${MAX_ATTEMPTS}",SMOKE_SAMPLES="${SMOKE_SAMPLES}" \
  "${ROOT}/scripts/data/build_no_shared_synthetic_partners_offline.sh")"
PREP_SLURM_ID="${PREP_SUBMIT%%;*}"

# 2) GPU training, only if preprocessing/smoke succeeded.
TRAIN_SUBMIT="$(sbatch --parsable \
  --dependency="afterok:${PREP_SLURM_ID}" \
  --job-name="${TRAIN_JOB_ID}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:${NUM_GPUS}" \
  --ntasks=1 \
  --cpus-per-task="${TRAIN_CPUS}" \
  --mem="${TRAIN_MEMORY}" \
  --time="${TRAIN_TIME}" \
  --output="${ROOT}/out/%x_%j.out" \
  --mail-type=ALL \
  --mail-user="${MAIL_USER}" \
  --export=ALL,PROJECT_DIR="${ROOT}",DATA_DIR="${DATA_DIR}",CONDA_ENV="${CONDA_ENV}",PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS}",JOB_ID="${TRAIN_JOB_ID}",EPOCHS="${EPOCHS}",LEARNING_RATE="${LEARNING_RATE}",NUM_GPUS="${NUM_GPUS}",EFFECTIVE_GLOBAL_BATCH_SIZE="${EFFECTIVE_GLOBAL_BATCH_SIZE}",REAL_TRAIN_SAMPLES_PER_EPOCH="${REAL_TRAIN_SAMPLES_PER_EPOCH}",REAL_SYNTHETIC_PARTNER_MANIFEST="${SYNTHETIC_MANIFEST}",NUM_NEGATIVES="${NUM_NEGATIVES}",SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE="${SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE}",AUGMENT=0,REAL_AUGMENT=0,REAL_AUG_STITCH_PROB=0,REAL_EXTRA_EXCLUDE_EVAL_PAGES=1,SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0 \
  "${ROOT}/scripts/train/run_real_finetune_partial_overlap.sh")"
TRAIN_SLURM_ID="${TRAIN_SUBMIT%%;*}"

# 3) Canonical held-out real NW evaluation, same protocol as the baseline.
NW_SUBMIT="$(sbatch --parsable \
  --dependency="afterok:${TRAIN_SLURM_ID}" \
  --job-name="${NW_JOB_NAME}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --ntasks=1 \
  --cpus-per-task="${EVAL_CPUS}" \
  --mem="${EVAL_MEMORY}" \
  --time="${NW_TIME}" \
  --output="${ROOT}/out/%x_%j.out" \
  --mail-type=ALL \
  --mail-user="${MAIL_USER}" \
  --export=ALL,PROJECT_DIR="${ROOT}",WEIGHTS="${TRAINED_WEIGHTS}",REAL_DATA_DIR="${DATA_DIR}",CONDA_ENV="${CONDA_ENV}",RUN_TAG="${NW_RUN_TAG}",RESULTS_ROOT="${NW_RESULTS_ROOT}",N_SAMPLES="${EVAL_N_SAMPLES}",START_INDEX=1,SPLIT_SEED=42,FEATURE=local,SCORE_MODE=mutual-z,SCORE_CLIP=4.0,THRESHOLD=0.45,GAP=-0.30,HEATMAP_SOURCE=dp-score,EVAL_JOB_NAME="${NW_JOB_NAME}" \
  "${ROOT}/Evaluation/evaluate_nw_real.sh")"
NW_SLURM_ID="${NW_SUBMIT%%;*}"

# 4) Same-NW positive-vs-no-shared discrimination, only after canonical NW passes.
DISC_SUBMIT="$(sbatch --parsable \
  --dependency="afterok:${NW_SLURM_ID}" \
  --job-name="${DISC_JOB_NAME}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --ntasks=1 \
  --cpus-per-task="${EVAL_CPUS}" \
  --mem="${EVAL_MEMORY}" \
  --time="${DISC_TIME}" \
  --output="${ROOT}/out/%x_%j.out" \
  --mail-type=ALL \
  --mail-user="${MAIL_USER}" \
  --export=ALL,PROJECT_DIR="${ROOT}",WEIGHTS="${TRAINED_WEIGHTS}",REAL_DATA_DIR="${DATA_DIR}",CONDA_ENV="${CONDA_ENV}",RUN_TAG="${DISC_RUN_TAG}",RESULTS_ROOT="${DISC_RESULTS_ROOT}",N_SAMPLES="${EVAL_N_SAMPLES}",START_INDEX=1,DATASET_SPLIT_SEED=42,FEATURE=local,SCORE_MODE=mutual-z,SCORE_CLIP=4.0,THRESHOLD=0.45,GAP=-0.30,HEATMAP_SOURCE=match-score,REAL_REQUIRE_BOX_ANNOTATIONS=0,EVAL_JOB_NAME="${DISC_JOB_NAME}" \
  "${ROOT}/Evaluation/evaluate_synpartner_nw_discrimination.sh")"
DISC_SLURM_ID="${DISC_SUBMIT%%;*}"

STATE_FILE="${ROOT}/out/${TRAIN_JOB_ID}_pipeline_jobs.txt"
cat > "${STATE_FILE}" <<EOF
PIPELINE_BRANCH=${CURRENT_BRANCH}
PIPELINE_COMMIT=$(git rev-parse HEAD)
STAGE1_WEIGHTS=${PRETRAINED_WEIGHTS}
PREPROCESS_JOB_NAME=${PREP_JOB_NAME}
PREPROCESS_SLURM_ID=${PREP_SLURM_ID}
TRAIN_JOB_ID=${TRAIN_JOB_ID}
TRAIN_SLURM_ID=${TRAIN_SLURM_ID}
TRAINED_WEIGHTS=${TRAINED_WEIGHTS}
NW_JOB_NAME=${NW_JOB_NAME}
NW_SLURM_ID=${NW_SLURM_ID}
NW_RESULTS=${NW_RESULTS_ROOT}
DISC_JOB_NAME=${DISC_JOB_NAME}
DISC_SLURM_ID=${DISC_SLURM_ID}
DISC_RESULTS=${DISC_RESULTS_ROOT}
EOF

printf '%s\n' \
  "" \
  "=== PIPELINE SUBMITTED ===" \
  "1. ${PREP_JOB_NAME}                  ${PREP_SLURM_ID}" \
  "2. ${TRAIN_JOB_ID}                  ${TRAIN_SLURM_ID}  afterok:${PREP_SLURM_ID}" \
  "3. ${NW_JOB_NAME}                   ${NW_SLURM_ID}  afterok:${TRAIN_SLURM_ID}" \
  "4. ${DISC_JOB_NAME}                 ${DISC_SLURM_ID}  afterok:${NW_SLURM_ID}" \
  "" \
  "job record=${STATE_FILE}" \
  "weights=${TRAINED_WEIGHTS}" \
  "NW results=${NW_RESULTS_ROOT}" \
  "discrimination results=${DISC_RESULTS_ROOT}" \
  "" \
  "Monitor:" \
  "  squeue -j ${PREP_SLURM_ID},${TRAIN_SLURM_ID},${NW_SLURM_ID},${DISC_SLURM_ID} -o '%.18i %.42j %.2t %.12M %.60R'"
