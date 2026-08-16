#!/usr/bin/env bash
# Resume conservative real adaptation from an already-trained/evaluated R0.
# Re-runs only the corrected lightweight R0 gate locally, then submits R1->R2->final.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

R0_NAME="${R0_NAME:-cnn_real_r0_image_text}"
R1_NAME="${R1_NAME:-cnn_real_r1_positive_pairs}"
R2_NAME="${R2_NAME:-cnn_real_r2_full_discrimination}"

cd "${PROJECT_DIR}"
mkdir -p out

R0_CHECKPOINT="${PROJECT_DIR}/Weights/${R0_NAME}/model_latest.pth"
R1_CHECKPOINT="${PROJECT_DIR}/Weights/${R1_NAME}/model_latest.pth"
R2_CHECKPOINT="${PROJECT_DIR}/Weights/${R2_NAME}/model_latest.pth"
PREPROCESS_ENV="${PROJECT_DIR}/Results/Evaluation/RealAdaptationOvernight/preprocessing.env"
R0_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R0_NAME}_discrimination"
R1_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination"
R2_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination"
PHASE3_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/cnn_phase3_adaptation_reference_discrimination"
FULL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest_full_pairs.jsonl"

for required in "${R0_CHECKPOINT}" "${PREPROCESS_ENV}" "${FULL_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 2; }
done
[[ -d "${R0_ROOT}" ]] || { echo "ERROR: missing R0 evaluation ${R0_ROOT}" >&2; exit 2; }
[[ -d "${PHASE3_ROOT}" ]] || { echo "ERROR: missing Phase-3 reference ${PHASE3_ROOT}" >&2; exit 2; }

source "${PREPROCESS_ENV}"
: "${STAGE1_BASELINE_ROOT:?preprocessing.env is missing STAGE1_BASELINE_ROOT}"
[[ -d "${STAGE1_BASELINE_ROOT}" ]] || {
  echo "ERROR: missing selected Stage-1 baseline ${STAGE1_BASELINE_ROOT}" >&2
  exit 2
}

# Prevent accidental duplicate continuation chains.
for name in "${R1_NAME}" eval_gate_real_r1 "${R2_NAME}" eval_gate_real_r2 final_real_adaptation_eval; do
  if squeue -h -u "${USER}" -n "${name}" | grep -q .; then
    echo "ERROR: active job already exists with name ${name}; refusing duplicate submission." >&2
    exit 2
  fi
done

# Corrected R0 gate is fast and uses existing CSVs; no GPU is needed.
echo "=== RECHECK R0 WITH RELATIVE PRESERVATION GATE ==="
python scripts/eval/check_real_discrimination_preservation.py \
  "${R0_ROOT}" \
  --baseline-root "${STAGE1_BASELINE_ROOT}"

echo "R0 accepted. Submitting continuation from existing checkpoint: ${R0_CHECKPOINT}"

R1_TRAIN_JOB_ID="$(sbatch --parsable \
  --job-name="${R1_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=18:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${R1_NAME}' PRETRAINED_WEIGHTS='${R0_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=1e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl'; bash scripts/train/run_real_positive_pair_adaptation.sh")"

R1_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R1_TRAIN_JOB_ID}" \
  --job-name=eval_gate_real_r1 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R1_CHECKPOINT}'; CHECKPOINT='${R1_CHECKPOINT}' RUN_NAME='${R1_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py '${R1_ROOT}' --baseline-root '${R0_ROOT}' --min-positive-steps 2.0 --min-step-gap 0.5")"

R2_TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R1_EVAL_JOB_ID}" \
  --job-name="${R2_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=20:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${R2_NAME}' PRETRAINED_WEIGHTS='${R1_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=1e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl' REAL_TRAIN_SAMPLES_PER_EPOCH=6000 REAL_CLEAN_VIEWS_PER_CYCLE=1 REAL_AUG_VIEWS_PER_CYCLE=1 SEQUENCE_RANKING_WEIGHT=0.03 IMAGE_PAIR_LOSS_WEIGHT=0.08 SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0.015; bash scripts/train/run_real_full_pair_discrimination.sh")"

R2_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R2_TRAIN_JOB_ID}" \
  --job-name=eval_gate_real_r2 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R2_CHECKPOINT}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py '${R2_ROOT}' --baseline-root '${R1_ROOT}'; python scripts/eval/check_real_discrimination_gate.py '${R2_ROOT}' --baseline-root '${PHASE3_ROOT}'")"

FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R2_EVAL_JOB_ID}" \
  --job-name=final_real_adaptation_eval \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== SELECTED STAGE1 ==='; python scripts/eval/summarize_real_discrimination.py \"\${STAGE1_BASELINE_ROOT}\"; echo '=== PHASE3 ==='; python scripts/eval/summarize_real_discrimination.py '${PHASE3_ROOT}'; echo '=== R0 ==='; python scripts/eval/summarize_real_discrimination.py '${R0_ROOT}'; echo '=== R1 ==='; python scripts/eval/summarize_real_discrimination.py '${R1_ROOT}'; echo '=== R2 ==='; python scripts/eval/summarize_real_discrimination.py '${R2_ROOT}'; echo '=== FINAL N=100 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_full_discrimination'")"

cat <<EOF
Submitted continuation from accepted R0:
  R1 TRAIN        ${R1_TRAIN_JOB_ID}  ${R1_NAME}
  R1 EVAL+GATE    ${R1_EVAL_JOB_ID}
  R2 TRAIN        ${R2_TRAIN_JOB_ID}  ${R2_NAME}
  R2 EVAL+GATES   ${R2_EVAL_JOB_ID}
  FINAL N=100     ${FINAL_JOB_ID}

R1 begins immediately. Every later job uses afterok and remains blocked if the preceding scientific gate fails.
EOF
