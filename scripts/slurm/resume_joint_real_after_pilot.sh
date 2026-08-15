#!/usr/bin/env bash
# Resume the joint-real experiment after a completed 5-epoch pilot.
# Starts directly from EVAL5 because the pilot checkpoint/configuration has already
# been validated by the previous CHECK5 log.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PILOT_NAME="${PILOT_NAME:-cnn_joint_real_from_stage1_v1}"
CONT_NAME="${CONT_NAME:-cnn_joint_real_from_stage1_v1_cont10}"
CPU_PARTITION="${CPU_PARTITION:-main}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

cd "${PROJECT_DIR}"
mkdir -p out

test -f "Weights/${PILOT_NAME}/model_latest.pth" || {
  echo "ERROR: missing pilot checkpoint Weights/${PILOT_NAME}/model_latest.pth" >&2
  exit 2
}

echo "Using completed pilot checkpoint: Weights/${PILOT_NAME}/model_latest.pth"

submit_cpu() {
  local dependency="$1" name="$2" wrap="$3"
  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --job-name="${name}" \
    --output="${PROJECT_DIR}/out/%x_%J.out" \
    --chdir="${PROJECT_DIR}" \
    --partition="${CPU_PARTITION}" \
    --tasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --mail-type=ALL --mail-user="${MAIL_USER}" \
    --wrap="${wrap}"
}

# Start immediately with the fixed-manifest 5-epoch evaluation.
EVAL5_JOB_ID="$(sbatch --parsable \
  --job-name=eval_joint_real_5ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; CHECKPOINT='${PROJECT_DIR}/Weights/${PILOT_NAME}/model_latest.pth' RUN_NAME='${PILOT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh")"

GATE5_JOB_ID="$(submit_cpu "${EVAL5_JOB_ID}" gate_joint_real_5ep "set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${PILOT_NAME}'")"

TRAIN10_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${GATE5_JOB_ID}" \
  --job-name="${CONT_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=1-00:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --export=ALL,JOB_ID="${CONT_NAME}",PRETRAINED_WEIGHTS="${PROJECT_DIR}/Weights/${PILOT_NAME}/model_latest.pth",EPOCHS=10,LEARNING_RATE=5e-6,NUM_GPUS=2,EFFECTIVE_GLOBAL_BATCH_SIZE=64,REAL_TRAIN_FRACTION=0.80,REAL_VALID_FRACTION=0.10,REAL_CLEAN_VIEWS_PER_CYCLE=1,REAL_AUG_VIEWS_PER_CYCLE=2,REAL_EFFECTIVE_EPOCH_MULTIPLIER=6,NUM_NEGATIVES=10,SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
  "${PROJECT_DIR}/scripts/train/run_stage1_joint_real_discrimination.sh")"

# No grep|head pipelines under pipefail: use grep -m for the first N matches.
CHECK10_WRAP="set -euo pipefail; cd '${PROJECT_DIR}'; test -f 'Weights/${CONT_NAME}/model_latest.pth'; LOG=\$(ls -t out/${CONT_NAME}_*.out | head -1); echo \"CONT_LOG=\$LOG\"; grep -m 20 -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' \"\$LOG\"; test \$(grep -c 'sequence_batch' \"\$LOG\" || true) -gt 0; echo '=== FIRST SEQUENCE BATCHES ==='; grep -m 10 'sequence_batch' \"\$LOG\"; echo '=== LAST SEQUENCE BATCHES ==='; grep 'sequence_batch' \"\$LOG\" | tail -10"
CHECK10_JOB_ID="$(submit_cpu "${TRAIN10_JOB_ID}" check_joint_real_10ep "${CHECK10_WRAP}")"

EVAL10_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${CHECK10_JOB_ID}" \
  --job-name=eval_joint_real_10ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; CHECKPOINT='${PROJECT_DIR}/Weights/${CONT_NAME}/model_latest.pth' RUN_NAME='${CONT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh")"

GATE10_JOB_ID="$(submit_cpu "${EVAL10_JOB_ID}" gate_joint_real_10ep "set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}'")"

FULL_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${GATE10_JOB_ID}" \
  --job-name=eval_joint_real_final \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; CHECKPOINT='${PROJECT_DIR}/Weights/${CONT_NAME}/model_latest.pth' RUN_NAME='${CONT_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh")"

FINAL_CHECK_JOB_ID="$(submit_cpu "${FULL_EVAL_JOB_ID}" final_check_joint_real "set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate manucripts_align; cd '${PROJECT_DIR}'; echo '=== PILOT 5-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${PILOT_NAME}'; echo; echo '=== CONTINUATION 10-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}'; echo; echo '=== FINAL LARGE EVAL ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_full'; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_full' --no-fail")"

cat <<EOF
Resumed joint-real pipeline after completed pilot:
  EVAL5       ${EVAL5_JOB_ID}
  GATE5       ${GATE5_JOB_ID}
  TRAIN10     ${TRAIN10_JOB_ID}  ${CONT_NAME}
  CHECK10     ${CHECK10_JOB_ID}
  EVAL10      ${EVAL10_JOB_ID}
  GATE10      ${GATE10_JOB_ID}
  FULL_EVAL   ${FULL_EVAL_JOB_ID}
  FINAL_CHECK ${FINAL_CHECK_JOB_ID}

TRAIN10 starts only if GATE5 passes.
FULL_EVAL starts only if GATE10 passes.
EOF
