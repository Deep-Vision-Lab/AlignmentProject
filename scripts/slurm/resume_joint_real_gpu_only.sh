#!/usr/bin/env bash
# Robust resume after the completed 5-epoch joint-real pilot.
# Uses GPU jobs only so BGU's sbatch wrapper never has to infer CPU-only jobs.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PILOT_NAME="${PILOT_NAME:-cnn_joint_real_from_stage1_v1}"
CONT_NAME="${CONT_NAME:-cnn_joint_real_from_stage1_v1_cont10}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"

cd "${PROJECT_DIR}"
mkdir -p out

PILOT_CHECKPOINT="${PROJECT_DIR}/Weights/${PILOT_NAME}/model_latest.pth"
CONT_CHECKPOINT="${PROJECT_DIR}/Weights/${CONT_NAME}/model_latest.pth"
[[ -f "${PILOT_CHECKPOINT}" ]] || {
  echo "ERROR: missing completed pilot checkpoint: ${PILOT_CHECKPOINT}" >&2
  exit 2
}

echo "Using pilot checkpoint: ${PILOT_CHECKPOINT}"

# 1) Evaluate the completed 5-epoch pilot and apply the structural gate in the
#    SAME GPU job. Exit code 3 from the gate prevents TRAIN10 via afterok.
EVAL5_GATE_JOB_ID="$(sbatch --parsable \
  --job-name=eval_gate_joint_real_5ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; CHECKPOINT='${PILOT_CHECKPOINT}' RUN_NAME='${PILOT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${PILOT_NAME}'")"

# 2) Ten ADDITIONAL epochs, only if the 5-epoch evaluation gate passes.
TRAIN10_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EVAL5_GATE_JOB_ID}" \
  --job-name="${CONT_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=1-00:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --export=ALL,JOB_ID="${CONT_NAME}",PRETRAINED_WEIGHTS="${PILOT_CHECKPOINT}",EPOCHS=10,LEARNING_RATE=5e-6,NUM_GPUS=2,EFFECTIVE_GLOBAL_BATCH_SIZE=64,REAL_TRAIN_FRACTION=0.80,REAL_VALID_FRACTION=0.10,REAL_CLEAN_VIEWS_PER_CYCLE=1,REAL_AUG_VIEWS_PER_CYCLE=2,REAL_EFFECTIVE_EPOCH_MULTIPLIER=6,NUM_NEGATIVES=10,SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
  --wrap="bash '${PROJECT_DIR}/scripts/train/run_stage1_joint_real_discrimination.sh'")"

# 3) Check the continuation log, evaluate N=20, and gate in one GPU job.
#    Avoid all early-closing pipelines under pipefail.
CHECK_EVAL10_GATE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${TRAIN10_JOB_ID}" \
  --job-name=check_eval_gate_joint_real_10ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; test -f '${CONT_CHECKPOINT}'; LOG=\$(ls -t out/${CONT_NAME}_*.out 2>/dev/null | sed -n '1p'); test -n \"\$LOG\"; echo \"CONT_LOG=\$LOG\"; grep -m 20 -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' \"\$LOG\"; test \$(grep -c 'sequence_batch' \"\$LOG\" || true) -gt 0; echo '=== FIRST SEQUENCE BATCHES ==='; grep -m 10 'sequence_batch' \"\$LOG\"; echo '=== LAST SEQUENCE BATCHES ==='; grep 'sequence_batch' \"\$LOG\" | tail -10; CHECKPOINT='${CONT_CHECKPOINT}' RUN_NAME='${CONT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}'")"

# 4) Larger final fixed-manifest evaluation and final summaries.
FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${CHECK_EVAL10_GATE_JOB_ID}" \
  --job-name=final_eval_joint_real \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=07:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; CHECKPOINT='${CONT_CHECKPOINT}' RUN_NAME='${CONT_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== PILOT 5-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${PILOT_NAME}'; echo; echo '=== CONTINUATION 10-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}'; echo; echo '=== FINAL LARGE EVAL ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_full'; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_full' --no-fail")"

cat <<EOF
Submitted robust GPU-only joint-real pipeline:
  EVAL5+GATE5          ${EVAL5_GATE_JOB_ID}
  TRAIN10              ${TRAIN10_JOB_ID}  ${CONT_NAME}
  CHECK+EVAL10+GATE10  ${CHECK_EVAL10_GATE_JOB_ID}
  FINAL N=100+SUMMARY  ${FINAL_JOB_ID}

TRAIN10 starts only if EVAL5+GATE5 exits successfully.
FINAL evaluation starts only if the continuation check/evaluation/gate succeeds.
EOF
