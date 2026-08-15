#!/usr/bin/env bash
# Robust resume after the completed 5-epoch joint-real pilot.
# Uses GPU jobs only and builds deterministic held-out diagnostics automatically.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
PILOT_NAME="${PILOT_NAME:-cnn_joint_real_from_stage1_v1}"
CONT_NAME="${CONT_NAME:-cnn_joint_real_from_stage1_v1_cont10}"
BASELINE_NAME="${BASELINE_NAME:-cnn_bilstm_phase3_joint_fixed}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"

cd "${PROJECT_DIR}"
mkdir -p out

PILOT_CHECKPOINT="${PROJECT_DIR}/Weights/${PILOT_NAME}/model_latest.pth"
CONT_CHECKPOINT="${PROJECT_DIR}/Weights/${CONT_NAME}/model_latest.pth"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
if [[ -z "${BASELINE_CHECKPOINT}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/Weights/cnn_bilstm_phase3/model_latest.pth" \
    "${PROJECT_DIR}/Weights/cnn_bilstm_phase3/model_best.pth"; do
    if [[ -f "${candidate}" ]]; then
      BASELINE_CHECKPOINT="${candidate}"
      break
    fi
  done
fi

[[ -f "${PILOT_CHECKPOINT}" ]] || {
  echo "ERROR: missing completed pilot checkpoint: ${PILOT_CHECKPOINT}" >&2
  exit 2
}
[[ -n "${BASELINE_CHECKPOINT}" && -f "${BASELINE_CHECKPOINT}" ]] || {
  echo "ERROR: missing Phase-3 baseline checkpoint under Weights/cnn_bilstm_phase3" >&2
  exit 2
}

echo "Using pilot checkpoint:    ${PILOT_CHECKPOINT}"
echo "Using Phase-3 baseline:    ${BASELINE_CHECKPOINT}"
echo "Diagnostic manifests will be built automatically from DataSet/ArabicDataset if absent."

# run_real_discrimination_sweep.sh appends _discrimination to every RUN_NAME.
# Keep the gate/summarizer paths exactly aligned with the directories it creates.
BASELINE_ROOT="Results/Evaluation/Representation_Diagnostics/${BASELINE_NAME}_discrimination"
PILOT_ROOT="Results/Evaluation/Representation_Diagnostics/${PILOT_NAME}_discrimination"
CONT_ROOT="Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_discrimination"
FINAL_ROOT="Results/Evaluation/Representation_Diagnostics/${CONT_NAME}_full_discrimination"

# 1) Build held-out fixed manifests (inside the sweep), evaluate Phase 3 and the
#    completed 5-epoch pilot on exactly the same 20+20 rows, then gate relative
#    to Phase 3. A failed scientific gate prevents TRAIN10 via afterok.
EVAL5_GATE_JOB_ID="$(sbatch --parsable \
  --job-name=eval_gate_joint_real_5ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; echo '=== PHASE3 SAME-MANIFEST BASELINE ==='; CHECKPOINT='${BASELINE_CHECKPOINT}' RUN_NAME='${BASELINE_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== 5-EPOCH PILOT ==='; CHECKPOINT='${PILOT_CHECKPOINT}' RUN_NAME='${PILOT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py '${PILOT_ROOT}' --baseline-root '${BASELINE_ROOT}'")"

# 2) Ten ADDITIONAL epochs, only if the 5-epoch same-manifest gate passes.
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

# 3) Check the continuation log, evaluate N=20 on the exact same manifests,
#    and gate against the same Phase-3 evaluation.
CHECK_EVAL10_GATE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${TRAIN10_JOB_ID}" \
  --job-name=check_eval_gate_joint_real_10ep \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; test -f '${CONT_CHECKPOINT}'; LOG=\$(ls -t out/${CONT_NAME}_*.out 2>/dev/null | sed -n '1p'); test -n \"\$LOG\"; echo \"CONT_LOG=\$LOG\"; grep -m 20 -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' \"\$LOG\"; test \$(grep -c 'sequence_batch' \"\$LOG\" || true) -gt 0; echo '=== FIRST SEQUENCE BATCHES ==='; grep -m 10 'sequence_batch' \"\$LOG\"; echo '=== LAST SEQUENCE BATCHES ==='; grep 'sequence_batch' \"\$LOG\" | tail -10; CHECKPOINT='${CONT_CHECKPOINT}' RUN_NAME='${CONT_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py '${CONT_ROOT}' --baseline-root '${BASELINE_ROOT}'")"

# 4) Larger final fixed-manifest evaluation and final summaries.
FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${CHECK_EVAL10_GATE_JOB_ID}" \
  --job-name=final_eval_joint_real \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" \
  --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; CHECKPOINT='${CONT_CHECKPOINT}' RUN_NAME='${CONT_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== PHASE3 SAME-MANIFEST BASELINE ==='; python scripts/eval/summarize_real_discrimination.py '${BASELINE_ROOT}'; echo; echo '=== PILOT 5-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py '${PILOT_ROOT}'; echo; echo '=== CONTINUATION 10-EPOCH ==='; python scripts/eval/summarize_real_discrimination.py '${CONT_ROOT}'; echo; echo '=== FINAL LARGE EVAL ==='; python scripts/eval/summarize_real_discrimination.py '${FINAL_ROOT}'; python scripts/eval/check_real_discrimination_gate.py '${FINAL_ROOT}' --baseline-root '${BASELINE_ROOT}' --no-fail")"

cat <<EOF
Submitted self-contained GPU-only joint-real pipeline:
  EVAL PHASE3 + EVAL5 + GATE5  ${EVAL5_GATE_JOB_ID}
  TRAIN10                       ${TRAIN10_JOB_ID}  ${CONT_NAME}
  CHECK + EVAL10 + GATE10      ${CHECK_EVAL10_GATE_JOB_ID}
  FINAL N=100 + SUMMARY         ${FINAL_JOB_ID}

All models use deterministic held-out manifests generated from the canonical 80/10/10 split.
TRAIN10 starts only if the 5-epoch model beats Phase 3 on those same rows with healthy paths.
FINAL evaluation starts only if the continuation check/evaluation/gate succeeds.
EOF
