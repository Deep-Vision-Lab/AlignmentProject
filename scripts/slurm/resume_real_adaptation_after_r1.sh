#!/usr/bin/env bash
# Resume conservative real adaptation from an already-trained R1 checkpoint.
# Intended for recovery when the R1 evaluation job failed mechanically before evaluation.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-agent/training-speed-optimization}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

R0_NAME="${R0_NAME:-cnn_real_r0_image_text}"
R1_NAME="${R1_NAME:-cnn_real_r1_positive_pairs}"
R2_NAME="${R2_NAME:-cnn_real_r2_full_discrimination}"

cd "${PROJECT_DIR}"
mkdir -p out

CURRENT_BRANCH="$(git branch --show-current)"
PINNED_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expected branch ${EXPECTED_BRANCH}, current=${CURRENT_BRANCH}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked working-tree changes are present; commit/stash them before resuming." >&2
  git status --short >&2
  exit 2
fi

R1_CHECKPOINT="${PROJECT_DIR}/Weights/${R1_NAME}/model_latest.pth"
R2_CHECKPOINT="${PROJECT_DIR}/Weights/${R2_NAME}/model_latest.pth"
PREPROCESS_ENV="${PROJECT_DIR}/Results/Evaluation/RealAdaptationOvernight/preprocessing.env"
R0_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R0_NAME}_discrimination"
R1_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination"
R2_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination"
PHASE3_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/cnn_phase3_adaptation_reference_discrimination"
FULL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest_full_pairs.jsonl"
EVAL_SWEEP="${PROJECT_DIR}/scripts/eval/run_real_discrimination_sweep.sh"
GATE="${PROJECT_DIR}/scripts/eval/check_real_discrimination_gate.py"
SUMMARY="${PROJECT_DIR}/scripts/eval/summarize_real_discrimination.py"
R2_LAUNCHER="${PROJECT_DIR}/scripts/train/run_real_full_pair_discrimination.sh"

for required in \
  "${R1_CHECKPOINT}" "${PREPROCESS_ENV}" "${FULL_MANIFEST}" \
  "${EVAL_SWEEP}" "${GATE}" "${SUMMARY}" "${R2_LAUNCHER}"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing required file ${required}" >&2; exit 2; }
done
[[ -d "${R0_ROOT}" ]] || { echo "ERROR: missing R0 evaluation ${R0_ROOT}" >&2; exit 2; }
[[ -d "${PHASE3_ROOT}" ]] || { echo "ERROR: missing Phase-3 reference ${PHASE3_ROOT}" >&2; exit 2; }

source "${PREPROCESS_ENV}"
: "${STAGE1_BASELINE_ROOT:?preprocessing.env is missing STAGE1_BASELINE_ROOT}"

# Refuse to create a duplicate live continuation chain.
for name in eval_gate_real_r1 "${R2_NAME}" eval_gate_real_r2 final_real_adaptation_eval; do
  if squeue -h -u "${USER}" -n "${name}" | grep -q .; then
    echo "ERROR: active job already exists with name ${name}. Cancel stale dependent jobs first." >&2
    exit 2
  fi
done

checkout_guard="git -C '${PROJECT_DIR}' rev-parse HEAD | grep -qx '${PINNED_COMMIT}' || { echo 'ERROR: project checkout changed after submission; expected ${PINNED_COMMIT}' >&2; exit 2; }; test -f '${EVAL_SWEEP}' || { echo 'ERROR: missing evaluator ${EVAL_SWEEP}' >&2; exit 2; }"

echo "Resuming from R1 checkpoint: ${R1_CHECKPOINT}"
echo "Pinned branch=${CURRENT_BRANCH} commit=${PINNED_COMMIT}"

# 1) Evaluate/gate the already-trained R1 checkpoint. No R1 retraining.
R1_EVAL_JOB_ID="$(sbatch --parsable \
  --job-name=eval_gate_real_r1 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; ${checkout_guard}; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R1_CHECKPOINT}'; CHECKPOINT='${R1_CHECKPOINT}' RUN_NAME='${R1_NAME}' N_SAMPLES=20 bash '${EVAL_SWEEP}'; python '${GATE}' '${R1_ROOT}' --baseline-root '${R0_ROOT}' --min-positive-steps 2.0 --min-step-gap 0.5")"

# 2) R2 starts only if R1 evaluation + scientific gate pass.
R2_TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R1_EVAL_JOB_ID}" \
  --job-name="${R2_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=20:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; git -C '${PROJECT_DIR}' rev-parse HEAD | grep -qx '${PINNED_COMMIT}' || { echo 'ERROR: project checkout changed after submission; expected ${PINNED_COMMIT}' >&2; exit 2; }; test -f '${R2_LAUNCHER}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${R2_NAME}' PRETRAINED_WEIGHTS='${R1_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=1e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl' REAL_TRAIN_SAMPLES_PER_EPOCH=6000 REAL_CLEAN_VIEWS_PER_CYCLE=1 REAL_AUG_VIEWS_PER_CYCLE=1 SEQUENCE_RANKING_WEIGHT=0.03 IMAGE_PAIR_LOSS_WEIGHT=0.08 SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0.015 TRAIN_EXPECTED_BRANCH='${CURRENT_BRANCH}' TRAIN_EXPECTED_COMMIT='${PINNED_COMMIT}'; bash '${R2_LAUNCHER}'")"

# 3) R2 must improve over R1 and beat the Phase-3 same-manifest reference.
R2_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R2_TRAIN_JOB_ID}" \
  --job-name=eval_gate_real_r2 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; ${checkout_guard}; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R2_CHECKPOINT}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}' N_SAMPLES=20 bash '${EVAL_SWEEP}'; python '${GATE}' '${R2_ROOT}' --baseline-root '${R1_ROOT}'; python '${GATE}' '${R2_ROOT}' --baseline-root '${PHASE3_ROOT}'")"

# 4) Larger final evaluation only after every prior gate passes.
FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R2_EVAL_JOB_ID}" \
  --job-name=final_real_adaptation_eval \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; ${checkout_guard}; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}_full' N_SAMPLES=100 bash '${EVAL_SWEEP}'; echo '=== SELECTED STAGE1 ==='; python '${SUMMARY}' \"\${STAGE1_BASELINE_ROOT}\"; echo '=== PHASE3 ==='; python '${SUMMARY}' '${PHASE3_ROOT}'; echo '=== R0 ==='; python '${SUMMARY}' '${R0_ROOT}'; echo '=== R1 ==='; python '${SUMMARY}' '${R1_ROOT}'; echo '=== R2 ==='; python '${SUMMARY}' '${R2_ROOT}'; echo '=== FINAL N=100 ==='; python '${SUMMARY}' 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_full_discrimination'")"

cat <<EOF
Submitted safe continuation from existing R1 checkpoint:
  R1 EVAL+GATE    ${R1_EVAL_JOB_ID}
  R2 TRAIN        ${R2_TRAIN_JOB_ID}
  R2 EVAL+GATES   ${R2_EVAL_JOB_ID}
  FINAL N=100     ${FINAL_JOB_ID}

Pinned checkout:
  branch=${CURRENT_BRANCH}
  commit=${PINNED_COMMIT}

Do not switch branches or change tracked files in ${PROJECT_DIR} until this chain finishes.
EOF
