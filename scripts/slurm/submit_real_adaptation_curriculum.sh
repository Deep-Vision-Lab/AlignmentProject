#!/usr/bin/env bash
# Overnight conservative real-domain adaptation curriculum.
# One command submits: preprocessing selection -> R0 -> gate -> R1 -> gate -> R2 -> gate -> final N=100.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"

STAGE1_NAME="${STAGE1_NAME:-cnn_bilstm_augmented_fixed63_27k}"
PHASE3_NAME="${PHASE3_NAME:-cnn_bilstm_phase3}"
R0_NAME="${R0_NAME:-cnn_real_r0_image_text}"
R1_NAME="${R1_NAME:-cnn_real_r1_positive_pairs}"
R2_NAME="${R2_NAME:-cnn_real_r2_full_discrimination}"

cd "${PROJECT_DIR}"
mkdir -p out Results/Evaluation/RealAdaptationOvernight

STAGE1_CHECKPOINT="${PROJECT_DIR}/Weights/${STAGE1_NAME}/model_latest.pth"
R0_CHECKPOINT="${PROJECT_DIR}/Weights/${R0_NAME}/model_latest.pth"
R1_CHECKPOINT="${PROJECT_DIR}/Weights/${R1_NAME}/model_latest.pth"
R2_CHECKPOINT="${PROJECT_DIR}/Weights/${R2_NAME}/model_latest.pth"
STATE_DIR="${PROJECT_DIR}/Results/Evaluation/RealAdaptationOvernight"
PREPROCESS_ENV="${STATE_DIR}/preprocessing.env"
FULL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest_full_pairs.jsonl"

[[ -f "${STAGE1_CHECKPOINT}" ]] || {
  echo "ERROR: missing Stage-1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
  exit 2
}

PHASE3_CHECKPOINT="${PHASE3_CHECKPOINT:-}"
if [[ -z "${PHASE3_CHECKPOINT}" ]]; then
  for candidate in \
    "${PROJECT_DIR}/Weights/${PHASE3_NAME}/model_latest.pth" \
    "${PROJECT_DIR}/Weights/${PHASE3_NAME}/model_best.pth"; do
    if [[ -f "${candidate}" ]]; then
      PHASE3_CHECKPOINT="${candidate}"
      break
    fi
  done
fi
[[ -n "${PHASE3_CHECKPOINT}" && -f "${PHASE3_CHECKPOINT}" ]] || {
  echo "ERROR: missing Phase-3 checkpoint under Weights/${PHASE3_NAME}" >&2
  exit 2
}

python scripts/data/build_full_line_pair_manifest.py \
  --root "${PROJECT_DIR}/DataSet/ArabicDataset" \
  --output "${FULL_MANIFEST}"

rm -f "${PREPROCESS_ENV}"

PREP_JOB_ID="$(sbatch --parsable \
  --job-name=real_adapt_preprocess \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; REAL_BINARIZE=1 CHECKPOINT='${STAGE1_CHECKPOINT}' RUN_NAME='cnn_stage1_real_bin' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; REAL_BINARIZE=0 CHECKPOINT='${STAGE1_CHECKPOINT}' RUN_NAME='cnn_stage1_real_gray' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/choose_real_preprocessing.py --binarized-root 'Results/Evaluation/Representation_Diagnostics/cnn_stage1_real_bin_discrimination' --gray-root 'Results/Evaluation/Representation_Diagnostics/cnn_stage1_real_gray_discrimination' --output-env '${PREPROCESS_ENV}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${PHASE3_CHECKPOINT}' RUN_NAME='cnn_phase3_adaptation_reference' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh")"

R0_TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB_ID}" \
  --job-name="${R0_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=18:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${R0_NAME}' PRETRAINED_WEIGHTS='${STAGE1_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=2e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl'; bash scripts/train/run_real_unique_image_text_adaptation.sh")"

R0_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R0_TRAIN_JOB_ID}" \
  --job-name=eval_gate_real_r0 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R0_CHECKPOINT}'; CHECKPOINT='${R0_CHECKPOINT}' RUN_NAME='${R0_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_preservation.py 'Results/Evaluation/Representation_Diagnostics/${R0_NAME}_discrimination' --baseline-root \"\${STAGE1_BASELINE_ROOT}\"")"

R1_TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R0_EVAL_JOB_ID}" \
  --job-name="${R1_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=18:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${R1_NAME}' PRETRAINED_WEIGHTS='${R0_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=1e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl'; bash scripts/train/run_real_positive_pair_adaptation.sh")"

# R1 is the first stage with partner supervision.  It must improve structural
# discrimination over R0 while establishing a positive path gap, but the final
# Phase-3-strength 8-step/2-step-gap requirement is reserved for R2.
R1_EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R1_TRAIN_JOB_ID}" \
  --job-name=eval_gate_real_r1 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=05:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R1_CHECKPOINT}'; CHECKPOINT='${R1_CHECKPOINT}' RUN_NAME='${R1_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination' --baseline-root 'Results/Evaluation/Representation_Diagnostics/${R0_NAME}_discrimination' --min-positive-steps 2.0 --min-step-gap 0.5")"

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
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${R2_CHECKPOINT}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination' --baseline-root 'Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination'; python scripts/eval/check_real_discrimination_gate.py 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination' --baseline-root 'Results/Evaluation/Representation_Diagnostics/cnn_phase3_adaptation_reference_discrimination'")"

FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${R2_EVAL_JOB_ID}" \
  --job-name=final_real_adaptation_eval \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${R2_CHECKPOINT}' RUN_NAME='${R2_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== SELECTED STAGE1 ==='; python scripts/eval/summarize_real_discrimination.py \"\${STAGE1_BASELINE_ROOT}\"; echo '=== PHASE3 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/cnn_phase3_adaptation_reference_discrimination'; echo '=== R0 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${R0_NAME}_discrimination'; echo '=== R1 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination'; echo '=== R2 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination'; echo '=== FINAL N=100 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${R2_NAME}_full_discrimination'")"

cat <<EOF
Submitted conservative real-adaptation curriculum:
  PREPROCESS + BASELINES   ${PREP_JOB_ID}
  R0 TRAIN                 ${R0_TRAIN_JOB_ID}  ${R0_NAME}
  R0 EVAL + PRESERVE GATE  ${R0_EVAL_JOB_ID}
  R1 TRAIN                 ${R1_TRAIN_JOB_ID}  ${R1_NAME}
  R1 EVAL + GATE           ${R1_EVAL_JOB_ID}
  R2 TRAIN                 ${R2_TRAIN_JOB_ID}  ${R2_NAME}
  R2 EVAL + GATES          ${R2_EVAL_JOB_ID}
  FINAL N=100              ${FINAL_JOB_ID}

Every downstream stage uses afterok. If a training/evaluation/gate fails, later GPU jobs stay blocked.
Preprocessing choice is stored in:
  ${PREPROCESS_ENV}
EOF
