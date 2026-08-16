#!/usr/bin/env bash
# One-command gated partial-overlap experiment.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
RUN_NAME="${RUN_NAME:-cnn_real_partial_overlap_v1}"

cd "${PROJECT_DIR}"
mkdir -p out Results/Evaluation/RealAdaptationOvernight

FULL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest_full_pairs.jsonl"
python scripts/data/build_full_line_pair_manifest.py \
  --root "${PROJECT_DIR}/DataSet/ArabicDataset" \
  --output "${FULL_MANIFEST}"

# Parent policy:
# - explicit PARENT_CHECKPOINT always wins;
# - otherwise use R1 only when its scientific gate explicitly passed;
# - otherwise fall back to the known-good Stage-1 synthetic checkpoint.
# This avoids selecting a half-written checkpoint merely because model_latest.pth
# exists while a training job is still running or after a failed gate.
PARENT_NAME="${PARENT_NAME:-}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-}"
if [[ -z "${PARENT_CHECKPOINT}" ]]; then
  R1_NAME=cnn_real_r1_positive_pairs
  R1_CHECKPOINT="${PROJECT_DIR}/Weights/${R1_NAME}/model_latest.pth"
  R1_GATE="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R1_NAME}_discrimination/gate_result.json"
  R1_PASSED=0
  if [[ -f "${R1_CHECKPOINT}" && -f "${R1_GATE}" ]]; then
    R1_PASSED="$(python - "${R1_GATE}" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print(1 if bool(data.get("pass")) else 0)
except Exception:
    print(0)
PY
)"
  fi
  if [[ "${R1_PASSED}" == 1 ]]; then
    PARENT_NAME="${R1_NAME}"
    PARENT_CHECKPOINT="${R1_CHECKPOINT}"
  else
    PARENT_NAME=cnn_bilstm_augmented_fixed63_27k
    PARENT_CHECKPOINT="${PROJECT_DIR}/Weights/${PARENT_NAME}/model_latest.pth"
  fi
fi
[[ -n "${PARENT_CHECKPOINT}" && -f "${PARENT_CHECKPOINT}" ]] || {
  echo "ERROR: no usable passed-R1 or Stage1 parent checkpoint found." >&2
  exit 2
}
[[ -n "${PARENT_NAME}" ]] || PARENT_NAME="$(basename "$(dirname "${PARENT_CHECKPOINT}")")"

CANDIDATE_CHECKPOINT="${PROJECT_DIR}/Weights/${RUN_NAME}/model_latest.pth"
PREPROCESS_ENV="${PROJECT_DIR}/Results/Evaluation/RealAdaptationOvernight/preprocessing.env"
if [[ ! -f "${PREPROCESS_ENV}" ]]; then
  cat > "${PREPROCESS_ENV}" <<'EOF'
export REAL_BINARIZE=1
export REAL_BINARIZE_METHOD=otsu
export REAL_BINARIZE_AUTOCONTRAST=1
export REAL_BINARIZE_AUTO_INVERT=1
EOF
fi

PARENT_EVAL_NAME="${PARENT_NAME}_partial_overlap_reference"
PARENT_ROOT="Results/Evaluation/Representation_Diagnostics/${PARENT_EVAL_NAME}_discrimination"
CANDIDATE_ROOT="Results/Evaluation/Representation_Diagnostics/${RUN_NAME}_discrimination"

BASELINE_JOB_ID="$(sbatch --parsable \
  --job-name=eval_partial_parent \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${PARENT_CHECKPOINT}' RUN_NAME='${PARENT_EVAL_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh")"

TRAIN_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${BASELINE_JOB_ID}" \
  --job-name="${RUN_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=20:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${RUN_NAME}' PRETRAINED_WEIGHTS='${PARENT_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=1e-6 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl'; bash scripts/train/run_real_partial_overlap_adaptation.sh")"

EVAL_GATE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --job-name=eval_gate_partial_overlap \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${CANDIDATE_CHECKPOINT}'; CHECKPOINT='${CANDIDATE_CHECKPOINT}' RUN_NAME='${RUN_NAME}' N_SAMPLES=20 bash scripts/eval/run_real_discrimination_sweep.sh; python scripts/eval/check_real_discrimination_gate.py '${CANDIDATE_ROOT}' --baseline-root '${PARENT_ROOT}'")"

FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EVAL_GATE_JOB_ID}" \
  --job-name=final_partial_overlap_eval \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${CANDIDATE_CHECKPOINT}' RUN_NAME='${RUN_NAME}_full' N_SAMPLES=100 bash scripts/eval/run_real_discrimination_sweep.sh; echo '=== PARENT ==='; python scripts/eval/summarize_real_discrimination.py '${PARENT_ROOT}'; echo '=== PARTIAL OVERLAP N=20 ==='; python scripts/eval/summarize_real_discrimination.py '${CANDIDATE_ROOT}'; echo '=== PARTIAL OVERLAP N=100 ==='; python scripts/eval/summarize_real_discrimination.py 'Results/Evaluation/Representation_Diagnostics/${RUN_NAME}_full_discrimination'")"

cat <<EOF
Submitted partial-overlap experiment:
  parent checkpoint        ${PARENT_NAME}
  PARENT EVAL              ${BASELINE_JOB_ID}
  PARTIAL TRAIN            ${TRAIN_JOB_ID}  ${RUN_NAME}
  EVAL + GATE              ${EVAL_GATE_JOB_ID}
  FINAL N=100              ${FINAL_JOB_ID}

All downstream jobs use afterok. If the candidate does not beat its parent on
fixed real diagnostics, the final evaluation is not released.
EOF
