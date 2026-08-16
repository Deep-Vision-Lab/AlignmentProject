#!/usr/bin/env bash
# Train the partial-overlap fix from the existing R2 checkpoint, then gate it
# against both R2 and Phase-3 on the same fixed real diagnostic manifests.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ahmedmas/BGU-Lab/AlignmentProject}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-agent/training-speed-optimization}"
GPU_PARTITION="${GPU_PARTITION:-rtx4090}"
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"
CONDA_ENV="${CONDA_ENV:-manucripts_align}"
MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"
RUN_NAME="${RUN_NAME:-cnn_real_partial_overlap_from_r2_v1}"
R2_NAME="${R2_NAME:-cnn_real_r2_full_discrimination}"

cd "${PROJECT_DIR}"
mkdir -p out Results/Evaluation/RealAdaptationOvernight

CURRENT_BRANCH="$(git branch --show-current)"
PINNED_COMMIT="$(git rev-parse HEAD)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: expected branch ${EXPECTED_BRANCH}, current=${CURRENT_BRANCH}" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked working-tree changes are present; commit/stash them before submitting." >&2
  git status --short >&2
  exit 2
fi

R2_CHECKPOINT="${PROJECT_DIR}/Weights/${R2_NAME}/model_latest.pth"
R2_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${R2_NAME}_discrimination"
PHASE3_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/cnn_phase3_adaptation_reference_discrimination"
if [[ ! -d "${PHASE3_ROOT}" ]]; then
  PHASE3_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/cnn_bilstm_phase3_joint_fixed_discrimination"
fi
PREPROCESS_ENV="${PROJECT_DIR}/Results/Evaluation/RealAdaptationOvernight/preprocessing.env"
FULL_MANIFEST="${PROJECT_DIR}/DataSet/ArabicDataset/dataset_manifest_full_pairs.jsonl"
PARTIAL_LAUNCHER="${PROJECT_DIR}/scripts/train/run_real_partial_overlap_adaptation.sh"
EVAL_SWEEP="${PROJECT_DIR}/scripts/eval/run_real_discrimination_sweep.sh"
GLOBAL_GATE="${PROJECT_DIR}/scripts/eval/check_real_discrimination_global_gate.py"
SUMMARY="${PROJECT_DIR}/scripts/eval/summarize_real_discrimination.py"
SMOKE_TEST="${PROJECT_DIR}/scripts/data/smoke_test_partial_overlap.py"
CANDIDATE_CHECKPOINT="${PROJECT_DIR}/Weights/${RUN_NAME}/model_latest.pth"
CANDIDATE_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${RUN_NAME}_discrimination"
FINAL_ROOT="${PROJECT_DIR}/Results/Evaluation/Representation_Diagnostics/${RUN_NAME}_full_discrimination"

for required in \
  "${R2_CHECKPOINT}" "${PREPROCESS_ENV}" "${PARTIAL_LAUNCHER}" \
  "${EVAL_SWEEP}" "${GLOBAL_GATE}" "${SUMMARY}" "${SMOKE_TEST}"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing required file ${required}" >&2; exit 2; }
done
for required_dir in "${R2_ROOT}" "${PHASE3_ROOT}"; do
  [[ -d "${required_dir}" ]] || { echo "ERROR: missing evaluation directory ${required_dir}" >&2; exit 2; }
done

# Make the full relationship manifest deterministic and current before smoke-test.
python scripts/data/build_full_line_pair_manifest.py \
  --root "${PROJECT_DIR}/DataSet/ArabicDataset" \
  --output "${FULL_MANIFEST}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== PARTIAL-OVERLAP CPU SMOKE TEST ==="
python "${SMOKE_TEST}" \
  --root "${PROJECT_DIR}/DataSet/ArabicDataset" \
  --samples 16

# Refuse duplicate live chains. Finished checkpoints/evaluations are allowed;
# RUN_NAME can be overridden for a second independent experiment.
for name in "${RUN_NAME}" eval_gate_partial_from_r2 final_partial_from_r2; do
  if squeue -h -u "${USER}" -n "${name}" | grep -q .; then
    echo "ERROR: active job already exists with name ${name}; refusing duplicate submission." >&2
    exit 2
  fi
done

source "${PREPROCESS_ENV}"
checkout_guard="git -C '${PROJECT_DIR}' rev-parse HEAD | grep -qx '${PINNED_COMMIT}' || { echo 'ERROR: project checkout changed after submission; expected ${PINNED_COMMIT}' >&2; exit 2; }; test -f '${EVAL_SWEEP}' || { echo 'ERROR: missing evaluator ${EVAL_SWEEP}' >&2; exit 2; }; test -f '${GLOBAL_GATE}' || { echo 'ERROR: missing global gate ${GLOBAL_GATE}' >&2; exit 2; }"

echo "=== SUBMIT PARTIAL-OVERLAP FIX ==="
echo "parent=${R2_CHECKPOINT}"
echo "R2_reference=${R2_ROOT}"
echo "Phase3_reference=${PHASE3_ROOT}"
echo "branch=${CURRENT_BRANCH}"
echo "commit=${PINNED_COMMIT}"

TRAIN_JOB_ID="$(sbatch --parsable \
  --job-name="${RUN_NAME}" \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:2" \
  --tasks=1 --cpus-per-task=16 --mem=96G --time=20:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; git -C '${PROJECT_DIR}' rev-parse HEAD | grep -qx '${PINNED_COMMIT}' || { echo 'ERROR: project checkout changed after submission; expected ${PINNED_COMMIT}' >&2; exit 2; }; test -f '${PARTIAL_LAUNCHER}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; export JOB_ID='${RUN_NAME}' PRETRAINED_WEIGHTS='${R2_CHECKPOINT}' EPOCHS=3 LEARNING_RATE=7.5e-7 NUM_GPUS=2 EFFECTIVE_GLOBAL_BATCH_SIZE=64 REAL_MANIFEST_NAME='dataset_manifest_full_pairs.jsonl' REAL_TRAIN_SAMPLES_PER_EPOCH=6000 REAL_PARTIAL_OVERLAP_POSITIVE_FRACTION=0.60 REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB=0.75 REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB=0.20 REAL_PARTIAL_OVERLAP_EDGE_DISTRACTOR_PROB=0.55 IMAGE_PAIR_LOSS_WEIGHT=0.10 SEQUENCE_RANKING_WEIGHT=0.03 SEQUENCE_RANKING_THRESHOLD=0.50 SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR=0.08 TRAIN_EXPECTED_BRANCH='${CURRENT_BRANCH}' TRAIN_EXPECTED_COMMIT='${PINNED_COMMIT}'; bash '${PARTIAL_LAUNCHER}'")"

EVAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB_ID}" \
  --job-name=eval_gate_partial_from_r2 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; ${checkout_guard}; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; test -f '${CANDIDATE_CHECKPOINT}'; CHECKPOINT='${CANDIDATE_CHECKPOINT}' RUN_NAME='${RUN_NAME}' N_SAMPLES=20 bash '${EVAL_SWEEP}'; echo '=== GLOBAL GATE VS R2 ==='; python '${GLOBAL_GATE}' '${CANDIDATE_ROOT}' --baseline-root '${R2_ROOT}' --min-positive-steps 8 --min-step-gap 2; echo '=== GLOBAL GATE VS PHASE3 ==='; python '${GLOBAL_GATE}' '${CANDIDATE_ROOT}' --baseline-root '${PHASE3_ROOT}' --min-positive-steps 8 --min-step-gap 2")"

FINAL_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EVAL_JOB_ID}" \
  --job-name=final_partial_from_r2 \
  --output="${PROJECT_DIR}/out/%x_%J.out" \
  --chdir="${PROJECT_DIR}" \
  --partition="${GPU_PARTITION}" --gpus="${GPU_RESOURCE}:1" \
  --tasks=1 --cpus-per-task=8 --mem=48G --time=08:00:00 \
  --mail-type=ALL --mail-user="${MAIL_USER}" \
  --wrap="set -euo pipefail; ${checkout_guard}; source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate '${CONDA_ENV}'; cd '${PROJECT_DIR}'; source '${PREPROCESS_ENV}'; CHECKPOINT='${CANDIDATE_CHECKPOINT}' RUN_NAME='${RUN_NAME}_full' N_SAMPLES=100 bash '${EVAL_SWEEP}'; echo '=== R2 N=20 ==='; python '${SUMMARY}' '${R2_ROOT}'; echo '=== PHASE3 N=20 ==='; python '${SUMMARY}' '${PHASE3_ROOT}'; echo '=== PARTIAL N=20 ==='; python '${SUMMARY}' '${CANDIDATE_ROOT}'; echo '=== PARTIAL N=100 ==='; python '${SUMMARY}' '${FINAL_ROOT}'")"

cat <<EOF
Submitted partial-overlap fix from the existing R2 checkpoint:
  PARTIAL TRAIN      ${TRAIN_JOB_ID}  ${RUN_NAME}
  EVAL + GLOBAL GATES ${EVAL_JOB_ID}
  FINAL N=100        ${FINAL_JOB_ID}

Training default exposure per 6000-sample epoch:
  ~1200 canonical positive exposures (20%)
  ~1800 partial-overlap positive exposures (30%)
  ~3000 no-shared negative exposures (50%)

The evaluation job must find a threshold with positive_steps>=8 and step_gap>=2,
and that healthy operating point must beat the GLOBAL best structural AUROC of
both R2 and Phase-3. Final N=100 remains blocked otherwise.

Pinned checkout:
  branch=${CURRENT_BRANCH}
  commit=${PINNED_COMMIT}

Do not switch branches or change tracked files in ${PROJECT_DIR} until this chain finishes.
EOF
