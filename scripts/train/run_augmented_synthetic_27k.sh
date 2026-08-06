#!/usr/bin/env bash
# Unified launcher for the pre-augmented 27,000-pair Arabic synthetic dataset.
set -euo pipefail
set -a
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
if [[ -n "${PROJECT_DIR:-}" ]]; then PROJECT_DIR="$(readlink -f "${PROJECT_DIR}")"; else SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"; PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"; fi
cd "${PROJECT_DIR}"; mkdir -p out logs
BRANCH_NAME="$(git branch --show-current 2>/dev/null || true)"; [[ -n "${BRANCH_NAME}" ]] || BRANCH_NAME="$(basename "${PROJECT_DIR}")"; SAFE_BRANCH_NAME="${BRANCH_NAME//\//-}"
JOB_ID="${JOB_ID:-${SAFE_BRANCH_NAME}-augmented-arabic-27k}"; NUM_SAMPLES="${NUM_SAMPLES:-27000}"; EPOCHS="${EPOCHS:-35}"; CONDA_ENV="${CONDA_ENV:-manucripts_align}"
if [[ -z "${DATA_DIR:-}" ]]; then for candidate in "${PROJECT_DIR}/DataSet/AugmentedArabicDataset" "${HOME}/BGU-Lab/AlignmentProject/DataSet/AugmentedArabicDataset"; do if [[ -d "${candidate}/images" && -d "${candidate}/texts" ]]; then DATA_DIR="${candidate}"; break; fi; done; fi
: "${DATA_DIR:?Could not locate DataSet/AugmentedArabicDataset. Set DATA_DIR explicitly.}"; DATA_DIR="$(readlink -f "${DATA_DIR}")"
[[ -d "${DATA_DIR}/images" && -d "${DATA_DIR}/texts" ]] || { echo "ERROR: dataset must contain images/ and texts/: ${DATA_DIR}" >&2; exit 2; }
PAIR_COUNT="$(find "${DATA_DIR}/images" -maxdepth 1 -type f -name 'img1_*.png' | wc -l | tr -d ' ')"; (( PAIR_COUNT >= NUM_SAMPLES )) || { echo "ERROR: requested ${NUM_SAMPLES} pairs, found ${PAIR_COUNT}" >&2; exit 2; }
DATASET_TYPE=synthetic; SYNTHETIC_MANUSCRIPT_AUGMENT=0; REAL_AUGMENT=0
LINE_WIDTH="${LINE_WIDTH:-1024}"; WINDOW_SIZE="${WINDOW_SIZE:-32}"; STRIDE_RATIO="${STRIDE_RATIO:-0.5}"; WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE:-custom}"
read -r STRIDE_PIXELS IMAGE_WINDOWS < <(python - "${LINE_WIDTH}" "${WINDOW_SIZE}" "${STRIDE_RATIO}" "${WINDOW_OVERLAP_MODE}" <<'PY'
import sys
w=int(sys.argv[1]); k=int(sys.argv[2]); r=float(sys.argv[3]); m=sys.argv[4].lower()
s=k if m=='no_overlap' else max(1,k//2) if m=='light_overlap' else max(1,k//4) if m=='dense_overlap' else max(1,int(k*r)) if m=='custom' else None
if s is None: raise SystemExit(f'Unknown WINDOW_OVERLAP_MODE={m!r}')
print(s,((w-k)//s)+1)
PY
)
MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS:-3}"; SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP:-${MAX_TEXT_SPAN_CHARS}}"
[[ "${MAX_TEXT_SPAN_CHARS}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: MAX_TEXT_SPAN_CHARS must be positive." >&2; exit 2; }
[[ "${SPAN_MAX_CORE_CHARS_CAP}" =~ ^[0-9]+$ ]] || { echo "ERROR: SPAN_MAX_CORE_CHARS_CAP must be non-negative." >&2; exit 2; }
if (( SPAN_MAX_CORE_CHARS_CAP > 0 && SPAN_MAX_CORE_CHARS_CAP < MAX_TEXT_SPAN_CHARS )); then EFFECTIVE_SPAN_CHARS="${SPAN_MAX_CORE_CHARS_CAP}"; else EFFECTIVE_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS}"; fi
SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN:-${EFFECTIVE_SPAN_CHARS}}"
[[ "${SPAN_CONNECTED_MAX_UNITS_PER_SPAN}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SPAN_CONNECTED_MAX_UNITS_PER_SPAN must be positive." >&2; exit 2; }
if (( SPAN_CONNECTED_MAX_UNITS_PER_SPAN < EFFECTIVE_SPAN_CHARS )); then
  echo "Overriding stale SPAN_CONNECTED_MAX_UNITS_PER_SPAN=${SPAN_CONNECTED_MAX_UNITS_PER_SPAN} with ${EFFECTIVE_SPAN_CHARS} for the variable-length dataset." >&2
  SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${EFFECTIVE_SPAN_CHARS}"
fi
IFS=$'\t' read -r WORST_REQUIRED_SPANS WORST_TEXT_LENGTH WORST_TEXT_FILE < <(python - "${DATA_DIR}/texts" "${NUM_SAMPLES}" "${EFFECTIVE_SPAN_CHARS}" <<'PY'
import math,pathlib,re,sys
root=pathlib.Path(sys.argv[1]); limit=int(sys.argv[2]); cap=int(sys.argv[3]); pat=re.compile(r'text[12]_(\d+)\.txt$'); worst=(-1,-1,''); seen=0
for p in root.glob('text[12]_*.txt'):
 m=pat.fullmatch(p.name)
 if not m or int(m.group(1))>limit: continue
 t=p.read_text(encoding='utf-8').strip(); req=0; i=0
 while i<len(t):
  if t[i].isspace(): req+=1; i+=1; continue
  j=i+1
  while j<len(t) and not t[j].isspace(): j+=1
  req+=math.ceil((j-i)/cap); i=j
 seen+=1; worst=max(worst,(req,len(t),p.name))
if not seen: raise SystemExit(f'No transcript files found under {root}')
print(f'{worst[0]}\t{worst[1]}\t{worst[2]}')
PY
)
if (( WORST_REQUIRED_SPANS > IMAGE_WINDOWS )); then echo "ERROR: ${WORST_TEXT_FILE} needs ${WORST_REQUIRED_SPANS} spans but only ${IMAGE_WINDOWS} windows are available." >&2; echo "Increase MAX_TEXT_SPAN_CHARS while keeping STRIDE_RATIO=${STRIDE_RATIO}." >&2; exit 2; fi
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"; NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"; NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"; NCCL_DEBUG="${NCCL_DEBUG:-WARN}"; CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ -f "${PROJECT_DIR}/scripts/train/run_connected_subword_synthetic.sh" ]]; then NUM_GPUS="${NUM_GPUS:-2}"; printf '%s\n' "Augmented Arabic synthetic training" "  branch=${BRANCH_NAME}" "  dataset=${DATA_DIR}" "  pairs=${PAIR_COUNT}" "  samples=${NUM_SAMPLES}" "  epochs=${EPOCHS}" "  online augmentation=disabled" "  geometry=window ${WINDOW_SIZE}, stride ${STRIDE_PIXELS}, ${IMAGE_WINDOWS} windows" "  text spans=1-${EFFECTIVE_SPAN_CHARS} chars; connected units/span=${SPAN_CONNECTED_MAX_UNITS_PER_SPAN}; worst transcript needs ${WORST_REQUIRED_SPANS}/${IMAGE_WINDOWS} positions" "  NCCL P2P=disabled" "  GPUs=${NUM_GPUS}" "  Slurm tasks=1" "  job id=${JOB_ID}"; exec bash "${PROJECT_DIR}/scripts/train/run_connected_subword_synthetic.sh"; fi
if [[ -f "${PROJECT_DIR}/train.py" ]] && grep -q -- '--dataset_type' "${PROJECT_DIR}/train.py"; then TRAINING_MODE=generic; NUM_GPUS="${NUM_GPUS:-2}"; elif [[ -f "${PROJECT_DIR}/run_train.sh" ]]; then TRAINING_MODE=legacy; NUM_GPUS="${NUM_GPUS:-1}"; else echo "ERROR: no supported training entry point on ${BRANCH_NAME}" >&2; exit 2; fi
GPU_RESOURCE="${GPU_RESOURCE:-rtx_4090}"; PARTITION="${PARTITION:-rtx4090}"; CPUS_PER_TASK="${CPUS_PER_TASK:-$((8*NUM_GPUS))}"; TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"; MAIL_USER="${MAIL_USER:-ahmedmas@post.bgu.ac.il}"; SLURM_JOB_NAME="${SLURM_JOB_NAME:-${JOB_ID}}"
printf '%s\n' "Augmented Arabic synthetic training" "  branch=${BRANCH_NAME}" "  dataset=${DATA_DIR}" "  pairs=${PAIR_COUNT}" "  samples=${NUM_SAMPLES}" "  epochs=${EPOCHS}" "  online augmentation=disabled" "  geometry=window ${WINDOW_SIZE}, stride ${STRIDE_PIXELS}, ${IMAGE_WINDOWS} windows" "  text spans=1-${EFFECTIVE_SPAN_CHARS} chars; connected units/span=${SPAN_CONNECTED_MAX_UNITS_PER_SPAN}; worst transcript needs ${WORST_REQUIRED_SPANS}/${IMAGE_WINDOWS} positions" "  NCCL P2P=disabled" "  Slurm request=${NUM_GPUS} ${GPU_RESOURCE} GPU(s), 1 task, ${CPUS_PER_TASK} CPUs, ${MEMORY}" "  job id=${JOB_ID}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then sbatch --job-name="${SLURM_JOB_NAME}" --output="${PROJECT_DIR}/out/%x_%J.out" --error="${PROJECT_DIR}/out/%x_%J.err" --chdir="${PROJECT_DIR}" --partition="${PARTITION}" --gpus="${GPU_RESOURCE}:${NUM_GPUS}" --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --time="${TIME_LIMIT}" --mail-type=ALL --mail-user="${MAIL_USER}" --export=ALL,PROJECT_DIR="${PROJECT_DIR}",DATA_DIR="${DATA_DIR}",JOB_ID="${JOB_ID}",NUM_SAMPLES="${NUM_SAMPLES}",EPOCHS="${EPOCHS}",NUM_GPUS="${NUM_GPUS}",TRAINING_MODE="${TRAINING_MODE}",SYNTHETIC_MANUSCRIPT_AUGMENT=0,REAL_AUGMENT=0,DATASET_TYPE=synthetic,WINDOW_SIZE="${WINDOW_SIZE}",STRIDE_RATIO="${STRIDE_RATIO}",WINDOW_OVERLAP_MODE="${WINDOW_OVERLAP_MODE}",MAX_TEXT_SPAN_CHARS="${MAX_TEXT_SPAN_CHARS}",SPAN_MAX_CORE_CHARS_CAP="${SPAN_MAX_CORE_CHARS_CAP}",SPAN_CONNECTED_MAX_UNITS_PER_SPAN="${SPAN_CONNECTED_MAX_UNITS_PER_SPAN}",NCCL_P2P_DISABLE=1,NCCL_SHM_DISABLE=0,NCCL_ASYNC_ERROR_HANDLING=1 "${SCRIPT_PATH}"; exit 0; fi
if command -v module >/dev/null 2>&1; then module load anaconda || true; fi
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" TOKENIZERS_PARALLELISM=false XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG CUDA_DEVICE_ORDER WINDOW_SIZE STRIDE_RATIO WINDOW_OVERLAP_MODE MAX_TEXT_SPAN_CHARS SPAN_MAX_CORE_CHARS_CAP SPAN_CONNECTED_MAX_UNITS_PER_SPAN
if [[ "${TRAINING_MODE}" == generic ]]; then TRAIN_ARGS=(train.py --job_id "${JOB_ID}" --dataset_type synthetic --data_dir "${DATA_DIR}" --num_samples "${NUM_SAMPLES}" --epochs "${EPOCHS}" --window_size "${WINDOW_SIZE}" --stride_ratio "${STRIDE_RATIO}" --window_overlap_mode "${WINDOW_OVERLAP_MODE}" --no-augment); if (( NUM_GPUS>1 )); then exec torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --max_restarts=0 "${TRAIN_ARGS[@]}"; fi; exec python "${TRAIN_ARGS[@]}"; fi
exec bash run_train.sh "${JOB_ID}" "${DATA_DIR}"
