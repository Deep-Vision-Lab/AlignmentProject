#!/usr/bin/env bash
# Isolate a torchrun rank to one Slurm-allocated GPU before Python starts.
set -euo pipefail

rank="${LOCAL_RANK:-0}"
world="${LOCAL_WORLD_SIZE:-${WORLD_SIZE:-1}}"
original_cuda="${CUDA_VISIBLE_DEVICES:-}"

expand_device_list() {
  local raw="$1"
  local -n out_ref=$2
  out_ref=()
  if [[ -z "$raw" ]]; then
    return 0
  fi
  IFS=',' read -ra tokens <<< "$raw"
  local token start end value
  for token in "${tokens[@]}"; do
    token="${token//[[:space:]]/}"
    [[ -n "$token" ]] || continue
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start <= end )); then
        for (( value=start; value<=end; value++ )); do
          out_ref+=("$value")
        done
      else
        for (( value=start; value>=end; value-- )); do
          out_ref+=("$value")
        done
      fi
    else
      out_ref+=("$token")
    fi
  done
}

# Some BGU Slurm jobs expose only CUDA_VISIBLE_DEVICES=0 even when the job was
# allocated non-contiguous GPUs such as SLURM_GPU_INDEX=0,2. Select the first
# allocation source that actually contains enough devices for all local ranks.
candidate_names=(
  "CUDA_VISIBLE_DEVICES"
  "SLURM_STEP_GPUS"
  "SLURM_JOB_GPUS"
  "SLURM_GPU_INDEX"
)
candidate_values=(
  "${CUDA_VISIBLE_DEVICES:-}"
  "${SLURM_STEP_GPUS:-}"
  "${SLURM_JOB_GPUS:-}"
  "${SLURM_GPU_INDEX:-}"
)

source_name=""
source_value=""
devices=()
for index in "${!candidate_names[@]}"; do
  candidate_devices=()
  expand_device_list "${candidate_values[$index]}" candidate_devices
  if (( world > 1 )); then
    (( ${#candidate_devices[@]} >= world )) || continue
  else
    (( ${#candidate_devices[@]} >= 1 )) || continue
  fi
  source_name="${candidate_names[$index]}"
  source_value="${candidate_values[$index]}"
  devices=("${candidate_devices[@]}")
  break
done

if [[ -z "$source_name" ]]; then
  if (( world > 1 )); then
    echo "ERROR rank_wrapper: cannot map ${world} ranks from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>} SLURM_GPU_INDEX=${SLURM_GPU_INDEX:-<unset>}" >&2
    exit 86
  fi
  source_name="LOCAL_RANK fallback"
  source_value="$rank"
  devices=("$rank")
fi

if (( rank < 0 || rank >= world || rank >= ${#devices[@]} )); then
  echo "ERROR rank_wrapper: LOCAL_RANK=${rank} outside available devices for world size ${world}" >&2
  exit 87
fi
selected="${devices[$rank]}"

export ORIGINAL_CUDA_VISIBLE_DEVICES="$original_cuda"
export RANK_SELECTED_CUDA_DEVICE="$selected"
export RANK_GPU_SOURCE_NAME="$source_name"
export RANK_GPU_SOURCE_VALUE="$source_value"
export RANK_WRAPPER_ISOLATED=1
export CUDA_VISIBLE_DEVICES="$selected"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "rank_wrapper rank=${rank} world=${world} source=${source_name} source_value=${source_value:-<unset>} original_cuda=${original_cuda:-<unset>} selected=${selected} process_visible=${CUDA_VISIBLE_DEVICES}" >&2

if [[ "${RANK_WRAPPER_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  echo "ERROR rank_wrapper: missing Python training command" >&2
  exit 88
fi

exec python "$@"
