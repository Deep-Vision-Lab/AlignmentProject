#!/usr/bin/env bash
# Isolate a torchrun rank to one Slurm-allocated GPU before Python starts.
set -euo pipefail

rank="${LOCAL_RANK:-0}"
world="${LOCAL_WORLD_SIZE:-${WORLD_SIZE:-1}}"
original_cuda="${CUDA_VISIBLE_DEVICES:-}"
source_value="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${SLURM_GPU_INDEX:-}}}}"
source_name="CUDA_VISIBLE_DEVICES"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    source_name="SLURM_STEP_GPUS"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    source_name="SLURM_JOB_GPUS"
  elif [[ -n "${SLURM_GPU_INDEX:-}" ]]; then
    source_name="SLURM_GPU_INDEX"
  else
    source_name="LOCAL_RANK fallback"
  fi
fi

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

devices=()
expand_device_list "$source_value" devices
if (( world > 1 )); then
  if (( ${#devices[@]} < world )); then
    echo "ERROR rank_wrapper: cannot map ${world} ranks from ${source_name}=${source_value:-<unset>}" >&2
    exit 86
  fi
  if (( rank < 0 || rank >= world )); then
    echo "ERROR rank_wrapper: LOCAL_RANK=${rank} outside world size ${world}" >&2
    exit 87
  fi
  selected="${devices[$rank]}"
else
  selected="${devices[0]:-${rank}}"
fi

export ORIGINAL_CUDA_VISIBLE_DEVICES="$original_cuda"
export RANK_SELECTED_CUDA_DEVICE="$selected"
export CUDA_VISIBLE_DEVICES="$selected"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "rank_wrapper rank=${rank} world=${world} source=${source_name} source_value=${source_value:-<unset>} selected=${selected} process_visible=${CUDA_VISIBLE_DEVICES}" >&2

if [[ "${RANK_WRAPPER_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  echo "ERROR rank_wrapper: missing Python training command" >&2
  exit 88
fi

exec python "$@"
