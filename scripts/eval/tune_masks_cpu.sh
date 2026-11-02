#!/usr/bin/env bash
# Submission only. Existing CUDA features and the original journal are reused.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
output="Results/Evaluation/mask_optuna_balanced"
job_name=mask_optuna_cpu
partition=cpu
account=jelsana
qos=normal
cpus=4
memory=24G
walltime=5-00:00:00
conda_env=manucripts_align
verify_only=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) output="$2"; shift 2 ;;
    --job-name) job_name="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --account) account="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --mem) memory="$2"; shift 2 ;;
    --time) walltime="$2"; shift 2 ;;
    --conda-env) conda_env="$2"; shift 2 ;;
    --verify-only) verify_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$cpus" =~ ^[1-9][0-9]*$ && "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo 'Invalid CPU count or job name' >&2; exit 2;
}
[[ "$output" == /* ]] || output="$ROOT/$output"
[[ -f "$output/run_manifest.json" && -f "$output/optuna.journal" && -d "$output/cache" ]] || {
  echo 'Existing CUDA study, journal, or cache is missing; no job submitted' >&2; exit 2;
}
mkdir -p "$output/logs"
output="$(cd "$output" && pwd -P)"
submit=(sbatch --parsable "--chdir=$ROOT" "--job-name=$job_name" "--partition=$partition"
  "--account=$account" "--qos=$qos" --ntasks=1 "--cpus-per-task=$cpus" "--mem=$memory"
  "--time=$walltime" --gpus=0 "--output=$output/logs/%x_%j.out"
  "--error=$output/logs/%x_%j.err" --open-mode=append --signal=B:USR1@120
  --export=ALL "$ROOT/scripts/eval/tune_masks_cpu.sbatch")
[[ "$verify_only" -eq 0 ]] || submit+=(--verify-only)
export MASK_OUTPUT_DIR="$output" MASK_CONDA_ENV="$conda_env"
if [[ "$dry_run" -eq 1 ]]; then printf '%q ' "${submit[@]}"; printf '\n'; exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch is unavailable; submit on the cluster login node' >&2; exit 2; }
"${submit[@]}"
