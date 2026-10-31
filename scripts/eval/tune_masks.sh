#!/usr/bin/env bash
# Submission only: no checkpoint loading, inference, or optimization on the login node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT"
checkpoint="Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt"
dataset="DataSet/ArabicData"
split_manifest=""
job_name="mask_optuna"
output_dir=""
cache_dir=""
journal_file=""
study_name=""
seed=42
stage1=300
stage2=200
fpr_limit=0.05
cpus=8
memory=48G
gpus=1
gpu_type=rtx_4090
partition=rtx4090
account=""
qos=""
walltime=03-00:00:00
conda_env=manucripts_align
resume=0
dry_run=0
usage() {
  cat <<'HELP'
Usage: bash scripts/eval/tune_masks.sh --split-manifest FILE [options]
  --checkpoint FILE       Default: Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt
  --dataset DIR           Default: DataSet/ArabicData
  --split-manifest FILE   Required reviewed validation/test pair manifest (no presumed negatives)
  --job-name NAME         Default: mask_optuna
  --output-dir DIR        Default: Results/Evaluation/<job-name> (durable project storage)
  --cache-dir DIR         Default: <output-dir>/cache
  --journal-file FILE     Default: <output-dir>/optuna.journal (NFS-compatible journal)
  --study-name NAME       Default: job name; keep it unchanged on resume
  --seed INT              Default: 42
  --stage1-trials INT     Total completed budget, default 300
  --stage2-trials INT     Total completed budget including reused seeds, default 200
  --max-negative-pair-fpr FLOAT  Predeclared operating limit, default 0.05
  --resume | --no-resume  Continue the same output/studies; never redo completed trials
  --cpus INT --mem SIZE --gpus 0|1 --gpu-type TYPE --time LIMIT
  --partition NAME --account NAME --qos NAME --conda-env NAME
  --dry-run               Print the exact submission command without submitting
Only sbatch is run here. Cache generation and both tuning stages run on the compute node.
Final test is never run by this launcher. stdout prints only the submitted job ID.
HELP
}
need_value() { if [[ $# -lt 2 || -z "${2-}" ]]; then echo "Missing value for $1" >&2; exit 2; fi; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) need_value "$@"; checkpoint="$2"; shift 2 ;;
    --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
    --split-manifest) need_value "$@"; split_manifest="$2"; shift 2 ;;
    --job-name) need_value "$@"; job_name="$2"; shift 2 ;;
    --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
    --cache-dir) need_value "$@"; cache_dir="$2"; shift 2 ;;
    --journal-file) need_value "$@"; journal_file="$2"; shift 2 ;;
    --study-name) need_value "$@"; study_name="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --stage1-trials) need_value "$@"; stage1="$2"; shift 2 ;;
    --stage2-trials) need_value "$@"; stage2="$2"; shift 2 ;;
    --max-negative-pair-fpr) need_value "$@"; fpr_limit="$2"; shift 2 ;;
    --cpus) need_value "$@"; cpus="$2"; shift 2 ;;
    --mem|--memory) need_value "$@"; memory="$2"; shift 2 ;;
    --gpus) need_value "$@"; gpus="$2"; shift 2 ;;
    --gpu-type) need_value "$@"; gpu_type="$2"; shift 2 ;;
    --partition) need_value "$@"; partition="$2"; shift 2 ;;
    --account) need_value "$@"; account="$2"; shift 2 ;;
    --qos) need_value "$@"; qos="$2"; shift 2 ;;
    --time) need_value "$@"; walltime="$2"; shift 2 ;;
    --conda-env) need_value "$@"; conda_env="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --no-resume) resume=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$job_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid job name' >&2; exit 2; }
for value in "$seed" "$stage1" "$stage2" "$cpus" "$gpus"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Seeds, budgets and resource counts must be nonnegative integers' >&2; exit 2; }
done
[[ "$cpus" -gt 0 && "$gpus" -le 1 ]] || { echo 'Use positive CPUs and at most one GPU for this one-worker evaluator' >&2; exit 2; }
[[ -f "$checkpoint" && -d "$dataset" && -n "$split_manifest" && -f "$split_manifest" ]] || {
  echo 'Checkpoint, dataset directory, and a real reviewed --split-manifest file must exist. No job submitted.' >&2; exit 2;
}
absolute_file() { printf '%s/%s' "$(cd "$(dirname "$1")" && pwd -P)" "$(basename "$1")"; }
checkpoint="$(absolute_file "$checkpoint")"
dataset="$(cd "$dataset" && pwd -P)"
split_manifest="$(absolute_file "$split_manifest")"
output_dir="${output_dir:-$ROOT/Results/Evaluation/$job_name}"
[[ "$output_dir" == /* ]] || output_dir="$ROOT/$output_dir"
cache_dir="${cache_dir:-$output_dir/cache}"
journal_file="${journal_file:-$output_dir/optuna.journal}"
[[ "$cache_dir" == /* ]] || cache_dir="$ROOT/$cache_dir"
[[ "$journal_file" == /* ]] || journal_file="$ROOT/$journal_file"
for path in "$output_dir" "$cache_dir" "$journal_file"; do
  case "$path" in /tmp|/tmp/*|/private/tmp|/private/tmp/*|/var/tmp|/var/tmp/*|/dev/shm|/dev/shm/*)
    echo "Choose durable project storage, not temporary storage: $path" >&2; exit 2 ;;
  esac
done
mkdir -p "$output_dir/logs"
output_dir="$(cd "$output_dir" && pwd -P)"
study_name="${study_name:-$job_name}"
device=cuda; [[ "$gpus" -ne 0 ]] || device=cpu
submit=(sbatch --parsable "--chdir=$ROOT" "--job-name=$job_name" "--partition=$partition"
  --ntasks=1 "--cpus-per-task=$cpus" "--mem=$memory" "--time=$walltime"
  "--output=$output_dir/logs/%x_%j.out" "--error=$output_dir/logs/%x_%j.err"
  --open-mode=append --signal=B:USR1@120 --export=ALL)
if [[ "$gpus" -eq 0 ]]; then
  submit+=(--gpus=0) # Override the batch file's one-GPU default for CPU smoke runs.
else
  submit+=("--gpus=$gpu_type:$gpus")
fi
[[ -z "$account" ]] || submit+=("--account=$account")
[[ -z "$qos" ]] || submit+=("--qos=$qos")
submit+=("$ROOT/scripts/eval/tune_masks.sbatch" --checkpoint "$checkpoint" --dataset "$dataset"
  --split-manifest "$split_manifest" --output "$output_dir" --cache-dir "$cache_dir"
  --storage "journal://$journal_file" --study-name "$study_name" --seed "$seed"
  --stage1-trials "$stage1" --stage2-trials "$stage2" --startup-trials 40 --threads "$cpus" --device "$device"
  --max-negative-pair-fpr "$fpr_limit" --fail-fast)
[[ "$resume" -eq 0 ]] || submit+=(--resume)
export MASK_CONDA_ENV="$conda_env" MASK_OUTPUT_DIR="$output_dir"
if [[ "$dry_run" -eq 1 ]]; then printf '%q ' "${submit[@]}"; printf '\n'; exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch is unavailable; run this submission script on BGU HPC' >&2; exit 2; }
reply="$("${submit[@]}")"
job_id="${reply%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch reply: $reply" >&2; exit 1; }
printf '%s\n' "$job_id"
