# Unattended BGU quantitative-mask search

Run from the current **AlignmentProject** checkout. `tune_masks.sh` only submits
an sbatch job: model loading, validation cache generation, the exact notebook
baseline and both Optuna stages run on the allocated compute node. The notebook,
matcher, masks, training, model and loss are unchanged. Final test is separate.

## Environment and data required before submission

The launcher follows `scripts/train/optuna.sbatch` and
`scripts/eval/evaluate.sbatch`: partition `rtx4090`, GPU `rtx_4090:1`, Conda
activation via `conda info --base` and environment **manucripts_align**. No existing
script specifies an account or QoS, so site defaults are used unless supplied.
Default resources: 8 CPUs, 48G RAM, one GPU, three days. Override with `--cpus`,
`--mem`, `--gpus 0|1`, `--gpu-type`, `--partition`, `--time`, `--account`, `--qos`.
`--gpus 0` is available for bounded CPU smoke tests. The evaluator has one worker
and does not benefit from requesting multiple GPUs.

Prepare dependencies in `manucripts_align` before submitting, including the
repository's PyTorch/torchvision dependencies and **Optuna 4.x or 5.x** (the
worker checks the required APIs rather than rejecting a compatible major version).
The worker checks constrained TPE, crash-recoverable journal locks,
CUDA and strict checkpoint reconstruction before cache generation. Checkpoint
loading uses `build_model(..., initialize=False)` and loads its complete saved
state; it does not fetch pretrained weights. All inputs must already be local
to the HPC project storage. The job never installs packages, downloads model
assets, asks questions, or connects to an interactive authentication service.
Hugging Face offline flags, `PIP_NO_INDEX`, offline W&B and unbuffered Python are
set before the worker starts. Missing dependencies/assets cause a clear failure.
The batch launcher also exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before starting
Python**, as required by the evaluator's deterministic CUDA setting. An explicitly
supplied `:16:8` is also accepted; other values fail before inference. The
preflight log prints the resolved value. Do not set it after PyTorch has started.

The real checkpoint currently present is
`Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt`; dataset root is
`DataSet/ArabicData`. By default, the worker uses `alignedPairs` masks and
seeded cross-book comparisons from `Unalignedlines`, restricted to the checkpoint's
saved validation membership. Missing masks are black under the notebook policy;
these comparison negatives are explicitly marked presumed in results. The current
exports supply 53 validation positives and 156 original unmatched comparisons.
The default `--class-balance balanced` selects **53 positives + 53 negatives =
106 validation pairs per trial**. The 500-pair target is an upper limit: 250 +
250 cannot be formed from only 53 eligible positive pairs. Counts are reduced
equally, without repeating positives or changing saved membership. The separate
final-test catalog supports 88 + 88 balanced pairs.

`--class-balance natural --pairs-per-split 500` restores the previous unequal
53-positive/447-negative configuration. Its supplemental negatives use existing
coarse `no_shared_content` labels and pair-specific black GT. Both policies
retain all source/label provenance; negative labels remain explicitly presumed.
See
[quantitative_mask_evaluation.md](quantitative_mask_evaluation.md) for the contract.

A reviewed manifest is optional for `--data-policy folder_masks`; its negatives
are **presumed**, not verified. Use `--data-policy reviewed --split-manifest FILE`
for the strict explicit-label/mask workflow. The batch worker accepts either
policy consistently with the submission launcher. Both modes prohibit split fallback
and training/test contamination of validation. Final test is separate.

## Submit, monitor, resume and cancel

On BGU, first `cd` into your actual AlignmentProject checkout. No model or
dataset processing occurs in the submission shell. These commands work in Bash
or Zsh; keep the shell variables for monitoring and resume:

```bash
JOB_NAME=mask_optuna_balanced
OUT="$PWD/Results/Evaluation/$JOB_NAME"
SUBMIT=(bash scripts/eval/tune_masks.sh
  --checkpoint "$PWD/Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt"
  --dataset "$PWD/DataSet/ArabicData"
  --data-policy folder_masks --class-balance balanced --pairs-per-split 500
  --job-name "$JOB_NAME" --output-dir "$OUT"
  --seed 42 --stage1-trials 300 --stage2-trials 200)

# Optional: inspect the exact sbatch command without submitting.
"${SUBMIT[@]}" --dry-run

# 1. Submit: prints only the job ID on stdout.
JOB_ID=$("${SUBMIT[@]}")
echo "$JOB_ID"

# 2. Monitor.
squeue -j "$JOB_ID"

# 3. Follow logs; -F also waits for files while the job is queued.
tail -F "$OUT/logs/${JOB_NAME}_${JOB_ID}.out" "$OUT/logs/${JOB_NAME}_${JOB_ID}.err"

# 4. After the previous job has ended, resume the SAME output and budgets.
JOB_ID=$("${SUBMIT[@]}" --resume)

# 5. Cancel the current job. Completed results are preserved.
scancel "$JOB_ID"
```

Choose durable project storage for output/cache/journal. By default all three
live below `$OUT`. Optional `--cache-dir`, `--journal-file` and `--study-name`
allow explicit locations/names; retain them on resume. Node scratch and temporary
paths (including symlinks to them) are rejected by the worker. The implementation
probes write/fsync, atomic rename and exclusive symlink creation; it cannot
prove a site's backup/durability policy. Use your site's persistent project mount.

## Storage, budgets, failure and recovery

The job uses **one Optuna worker**, constrained TPE, 40 startup trials per study,
seed 42 and `NopPruner`. The default predeclared FPR limit is .05; override only
as an operating requirement with `--max-negative-pair-fpr`, never as a tuned
parameter. Every fresh completed trial evaluates the complete validation set.
Stage 2 also imports up to five feasible Stage 1 results that have identical
settings and full-validation scores, avoiding duplicate work.

Studies are saved in `optuna.journal` using Optuna `JournalStorage`,
`JournalFileBackend` and `JournalFileSymlinkLock`. This avoids SQLite's NFS
locking limitations. See the [Optuna journal recipe](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/011_journal_storage.html).
The job checks symlink-lock support; do not substitute SQLite on a shared mount.
The journal is the authoritative persistent record; CSVs are derived exports.

Trial budgets are **total desired COMPLETE counts**, not additional trials per
submission. Imported completed Stage 2 seeds count toward its budget; failed or
interrupted records do not. An unchanged 300/200 resume only evaluates the
remaining budget. Lowering budgets below existing completed counts is an error.
New trials after resume use the persistent history; restarting the seeded TPE
sampler is not claimed to reproduce a bitwise uninterrupted suggestion sequence.

An exclusive output lease prevents simultaneous batch writers. On resume the
worker checks the previous job against `squeue`; controller errors, live jobs,
unknown trial owners or incompatible provenance cause failure rather than
overwriting work. An old confirmed-dead owner's RUNNING trials become FAIL and
are replaced within the completed budget. SIGUSR1 is requested two minutes
before timeout; TERM/INT/USR1 preserve status/results. SIGKILL can prevent the
last CSV update, but the completed journal records remain available for resume.
Do not manually remove locks while a job may still be alive.

Use a new output/job name when changing the earlier unequal pool to balanced sampling; its
selected pair list and source fingerprints differ.

Dataset/GT hashes, checkpoint identity, preprocessing, evaluator/notebook source,
packages, seed, objective and search-space version must match on resume. Trial
budgets/resources may increase/change. Keep job/study names, input paths and
output/cache/storage flags unchanged. Existing incompatible studies require a
new output, not a forced resume. A frozen final-test selection cannot be tuned.

Each trial logs IoU, Dice, precision, recall, negative-pair FPR, negative masked
area, feasibility and seconds. First trial failure stops the batch with its
actual error; failed trials never receive zero-valued valid scores. Completed
exports remain. If neither stage has a feasible trial, exit code 3 reports
`NO_FEASIBLE_TRIAL`, preserves trade-offs, and produces no selected model.

Outputs include:

* `optuna.journal`, `trials.csv`, per-trial `aggregate.json/csv`, per-pair and
  per-line CSVs, diagnostics and complete requested/effective settings.
* `selected.json`, `best_effective_settings.json`, `notebook_settings.py` and
  `selection_status.json` (selected files exist only if a feasible trial exists).
* `cache/`, `baseline/`, `val_split_manifest.json`, `split_audit.json`,
  `run_manifest.json`, `val_provenance.json`, source/package/checkpoint identities.
* `logs/<job-name>_<job-id>.out/.err`, `batch_status.json` and durable
  `batch_attempts/` metadata (scheduler resources, owner, recovery, errors).

## Explicit final test and paper export

Never run final test during model selection. After both stages finish, this
separate command requests a compute allocation, freezes the validation-selected
configuration and evaluates test baseline/selected/extension-off/merging-off:

```bash
srun --partition=rtx4090 --gpus=rtx_4090:1 --cpus-per-task=8 --mem=48G --time=04:00:00 \
  bash -c 'set -euo pipefail
    assigned_cuda="${CUDA_VISIBLE_DEVICES-}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate manucripts_align
    export CUDA_VISIBLE_DEVICES="$assigned_cuda"
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1
    export WANDB_MODE=offline PYTHONUNBUFFERED=1
    exec python -u scripts/eval/tune_masks.py test "$@"' bash \
  --checkpoint "$PWD/Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt" \
  --dataset "$PWD/DataSet/ArabicData" --data-policy folder_masks --class-balance balanced --pairs-per-split 500 \
  --output "$OUT" --device cuda --threads 8 --seed 42

# Regenerate CSV/LaTeX tables from saved final-test results, without inference.
conda run -n manucripts_align python scripts/eval/tune_masks.py export --output "$OUT"
```

Add your site's account/QoS flags to `srun` if required. Match any nondefault
seed, preprocessing, cache directory and bootstrap options from tuning.
Folder mode applies the same class-balance policy to saved-test members. The
current balanced test catalog has 88 positive and 88 negative pairs. No test labels influence Optuna selection.
For reviewed mode, pass the same reviewed manifest and data policy used in tuning,
with positive and negative test entries belonging to the saved test membership.

## Local verification

```bash
bash -n scripts/eval/tune_masks.sh scripts/eval/tune_masks.sbatch
PYTHONPATH=.:tests .venv-mask-eval/bin/python -m pytest \
  tests/test_mask_batch.py tests/test_quantitative_masks.py -q
```

Tests mock sbatch/Conda/controller boundaries and run a bounded CPU journal
worker against synthetic frozen-checkpoint data. They verify completed budgets,
no-op resume, interrupted-trial replacement, live-owner rejection, login-node
guards, temporary-storage rejection, resource forwarding and metric logs.
This is not a real BGU submission or a real manuscript performance experiment.

Saved bounded run: `Results/Evaluation/mask_batch_smoke_20261031/`. It completed
6 Stage 1 and 8 Stage 2 records on one synthetic positive and one synthetic
negative pair, then resumed without changing any completed trial or cached NPZ.
Nine records met the .05 FPR requirement. The selected synthetic configuration
has IoU 0 and FPR 0; this verifies constrained selection, not useful manuscript
performance. The checkpoint was unchanged and final test was not run.
`verification.json`, `worker.log` and `resume.log` retain the evidence.
All **24 focused tests passed**; Bash syntax and Python compilation passed.
