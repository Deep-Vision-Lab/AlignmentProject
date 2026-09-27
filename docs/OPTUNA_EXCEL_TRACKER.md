# Optuna alignment search and Excel tracker

Install the project dependencies from `requirements.txt`, then run from the
repository root:

```bash
python scripts/train/run_optuna.py \
  --dataset DataSet/ArabicDataset \
  --n-trials 30 --epochs 20 \
  --study-name alignment_no_pruning_v2
```

On the project SLURM cluster, submit the same arguments with
`sbatch --job-name=my_optuna_job scripts/train/optuna.sbatch --dataset DataSet/ArabicDataset --n-trials 30 --epochs 20`.
The Optuna job uses one GPU and writes the Excel file to the shared checkout.

Every Optuna job owns one result folder: `results/<job-name>/`. The job name
is the explicit `--job-name` if given, otherwise `$SLURM_JOB_NAME` under
SLURM, otherwise the study name for local runs (sanitized to
`A-Z a-z 0-9 _ - .`). The workbook is
`results/<job-name>/optuna_alignment_experiment_tracker.xlsx`.
It exists before the first trial. After each trial finishes, the Optuna
callback reads that trial's saved `history.json`, writes both train and
validation rows for **every** completed epoch, updates `Trials`, physically
sorts `RankedResults`, refreshes `Dashboard`, and atomically replaces the
workbook. A failed trial retains all epochs that finished before its error.
Optimization pruning is disabled; valid trials run all configured epochs.
SQLite study state is stored in `results/<job-name>/study.db` by default.
Checkpoints live in `results/<job-name>/checkpoints/trial_NNNNN/` and logs in
`results/<job-name>/logs/trial_NNNNN.log`; the sampled search space and job
metadata are recorded in `results/<job-name>/search_space.json`. Explicit
`--tracker`, `--checkpoint-root`, `--log-root`, and `--storage` arguments
override these defaults.

`RankedResults` orders only `COMPLETE` trials at the top. Final validation
Alignment F1 takes precedence when available, then Mask IoU, then lowest
total loss. If final values of the same metric differ by at most `1e-4`,
validation improvement from epoch 1 breaks the tie. `FAILED` trials follow
completed trials without a rank number. Workbook objective
cells show the actual metric value: F1 and IoU are higher-is-better, and
loss is lower-is-better. The `Learning_Improvement` column is positive for
either rising F1/IoU or falling loss.

The current training dataset does not supply alignment or mask labels, so
Alignment F1 and Mask IoU remain blank. Validation gradients are blank
because validation runs without backpropagation. Training gradient norms
refer to the CNN, Transformer, and fusion modules. Positive similarity is
the average best visual-window cosine for each transcript character;
negative similarity is the average best cosine to an Arabic alphabet
character absent from the transcript. Similarity matrix summaries are for
visual windows against that line's own transcript. GPU memory is the peak
allocated memory during each split, in MiB.

Use `--base-config config.json` to override fixed `Config` defaults and
`--search-space choices.json` to provide categorical lists for the seven
searched parameters shown in `SearchSpace` (fusion, vector dimension,
transformer layers/heads, window size, CNN layers, SIGReg weight). Dropout
(0.1), DTW gamma (0.5), stride ratio (0.5, i.e. 50% overlap), and the simple
CNN (`cnn_type=simple`, `cnn_pretrained=False`) are fixed constants, recorded
in the `FixedParameters` sheet and on the `Dashboard` marked `(FIXED)`.
`--max-batches` is for smoke runs only;
zero uses full splits. A run can resume its Optuna study with the same
`--study-name`, `--storage`, and workbook path; existing trial rows are
upserted by study-qualified trial ID.

The Optuna objective is `-validation positive_dtw` (maximize): total loss is
not comparable across trials because SIGReg ON adds `0.3 * sigreg` to it, so
ranking uses validation positive DTW only. The workbook still records total,
DTW, and SIGReg losses for every epoch.

The search samples vector dimensions `64`, `128`, and `256` with heads
`[1, 2, 4]`; only dimension-divisible heads are offered per trial, so no
invalid vector/head combination is ever launched. Old studies recorded with a
different searched-parameter set (including the pre-`cnn_layers` layout and
the dropout/DTW-gamma/stride search) **cannot safely resume**; the runner
rejects them before starting a trial. Use a new `--study-name` and job name.
