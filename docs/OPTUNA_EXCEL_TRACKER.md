# Optuna alignment search and Excel tracker

Install the project dependencies from `requirements.txt`, then run from the
repository root:

```bash
python scripts/train/run_optuna.py \
  --dataset DataSet/ArabicDataset \
  --n-trials 30 --epochs 20
```

On the project SLURM cluster, submit the same arguments with
`sbatch scripts/train/optuna.sbatch --dataset DataSet/ArabicDataset --n-trials 30 --epochs 20`.
The Optuna job uses one GPU and writes the Excel file to the shared checkout.

The workbook is `results/optuna/optuna_alignment_experiment_tracker.xlsx`.
It exists before the first trial. After each trial finishes, the Optuna
callback reads that trial's saved `history.json`, writes both train and
validation rows for **every** completed epoch, updates `Trials`, physically
sorts `RankedResults`, refreshes `Dashboard`, and atomically replaces the
workbook. A failed or pruned trial retains all epochs that finished before
it stopped. SQLite study state and per-trial checkpoints/logs are also under
`results/optuna/`.

`RankedResults` orders only `COMPLETE` trials at the top. Final validation
Alignment F1 takes precedence when available, then Mask IoU, then lowest
total loss. If final values of the same metric differ by at most `1e-4`,
validation improvement from epoch 1 breaks the tie. `PRUNED` and `FAILED`
trials follow completed trials without a rank number. Workbook objective
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
`--search-space choices.json` to provide categorical lists for the nine
parameters shown in `SearchSpace`. `--max-batches` is for smoke runs only;
zero uses full splits. A run can resume its Optuna study with the same
`--study-name`, `--storage`, and workbook path; existing trial rows are
upserted by trial ID.
