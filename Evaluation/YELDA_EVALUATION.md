# Yelda window-CNN evaluation

Use `python -m Evaluation.eval_yelda` for the two current experiments. This
entry point reconstructs the architecture from `model_config` and strictly
loads the CNN, visual Transformer, and (for cross checkpoints) saved pair-attention
weights. It never constructs AraBERT or a tokenizer and performs no text inference
or Hugging Face downloads. Text fields in real manifests are dataset metadata only.

| Representation | Hierarchy checkpoint | Cross-attention checkpoint |
|---|---|---|
| `joint` (default) | Weighted local + independent contextual cosine | Weighted local + fused contextual cosine |
| `primary` | Independent contextual C1/C2 | Fused F1/F2 after bidirectional image-pair attention |
| `local` | Direct CNN window vectors, before Transformer | Same stage, without pair fusion |
| `independent` | Same as primary | Contextual C1/C2 before pair fusion |

Every mode computes cosine similarity, applies the same ink-aware score policy,
runs global Needleman–Wunsch, and extracts supported matching components/masks.
The default scoring is raw cosine minus 0.45, gap -0.30, minimum ink 0.02.
The CNN windows are 32 pixels wide, 128 high, with the checkpoint stride (16 in
the current runs); a 1024-pixel canvas therefore gives 63 windows.

## Joint local + contextual alignment (default)

Joint mode builds one score matrix:

`S_joint = local_weight * cosine(L1, L2) + (1 - local_weight) * cosine(C1, C2)`.

On cross-attention checkpoints, C1/C2 in this formula are the fused F1/F2 vectors.
Local features are preserved. Both terms use the same window indices and are
mixed before score normalization, thresholding, ink masking and a single global
NW alignment. The default local weight is 0.5; use validation data to select a
weight and keep it fixed on test. Weights must be finite and in [0, 1].

Use `--representation joint --local-weight 0.5` with the Python entry point.
The launchers default to joint mode and accept `LOCAL_WEIGHT=0.5`. Existing
`primary`, `local`, and `independent` modes remain available for comparisons.

Each joint evaluation saves `local_cosine_similarity`,
`contextual_cosine_similarity`, and `joint_similarity` as both NPY and CSV.
On the cross branch the contextual matrix uses fused features. The existing
`cosine_similarity.npy` and main similarity heatmap now contain the combined
matrix in joint mode; the plot title labels the two weights. Per-pair reports
record the local weight, and run/summary JSON records it in `arguments` alongside
`feature_stage=joint_local_contextual` or `joint_local_fused_contextual`.

## Synthetic evaluation

The current training population is the first 6000 numbered samples of Synthetic63,
split 3600/1200/1200 by PyTorch random_split with seed 42. Evaluation reproduces
that permutation and defaults to 100 test pairs. It does not take the first 100
files or include the unused samples 6001–10000. `--training-samples` and
`--split-seed` are explicit evaluation settings because these older checkpoints
do not record both values. Match them to the original training run if it differs.
The exact selected image paths and IDs are saved for comparison.

Run in the corresponding branch checkout after obtaining a checkpoint:

```bash
mkdir -p out

# Hierarchy
PROJECT_DIR="$PWD" \
WEIGHTS="$PWD/Weights/yelda_cnn_hierarchy/model_best.pth" \
sbatch --job-name=yelda_eval_hierarchy scripts/eval_yelda.sbatch

# Cross-attention
PROJECT_DIR="$PWD" \
WEIGHTS="$PWD/Weights/yelda_cnn_cross/model_best.pth" \
REPRESENTATIONS="joint primary independent local" \
sbatch --job-name=yelda_eval_cross scripts/eval_yelda.sbatch
```

The script uses one RTX4090 and the `manucripts_align` environment. Its default
dataset is `DataSet/Synthetic63`; it evaluates the joint representation
unless `REPRESENTATIONS` is specified. All representations use a single snapshot
of the checkpoint, retained in the result directory. This also supports ongoing
training: a saved checkpoint must exist, and a file being overwritten during the
copy is rejected. Each SLURM job gets a separate result directory.

Use `N_SAMPLES=0` for all 1200 test pairs (this writes detailed plots for every
pair). Use `EVAL_SPLIT=valid` for threshold/gap selection, then keep those settings
fixed on test. A `model_latest.pth` path can be supplied explicitly for a progress
check; `model_best.pth` is the default checkpoint for the final evaluation.

Direct single-mode command:

```bash
python -m Evaluation.eval_yelda \
  --dataset DataSet/Synthetic63 \
  --weights Weights/yelda_cnn_cross/model_best.pth \
  --branch cross --representation joint --local-weight 0.5 \
  --split test --training-samples 6000 --split-seed 42 \
  --n-samples 100 --score-mode raw --threshold 0.45 --gap -0.30 \
  --output-dir Results/Evaluation/Yelda/cross_test_primary
```

An existing nonempty output directory is rejected to prevent mixing experiments.

## Real manuscript evaluation

```bash
PROJECT_DIR="$PWD" \
WEIGHTS="$PWD/Weights/yelda_cnn_cross/model_best.pth" \
DATASET="$PWD/DataSet/ArabicDataset" EVAL_SPLIT=test \
REPRESENTATIONS="joint primary independent local" \
sbatch --job-name=yelda_eval_cross_real scripts/eval_yelda.sbatch
```

Replace the weight path and job name for hierarchy. Real ArabicDataset evaluation
uses the existing label-filtered, balanced page-pair group split, then samples
round-robin across groups. This is the existing zero-shot evaluation split; it is
not a claim to reproduce a separately fine-tuned real training split. Explicit
train/valid/test manifests are also supported. A generic manifest without split
membership requires `--split all` (or `EVAL_SPLIT=all`).

Both domains use the deterministic training preprocessor: foreground cropping,
source-compatible ink-height normalization, and the configured binarization,
followed by ImageNet tensor normalization. Checkpoint geometry takes precedence;
preprocessing flags missing from older checkpoints use the branch defaults.
Resolved preprocessing is recorded per image. Model-side binarization remains
controlled by the checkpoint's `vit_binarize_input` flag.

## Outputs and interpretation

Results default to:

```text
Results/Evaluation/Yelda/<weight-folder>/<dataset>/<split>/<job-id>/<representation>/
```

- `run.json`: checkpoint SHA-256, epoch, full model configuration, evaluation
  revision, selected feature stage, geometry, scoring and component settings.
- `selected_pairs.json`: the exact evaluation sample list.
- `samples.csv` and `summary.json`: per-pair and aggregate NW scores, path
  similarities, mask IoU/Dice/precision/recall, GT counts, and failures.
- Per pair: normalized model inputs; line overlays; annotated cosine, match-score
  and accumulated-DP heatmaps; CSV/NPY matrices; full traceback and supported
  component evidence; predicted masks in model and original source coordinates.

Mask metrics compare full-height predicted alignment regions with supplied masks
in the original source-image coordinates. Crop/scale/padding are inverted before
comparison; the raw GT is never stretched to the model canvas. These metrics
measure region coverage, not correctness of each window-to-window correspondence
or character recognition. Missing GT yields null metrics and a zero GT count.
Path cosine/NW scores are diagnostics, not ground-truth accuracy.

The command returns a nonzero exit status if any pair fails; partial reports
retain the errors rather than silently presenting a successful evaluation.

## Validation

`python -m pytest tests/test_yelda_evaluation.py -q` checks exact visual checkpoint
restoration, fused-feature equivalence to the training attention module, missing
weight rejection, the training synthetic split, source-coordinate geometry, and
synthetic/real end-to-end reporting. Cross-specific tests skip in the hierarchy
branch, where the pair-attention implementation is intentionally absent.
These are CPU checks with generated inputs and test checkpoints, not trained-model
results. A separate smoke run exercises and visually checks the actual renderer.

