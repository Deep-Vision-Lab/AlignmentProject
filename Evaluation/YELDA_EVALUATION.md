# Evaluation of the original letter-depiction branches

This evaluator is for `agent/vlm-letter-depiction-hierarchy` and
`agent/vlm-letter-depiction-cross-attention`. It restores the original trainable
full-window Conv2D projection and residual depiction MLP before the visual
Transformer. It deliberately rejects window-CNN checkpoints; evaluate those in
their corresponding `*-window-cnn` branches.

No AraBERT model or tokenizer is constructed. The cross branch loads only the
saved `pair_cross_attention.*` tensors from the checkpoint's text-side state.
Both visual and pair-attention weights load strictly: missing or unexpected
weights stop evaluation instead of leaving randomly initialized parameters.

| Representation | Hierarchy | Cross-attention |
|---|---|---|
| `joint` (default) | Weighted local + independent contextual cosine | Weighted local + fused contextual cosine |
| `primary` | Independent contextual vectors | Contextual vectors after bidirectional image-pair fusion |
| `independent` | Same as primary | Contextual vectors before pair fusion |
| `local` | Depiction-head output before Transformer | Depiction-head output before Transformer |

All modes use cosine similarity, the same ink-aware scores, and global
Needleman–Wunsch with component masks. Defaults: raw cosine minus 0.45,
gap -0.30, minimum ink 0.02. The local representation on these original branches
includes the depiction MLP; it is not the primitive patch projection.

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

## Interactive commands

Run in an allocated GPU terminal, after activating `manucripts_align` and pulling
the corresponding branch. These launch in the foreground and show progress.

Hierarchy (default weights match its training script's `vit_vlm_letters` job):

```bash
WEIGHTS="$PWD/Weights/vit_vlm_letters/model_best.pth" bash scripts/eval_yelda.sh
```

Cross-attention (default weights match its `vit_vlm_cross` job):

```bash
WEIGHTS="$PWD/Weights/vit_vlm_cross/model_best.pth" bash scripts/eval_yelda.sh
```

Use your actual checkpoint path if you trained with a different job name.
The checkpoint must exist. Use `model_latest.pth` explicitly for an interim
evaluation while training continues. A checkpoint modified in place during
loading is rejected; loaded weights are then fixed for the evaluation run.

For local or independent vectors, prefix the command with
`REPRESENTATION=local` or `REPRESENTATION=independent`. `DEVICE=cpu` is supported.
The command-line equivalent on the cross branch is:

```bash
python -u -m Evaluation.eval_yelda \
  --dataset DataSet/Synthetic63 \
  --weights Weights/vit_vlm_cross/model_best.pth \
  --branch cross --representation joint --local-weight 0.5 --split test \
  --training-samples 6000 --split-seed 42 --n-samples 100 --device cuda \
  --output-dir Results/Evaluation/Yelda/original_cross_test
```

## Data and outputs

The default selects 100 held-out Synthetic63 test pairs, reproducing PyTorch's
60/20/20 split of the first 6000 samples with seed 42. The split population and
seed are explicit evaluation settings because old checkpoints do not record
both. Match `TRAINING_SAMPLES` and `SPLIT_SEED` to the actual training run.
`N_SAMPLES=0` evaluates the whole selected split. Tune scores on
`EVAL_SPLIT=valid`, then keep those settings fixed on test.

For real data, set `DATASET="$PWD/DataSet/ArabicDataset"`. Real manifests use the
existing balanced page-pair group split and round-robin selection within it.
This is the zero-shot evaluation split, not a reconstruction of an arbitrary
real fine-tuning split. Explicit split manifests are supported; generic manifests
without split membership require `EVAL_SPLIT=all`.

Training-equivalent deterministic preprocessing is applied to both domains.
Checkpoint geometry overrides defaults; missing preprocessing flags use branch
defaults, with resolved geometry/binarization recorded in each pair report.
Predicted regions are mapped back through crop/scale/padding to source-image
coordinates before comparison against ground-truth masks.

Results go under `Results/Evaluation/Yelda/<job>/<split>/<representation>_<time>/`:

- `run.json` records the checkpoint SHA-256, epoch, model configuration, evaluator
  revision, feature stage, split and scoring settings.
- `selected_pairs.json` lists exact image paths and sample IDs.
- `samples.csv` and `summary.json` report NW scores, path cosines, mask
  IoU/Dice/precision/recall, GT counts, and errors.
- Each pair has model-input images, line overlays, annotated cosine/match/DP
  heatmaps, numerical CSV/NPY matrices, traceback evidence, and predicted masks
  in model and source coordinates.

Mask metrics measure region coverage, not per-character or correspondence
accuracy. Missing ground-truth masks produce null metrics. Any failed pair
causes a nonzero process exit while preserving partial results. Existing nonempty
output directories are rejected to avoid mixing runs.

## Checks

```bash
python -m pytest tests/test_yelda_evaluation.py -q
```

The CPU checks cover original-head checkpoint reconstruction, exact visual/fused
features, missing weights, incompatible window-CNN rejection, deterministic
splits, source-coordinate geometry, and synthetic/real reporting. Cross-specific
tests skip on hierarchy. They use generated inputs and test checkpoints and do
not establish trained-model accuracy.

