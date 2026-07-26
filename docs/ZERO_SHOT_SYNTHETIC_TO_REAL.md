# Strict Zero-Shot Synthetic-to-Real Alignment

This profile trains only on synthetic Arabic line pairs and evaluates on real
`DataSet/ArabicDataset` images. It does not use real transcripts, real alignment
labels, pseudo-labels, or target-domain fine-tuning.

## Why the old checkpoint transferred poorly

The previous synthetic transform stretched every line directly to `128x1024`
and kept anti-aliased grayscale pixels. Real evaluation resized, autocontrasted,
and Otsu-binarized the scan. The model therefore saw different geometry, stroke
statistics, and image intensity distributions at training and evaluation time.
Real blank windows also received high similarity and created long local SW paths.

## Changes

### Shared geometry

`zero_shot_preprocessing.ManuscriptLinePreprocessor`:

1. converts the line to grayscale and autocontrasts it;
2. estimates foreground polarity with Otsu;
3. crops excess page margins around foreground ink;
4. scales the cropped line without changing its aspect ratio;
5. centers and pads it to `128x1024`;
6. binarizes it and forces a white page background.

The target foreground height defaults to 72% of the 128-pixel canvas. Very wide
lines are limited by the 1024-pixel width instead of being geometrically warped.

### Synthetic manuscript domain randomization

Only the synthetic training split receives random degradation. Validation and
test remain deterministic. The augmentations include:

- horizontal and vertical scale variation;
- small rotation;
- Gaussian blur;
- black-stroke erosion or dilation;
- grayscale scanner noise;
- short faded/broken stroke regions;
- dust and bleed-through-like speckles;
- randomized threshold around Otsu;
- horizontal placement jitter after padding.

Twenty percent of samples remain clean by default so the source distribution is
not discarded.

### Local and grouped feature supervision

The existing local hard-negative loss previously received only raw local CNN
windows. Under the zero-shot profile it receives a 50/50 blend of raw local and
three-window grouped features. This keeps the existing efficient hard-path reuse
while training both fine stroke evidence and short-range context.

### Domain-robust normalization

The default zero-shot mode freezes all ResNet BatchNorm layers in evaluation
mode and freezes their affine parameters. Synthetic-only batches therefore do
not overwrite ImageNet running statistics with source-specific binary moments.
`ZERO_SHOT_NORM_MODE=groupnorm` is available as an architectural ablation.

### Ink-aware Smith-Waterman

Before local DP, window scores are corrected using the model's ink estimate:

- blank versus blank is capped at `-0.20`;
- blank versus ink is capped at `-0.50`;
- ink versus ink keeps the selected raw/centered/mutual-z score.

This prevents empty page margins from accumulating into a long false alignment.

### Diverse real evaluation

ArabicDataset groups remain isolated by `pair_id`. When at least six groups are
available, validation and test are each seeded with at least two complete groups.
Batch output is round-robin across `pair_id`, so the first 20 figures do not all
come from one page pair.

## Train a new strict zero-shot checkpoint

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git checkout agent/training-speed-optimization
git pull --ff-only origin agent/training-speed-optimization

JOB_ID=synthetic_arabic_zero_shot_8k \
NUM_SAMPLES=8000 \
NUM_GPUS=2 \
bash scripts/train/run_zero_shot_synthetic_to_real.sh
```

The launcher delegates to the optimized Slurm launcher and records every
zero-shot setting in `model_config` inside the checkpoint.

## Evaluate the real dataset

```bash
conda activate manucripts_align

export WEIGHTS="$PWD/Weights/synthetic_arabic_zero_shot_8k/model_latest.pth"
export ZERO_SHOT_PREPROCESS=1
export REAL_EVAL_BALANCED=1
export SW_INK_AWARE=1
export SW_MIN_INK=0.02
export SW_BLANK_BLANK_SCORE=-0.20
export SW_BLANK_INK_SCORE=-0.50

python -m Evaluation.eval_img_align_sw \
  --weights "$WEIGHTS" \
  --data-dir DataSet/ArabicDataset \
  --dataset-type real \
  --batch \
  --real-split test \
  --start-index 1 \
  --n-samples 20 \
  --feature contextual \
  --score-mode auto \
  --score-clip 4.0 \
  --threshold 0.45 \
  --gap -0.30 \
  --heatmap-source dp-score \
  --output-dir Results/Evaluation/SW/ArabicDataset_zero_shot
```

## Recommended ablations

Keep all settings fixed and change one variable at a time:

```bash
# Feature level
--feature local
--feature grouped
--feature contextual

# Disable blank-window correction
SW_INK_AWARE=0

# Use the old stretched geometry
ZERO_SHOT_PRESERVE_ASPECT=0
ZERO_SHOT_FOREGROUND_CROP=0

# Normalization alternatives for a new training run
ZERO_SHOT_NORM_MODE=train-bn
ZERO_SHOT_NORM_MODE=groupnorm
```

Do not tune these choices on the labeled real test set. Select them using
synthetic validation with held-out degradation families, then report real-test
performance once.
