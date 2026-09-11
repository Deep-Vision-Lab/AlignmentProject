# Restoration + Positive Letter-DTW Window Encoder

Branch: `agent/restoration-positive-dtw-window-encoder`

## Research hypothesis

Train every manuscript line independently. A window embedding should retain the
physical stroke information inside the window while also being learnable toward
a stable character identity. Corresponding shapes in different manuscripts are
never explicitly paired during training.

The active objective has exactly two conceptual losses:

```
L_total = 1.0 * L_positive_letter_DTW + 0.10 * L_restoration
```

No negative transcripts, image-image pair contrastive loss, cross-attention,
context-consistency loss, dense-letter loss, or variance loss are active.

## Architecture

```
full 128x1024 RGB line
        |
        | full-height sliding projection, window=32, stride=16
        v
P_1 ... P_T                         primitive stroke tokens
 |         |
 |         +---- lightweight decoder ----> reconstructed stroke map
 |
 +---- residual semantic adapter --------> L_1 ... L_T
                                             |
                                             | cosine costs to fixed letter IDs
                                             v
                                   positive monotonic Soft-DTW
                                   against this line transcript
```

There is one token for every physical horizontal window. The active model does
not use the inherited global Transformer. The shared Transformer container is
kept only for checkpoint/model compatibility and its parameters are frozen.

### Primitive token P_i

The full-height Conv2d window projection produces a 128-D vector for each
physical window. The restoration decoder is attached directly to this token.

Its responsibility is: **remember the visible stroke structure**.

### Stroke restoration

The decoder maps each primitive token back to a `1 x 128 x 32` soft stroke
map. It does not reconstruct parchment RGB. The target is local foreground
contrast computed from the original RGB window.

```
L_restoration = L_pixel + 0.5 * L_edge
```

`L_pixel` is foreground-weighted L1 (default foreground weight 2.0).
`L_edge` compares horizontal and vertical finite differences so boundaries,
dots, curves, and connections matter.

### Semantic token L_i

A residual MLP + LayerNorm maps `P_i` to `L_i`. This is the representation
used by the positive letter-DTW objective and the primary image-only evaluation.

Its responsibility is: **represent which character identity the visible strokes
belong to**.

### Fixed letter codebook

The text side is `OrthogonalCharEmbedding`, already available in this
repository. Every Unicode character has a deterministic frozen vector. There is
no AraBERT and no trainable text projection in this experiment.

The fixed vectors are not supposed to know what Arabic letters look like. They
are stable identity destinations. Repeated appearances of the same transcript
letter across many lines teach the visual encoder to map different handwritten
forms toward the same destination.

Therefore isolated glyph images are **not required**. A separate multi-font
glyph warm-up remains a later ablation if line-level weak supervision is hard to
bootstrap.

## Positive-only monotonic DTW

For cleaned transcript letters `c_1 ... c_L` and semantic window vectors
`L_1 ... L_T`:

```
C[i,j] = 1 - cosine(L_i, e_{c_j})
```

The differentiable DP permits:

- diagonal: next window, next letter;
- vertical: another window still belongs to the current letter;
- horizontal: one overlapping window can contain evidence from two adjacent letters.

Only the true transcript of the current line is used. There are no negative
transcripts and no margin against another line.

## Paired files are not pair supervision

Some datasets physically store `line1` and `line2` together. The branch keeps
both so data is not discarded, but computes:

```
loss = 0.5 * (loss(line1, text1) + loss(line2, text2))
```

There is no term that compares line1 with line2. They are two independent
single-line training examples sharing the same encoder and frozen letter
codebook.

## Training

```bash
git fetch origin
git checkout agent/restoration-positive-dtw-window-encoder
git pull

JOB_NAME=vit_restore_dtw_s16 \
sbatch scripts/train_restoration_positive_dtw_2x4090.sbatch
```

Default first experiment: window 32, stride 16.

A denser stride-8 experiment can follow:

```bash
JOB_NAME=vit_restore_dtw_s8 \
RESTORATION_DTW_STRIDE_RATIO=0.25 \
sbatch scripts/train_restoration_positive_dtw_2x4090.sbatch
```

## What to monitor

The important training signals are:

- `minimal/positive_letter_dtw` should decrease;
- `minimal/restoration` should decrease;
- `minimal/restoration_pixel` should decrease;
- `minimal/restoration_edge` should decrease.

Do not expect negative-gap metrics because this experiment intentionally has no
negative samples.

## Image-only evaluation

The evaluator never needs the transcript or text codebook.

```bash
WEIGHTS="$PWD/Weights/vit_restore_dtw_s16/model_best.pth" \
REPRESENTATION=primary \
bash scripts/eval_yelda_restoration_positive_dtw.sh
```

Representations:

- `primary`: semantic letter-aligned token `L_i` (main result);
- `local`: primitive stroke token `P_i`;
- `joint`: equal-weight combination for diagnosis.

Run all three:

```bash
WEIGHTS="$PWD/Weights/vit_restore_dtw_s16/model_best.pth" REPRESENTATION=local \
bash scripts/eval_yelda_restoration_positive_dtw.sh

WEIGHTS="$PWD/Weights/vit_restore_dtw_s16/model_best.pth" REPRESENTATION=primary \
bash scripts/eval_yelda_restoration_positive_dtw.sh

WEIGHTS="$PWD/Weights/vit_restore_dtw_s16/model_best.pth" REPRESENTATION=joint \
bash scripts/eval_yelda_restoration_positive_dtw.sh
```

The desired pattern is that primitive features retain stroke detail while the
semantic representation gives the best cross-manuscript alignment.

## First ablations after the baseline

Do not add extra losses until the baseline tells us why they are needed.

1. `RESTORATION_WEIGHT=0`: does DTW alone forget stroke information?
2. `POSITIVE_LETTER_DTW_WEIGHT=0`: does restoration alone fail semantic alignment?
3. stride 8 vs stride 16.
4. optional multi-font contextual-form glyph warm-up only if DTW cannot bootstrap.

A later diagnostic should visualize the window-by-letter cosine matrix together
with the positive DTW path and reconstructed stroke windows.
