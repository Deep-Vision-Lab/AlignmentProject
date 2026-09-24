# Restoration + Fused DTW Window Encoder

Branch: `agent/restoration-positive-dtw-window-encoder`

## Active recommendation pipeline

The active path is now:

```
original line image
  -> temporary foreground mask ONLY to find outer crop
  -> crop original RGB with safety margin
  -> proportional resize to the fixed 128x1024 canvas
  -> keep internal spaces and record crop/scale/offset metadata
  -> overlapping 128x32 windows, stride 16
  -> shared local CNN
  -> local vector L_i
       |\
       | +--> RGB decoder --> complete original-window reconstruction
       |
       +--> position + Transformer --> contextual vector C_i
                                      |
                         concat(L_i,C_i)
                                      |
                           trainable projection
                                      |
                              L2 normalize
                                      |
                         fused alignment vector Z_i
                                      |
                   positive + negative letter DTW
```

The text character codebook is frozen and used only for training supervision.
Evaluation uses two manuscript images only: both lines pass through the same
local encoder, Transformer and fusion head, then their fused vectors are aligned.

## Stage A — local RGB restoration

Stage A trains the local CNN and decoder only. Transformer/fusion parameters are
present in the checkpoint but frozen.

```
L_A = L_RGB_reconstruction
```

The reconstruction target is the exact, complete original RGB window after line
crop/resize and ImageNet de-normalization. It is not a binarized stroke map.
Artificial outer padding is excluded from reconstruction loss. The decoder sees
only the local vector; there is no image skip connection and no contextual vector
available to it.

Run without arguments:

The former Stage A launcher has been retired.

Default output:

```
Weights/restore_rgb_pretrain_s16/
```

## Stage B — contextual fused DTW

Stage B initializes the exact Stage-A architecture/checkpoint, unfreezes the
Transformer and fusion head, and keeps the RGB reconstruction term as an
anti-forgetting regularizer.

```
L_B =
    1.0 * L_positive_letter_DTW
  + 0.5 * L_negative_margin_DTW
  + 0.1 * L_RGB_reconstruction
```

Ten negative transcripts are generated per positive sample by default. The
positive transcript must obtain a lower DTW cost than negative transcripts by
the configured margin.

Run without arguments:

The former Stage B launcher has been retired.

It expects:

```
Weights/restore_rgb_pretrain_s16/model_best.pth
```

and writes the alignment run under:

```
Weights/restore_fused_rgb_dtw_s16/
```

## Geometry and padding

Only surrounding blank margin is removed. Internal blank spaces remain part of
the line sequence because they carry positional/alignment information. The
foreground mask is temporary: it chooses the crop box but is never fed to the
model.

The preprocessor records source width/height, crop offsets, crop size, resize
scale, resized size, canvas offsets and canvas size. Evaluation uses these values
to map aligned intervals back to source-image coordinates.

For model padding, `token_valid` masks fully artificial sequence positions and
`restoration_valid_mask` masks padded pixels in the RGB reconstruction loss.

## Local encoder detail preservation

The local CNN no longer collapses width from 32 to 2 immediately. Its spatial
path is:

```
128x32
 -> 64x32
 -> 32x32
 -> 16x16
 -> 8x8
 -> spatial flatten/projection
 -> local vector
```

The optional `spatial_vectors(..., vectors_per_window=K)` API exposes multiple
vectors per window for the recommendation-11 ablation. The default model still
uses one local vector per window.

## Context and fusion

The default Stage-B context network is a two-layer Transformer. Local vectors
receive positional embeddings before the Transformer. The DTW/evaluation vector
is not the local vector and not the contextual vector by itself:

```
Z_i = normalize(Projection([L_i ; C_i]))
```

This exact fused representation is also the primary representation during
image-only evaluation.

## DTW transitions

Vertical transitions allow multiple image windows to map to one letter.
Horizontal transitions allow a single image window to advance across multiple
letters when that is mathematically required (`letters > windows`).

When `windows >= letters`, horizontal moves are disabled by default because
diagonal + vertical moves can cover every letter without allowing one window to
absorb an unnecessary run of characters.

The DTW cost/path is recomputed from the current model features every forward
pass. Nothing forces the route to change from epoch to epoch. The epoch probe
instead records whether the cost matrix, encoder parameters and diagnostic route
actually changed.

## Pretrained restoration comparison (Restormer)

Restormer is an explicit comparison/probe, not silently assumed to be better.
The official third-party model is kept outside this repository. Place:

```
third_party/Restormer/
Weights/Pretrained/Restormer/<checkpoint>.pth
```

then run:

The former Restormer probe launcher has been retired.

The probe loads the pretrained Restormer, captures its latent encoder feature,
checks that two manuscript-like windows produce distinct latent features, and
performs one identity-restoration fine-tuning step. If the external assets are
not present, it prints an explicit SKIP instead of pretending the experiment was
performed.

## Small overfit gate

Before a full run:

The former small-overfit launcher has been retired.

This trains the local encoder/decoder on eight distinct windows, checks that
reconstruction loss falls, checks that decoded windows do not collapse to one
template, and verifies that swapping encoded features changes the outputs.

## Image-only evaluation

After Stage B:

The former image-only evaluation launcher has been retired.

The evaluation launcher defaults to the same training crop/resize geometry and
to `REPRESENTATION=primary`, which is the fused image vector. No text encoder is
loaded for the alignment itself.

## One-command recommendation checks

Run all thirteen recommendation diagnostics:

The former all-points and per-point launchers have been retired. The archived
diagnostic descriptions remain in `RESTORATION_RECOMMENDATION_CHECKLIST.md`.
