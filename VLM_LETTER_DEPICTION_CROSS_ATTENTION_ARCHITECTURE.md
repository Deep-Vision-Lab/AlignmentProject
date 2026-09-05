# VLM Letter-Depiction + Bidirectional Cross-Attention

Branch: `agent/vlm-letter-depiction-cross-attention`

Parent ablation: `agent/vlm-letter-depiction-hierarchy`

## Purpose

Test whether pair-aware cross-attention improves image-to-image alignment after
each line has already learned an independent letter-grounded representation.

The branch intentionally does **not** let line 1 see line 2 during local depiction
or self-context learning. Cross-line interaction begins only after both lines have
completed their own 4-layer visual Transformer.

## Architecture

```text
LINE 1                                      LINE 2
  |                                           |
  v                                           v
raw RGB                                    raw RGB
  |                                           |
  v                                           v
32px full-height patch projection          32px full-height patch projection
  |                                           |
  v                                           v
primitive local tokens                     primitive local tokens
  |                                           |
  v                                           v
trainable LETTER DEPICTION HEAD            trainable LETTER DEPICTION HEAD
  |                                           |
  +--> local letter monotonic DTW             +--> local letter monotonic DTW
  |     + positive/10 negatives               |     + positive/10 negatives
  v                                           v
letter-depiction tokens                    letter-depiction tokens
  |                                           |
  v                                           v
4-layer SELF-attention Transformer         4-layer SELF-attention Transformer
  |                                           |
  v                                           v
C1: independent contextual tokens          C2: independent contextual tokens
  |                                           |
  +--> independent image-text Span-DTW       +--> independent image-text Span-DTW
  |                                           |
  +---------------------+---------------------+
                        |
                        v
              BIDIRECTIONAL CROSS-ATTENTION

          Q=C1, K=C2, V=C2      Q=C2, K=C1, V=C1
                 |                      |
                 v                      v
                X1                     X2
                 |                      |
                 v                      v
          F1 = residual fusion   F2 = residual fusion
                 |                      |
                 +----------+-----------+
                            |
                            v
                     cosine(F1, F2)
                            |
                            v
                  pair contrastive/order loss
                            |
                            v
                   final NW/DTW alignment
```

## Cross-attention block

The same cross-attention weights are used in both directions because line 1 and
line 2 are the same modality and should not have privileged roles.

Each direction is:

```text
query contextual tokens
      |
      +--> LayerNorm --> Q

other line contextual tokens
      |
      +--> LayerNorm --> K,V

MultiHeadAttention(Q,K,V), 4 heads
      |
      v
learned gated residual addition
      |
      v
LayerNorm
      |
      v
FFN (128 -> 256 -> 128)
      |
      v
residual + LayerNorm
      |
      v
fused cross-aware representation
```

The cross residual starts with a gate of 0.20 so the model begins mostly from its
independent representation and learns how much pair information is genuinely useful.

Low-ink query windows remain equal to their independent contextual vectors, and
low-ink memory windows are masked from cross-attention. This prevents blank page
regions from acquiring semantic content merely by reading the opposite line.

## Why independent image-text grounding is preserved

The text losses are computed on `C1` and `C2` **before** cross-attention:

```text
C1 -> Span-DTW(text1)
C2 -> Span-DTW(text2)
```

This prevents the model from using the other image as a shortcut to satisfy its
text supervision. Each image must remain useful by itself.

## Pair objectives

The original hierarchy's pair loss is retained:

```text
local depiction regions line1 <-> local depiction regions line2
```

A second pair objective is added on the cross-aware features:

```text
cross-aware contextual regions F1 <-> F2
      -> cosine similarity
      -> same-text positives
      -> hard different-text negatives
```

Cross-aware order consistency is also added.

Therefore this branch adds cross-attention rather than replacing the parent losses.

## DDP / image separation

Image 1 and image 2 are still encoded in two separate visual forwards:

```text
images1 -> ViT -> C1
images2 -> ViT -> C2
```

There is no `torch.cat([images1, images2])` image batch.

The pair cross-attention module is registered with the trainable text-side runtime
owner only so its gradients use the existing explicit distributed all-reduce path.
This is an implementation detail; it does not participate in text encoding.

## Quality-first settings

Inherited from the letter-depiction hierarchy:

- 10,000-pair dataset cap;
- raw RGB model input;
- 4 self-attention Transformer layers;
- letter-level local depiction loss;
- 10 negative transcripts considered for hardest-negative selection;
- image-text loss on both lines;
- local hard-negative stage every batch, all samples;
- original image-image pair stage every batch, all samples;
- frozen AraBERT surface cache disabled for this experiment.

Added here:

- 4-head bidirectional shared cross-attention;
- gated residual fusion;
- cross-aware cosine pair contrastive weight: 0.30;
- cross-aware order weight: 0.05.

## Training

```bash
sbatch scripts/train_synthetic63_2x4090.sbatch
```

Slurm job name: `vit_vlm_cross`.

## Comparable evaluation

Use the branch-specific wrapper so the standard NW diagnostic evaluates the
**post-cross-attention cosine matrix**:

```bash
python Evaluation/eval_img_align_nw_cross_attention.py \
  --dataset DataSet/Synthetic63 \
  --weights Weights/vit_vlm_cross/model_best.pth \
  --feature contextual \
  --n-samples 100
```

This produces the same NW/mask metrics and visual evidence as the normal diagnostic,
so the parent hierarchy and cross-attention branch can be compared directly.
