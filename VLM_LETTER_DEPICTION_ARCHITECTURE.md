# Hierarchical VLM Letter-Depiction Experiment

Branch: `agent/vlm-letter-depiction-hierarchy`

## Goal

Train each horizontal visual region to answer **what Arabic letter content is depicted here?** before asking the Transformer to reason about the larger line context.

This is intentionally different from treating a window only as a free vector that receives supervision after contextualization.

## Image path

Each manuscript line is encoded independently. Image 1 and image 2 are **never concatenated** before the visual encoder.

```text
line image [3,128,1024]
        |
        v
full-height 32px patch projection, stride 16
(trainable Conv2d)
        |
        v
primitive visual tokens [~63,128]
        |
        v
trainable depiction head
Linear -> GELU -> Linear + residual + LayerNorm
        |
        v
LOCAL DEPICTION TOKENS
"what letter strokes/content does this region depict?"
        |
        +-------------------------------+
        |                               |
        | letter-level VLM supervision  |
        |                               v
        |                    single-letter AraBERT prototypes
        |                    frozen AraBERT backbone
        |                    + trainable text projection
        |                               |
        |                               v
        |                    monotonic letter soft-DTW
        |                    positive transcript
        |                    vs all 10 negative candidates
        |                    -> hardest negative margin
        |
        v
learned positional embedding
        |
        v
4-layer Transformer
        |
        v
CONTEXTUAL TOKENS
"what does this depicted content mean in the surrounding line?"
        |
        +--> contextual Span-DTW to Arabic short-span text representations
        |
        +--> image-image region contrastive loss
        |
        +--> sequence/order consistency
        |
        +--> variance/local hard-negative regularization
```

## Local letter alignment

The local objective aligns the depiction-token sequence directly against a sequence of **single Arabic letter prototypes**.

The DTW permits three monotonic moves:

- diagonal: next window and next letter;
- vertical: another window depicts the same letter;
- horizontal: the same 32px window also depicts the next adjacent letter.

The horizontal move is important because one overlapping 32px window can visibly contain strokes from more than one neighboring Arabic letter.

Whitespace, Tatweel, and combining marks are not semantic letter states in this local objective. Low-ink windows are removed before local letter DTW, so empty word gaps do not need to impersonate letters.

## Positive/negative supervision

For each image line:

1. compute local letter-DTW cost for the correct transcript;
2. evaluate every supplied negative transcript candidate;
3. find the negative transcript with the lowest local letter-DTW cost (the visually most confusing negative);
4. optimize the positive cost plus a margin between positive and hardest negative.

The branch generates 10 negative transcripts. The existing contextual Span-DTW also sees all generated negatives and uses the hardest negative gradient.

## Contextual level

The Transformer does not contextualize the primitive patch vectors. It contextualizes the **letter-depiction tokens**.

Therefore the hierarchy is:

```text
visual evidence
  -> local depicted-letter concept
  -> contextual short-span/word concept
  -> paired-line semantic alignment
```

This mirrors the compositional VLM idea: identify the parts first, then learn how those parts compose in context.

## Existing losses retained

Nothing is removed from the full-quality training path:

- contextual image-text Span-DTW;
- local hard-negative loss;
- image-text loss on both lines;
- image-image contrastive loss;
- order/sequence consistency loss;
- visual variance regularizer.

The new local letter-depiction loss is added before these contextual/pair objectives.

## Quality-first settings

- dataset cap: 10,000 pairs;
- raw RGB input (no model-side Otsu);
- 4 Transformer layers;
- 32px windows, stride 16;
- 10 negative transcripts generated;
- no rotating/subsampling of contextual negative candidates;
- local hard-negative stage every batch, all samples;
- image-pair stage every batch, all samples;
- frozen AraBERT feature cache disabled in this branch;
- image 1 and image 2 always use separate visual forward calls.

## Training job

```bash
sbatch scripts/train_synthetic63_2x4090.sbatch
```

Expected Slurm job name: `vit_vlm_letters`.
