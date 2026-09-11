# CFM-Inspired Spatial Language Alignment

Branch: agent/cfm-inspired-spatial-language-alignment

Base: agent/vlm-letter-depiction-cross-attention at 15eb68a.

## Goal

The branch borrows the most useful principle from CFM without copying CFM
literally: preserve a semantic representation for every physical image location,
align that local representation to language, and only then allow context to
refine it. For Arabic handwriting, a "location" is an overlapping full-height
horizontal window rather than a square image patch.

The core invariant is:

    window i -> L_i -> F_i

Window i is never pooled into a line-level vector. L_i remains the local
language-grounded token for that x-position, and F_i is a context-refined
version of the same position.

## Architecture

    original full-width RGB line
              |
              v
    32px full-height windows, overlapping
              |
              v
    primitive visual token P_i
              |
              v
    bounded semantic-affinity voting
              |
              v
    image-side language adapter
              |
              v
    LOCAL LANGUAGE TOKEN L_i
       |              |
       |              +--> local letter Soft-DTW
       |              +--> dense Arabic-letter InfoNCE
       |
       v
    + positional embedding
       |
       v
    4-layer self-attention Transformer
       |
       v
    contextual proposal C_i
       |
       v
    F_i = LN(L_i + g_ctx * (C_i - L_i))
       |
       +--> local/context consistency
       +--> contextual Span-DTW vs transcript spans
       +--> independent image-image contrastive/order loss
       |
       v
    cosine(F1, F2) -> NW / downstream image-image alignment

There is NO cross-line attention in the recommended main model. Each line must
be meaningful by itself.

## Stage 1: original RGB and physical windows

The model keeps the full-width line and disables model-side Otsu binarization.
The first experiment uses window size 32 and stride 16 to keep the architecture
comparison fair. A second experiment should use stride 8.

Each token has an exact physical window index, so later letter evidence can be
mapped back to the line.

## Stage 2: bounded semantic-affinity voting

For primitive tokens P_i and P_j:

    a_ij = cosine(P_i, P_j) / tau - lambda_d * |i-j|

Only |i-j| <= r is allowed. Defaults are r=3, tau=0.15 and distance penalty
0.12. Softmax over j gives voting weights.

The voted feature is added through a learned sigmoid gate initialized at 0.15.
This is the analogue of CFM's semantic region reinforcement: adjacent windows
that look semantically related support each other, but distant repeated Arabic
letters cannot collapse spatial identity.

## Stage 3: image-side language adapter

A residual MLP transforms the voted visual representation into L_i. L_i itself
is compared with frozen text vectors. This is deliberately different from a
temporary classifier head: the representation used later for alignment is the
representation that learns language semantics.

## Stage 4: frozen Arabic language anchor

AraBERT remains frozen and, in this branch, its projection/norm/space/blank
parameters are frozen too. Only the image side moves.

Recommended initialization is a previously learned text projection:

    TEXT_ANCHOR_WEIGHTS="$PWD/Weights/vit_vlm_cross/model_best.pth"

If no anchor is supplied, the projection is created with the same deterministic
seed on every DDP rank and then frozen.

## Loss 1: local monotonic letter Soft-DTW

For the local token sequence L and transcript letters T, the cost matrix is:

    D_ij = 1 - cosine(L_i, T_j)

Soft-DTW allows diagonal, vertical and horizontal transitions. Therefore one
Arabic letter may occupy several overlapping windows and one 32px window may
contain strokes from adjacent letters.

Ten negative transcripts are scored. The closest negative is the hard negative.

    L_local =
        beta * cost_positive
        + ReLU(cost_positive - cost_hard_negative + margin)

Defaults: beta=0.35, margin=0.20, gamma=0.05, step penalty=0.02.
The total-loss weight is 0.40.

## Loss 2: dense letter vocabulary objective

A detached hard monotonic path from the positive local alignment produces
pseudo-labels for physical windows. Each window is then compared with the full
Arabic letter inventory.

    logits(i,k) = cosine(L_i, prototype_k) / 0.07

If a window covers two adjacent letters, both are treated as positives via a
multi-positive InfoNCE objective. This is the most segmentation-like component:
it asks every physical image location which Arabic letter concept it depicts.

Weight: 0.15.

## Stage 5: self-context

L_i plus a positional embedding enters the existing four-layer Transformer.
This creates C_i. The Transformer can use the whole line to disambiguate local
strokes, Arabic connectivity and character shape variants.

## Stage 6: local-preserving context fusion

Context is not allowed to replace the local token:

    F_i = LN(L_i + g_ctx * (C_i - L_i))

The sigmoid gate starts at 0.25 and is trainable. Early training is therefore
mostly local and language grounded; useful context can grow gradually.

## Loss 3: local/context consistency

For ink windows:

    L_cons = mean(1 - cosine(F_i, stopgrad(L_i)))

Weight: 0.05. The stop-gradient means L_i remains the semantic anchor and the
contextual representation is discouraged from forgetting what is physically
present at that x-position.

## Loss 4: contextual Span-DTW

The existing transcript-span contrastive Soft-DTW is retained on F_i. Text
spans are up to three characters. Ten transcript negatives are considered and
the hardest is used for gradient.

This loss answers a different question than local letter DTW: local DTW asks
"which letter strokes are here?", while Span-DTW asks "which short text span
does this context-refined region represent?"

## Loss 5: image-image pair contrastive

The two manuscript lines are encoded independently. Transcript-derived regions
with matching text are pulled together and mismatching regions are pushed apart.
The existing weight remains 0.40.

No image can read information from the other image before producing its own
representation. That is important for image-only evaluation.

## Loss 6: order consistency

The existing soft positional/monotonic pair consistency remains at weight 0.05.
It discourages visually plausible but globally reordered matches.

## Loss 7: variance regularization

The existing 0.01 image-variance regularizer remains to reduce representation
collapse.

## Recommended total objective

    L =
        L_span
        + 0.40 * L_local_letter_DTW
        + 0.15 * L_dense_letter
        + 0.05 * L_local_context_consistency
        + 0.40 * L_image_pair
        + 0.05 * L_order
        + 0.01 * L_variance

The old local-hard-negative heuristic is disabled because the direct local
language losses now provide stronger, semantically meaningful local supervision.

## Recommended experiment order

1. Main architecture, stride 16, frozen text anchor.
2. Same model at stride 8.
3. Dense-letter ablation: DENSE_LETTER_WEIGHT=0.
4. Context-preservation ablation: LOCAL_CONTEXT_CONSISTENCY_WEIGHT=0.
5. Voting ablation: radius 0.
6. Only after these, test whether pair cross-attention gives additional value.

## Training

Main fair-comparison run:

    TEXT_ANCHOR_WEIGHTS="$PWD/Weights/vit_vlm_cross/model_best.pth" \
      sbatch scripts/train_cfm_spatial_2x4090.sbatch

Dense stride-8 run:

    TEXT_ANCHOR_WEIGHTS="$PWD/Weights/vit_vlm_cross/model_best.pth" \
    CFM_STRIDE_RATIO=0.25 \
    JOB_NAME=vit_cfm_spatial_s8 \
      sbatch scripts/train_cfm_spatial_2x4090.sbatch

## Evaluation interpretation

Use the normal independent-line evaluator, not the cross-attention wrapper.

- local = L_i: measures pure local language grounding.
- contextual = F_i: measures context-refined spatial semantics.
- joint = combines both views for diagnostics.

The best outcome is not merely a lower training loss. We want sharper
letter-vocabulary similarity maps, a better diagonal/image-image similarity
structure, improved NW paths and improved real manuscript localization while
retaining robust image-only behavior.
