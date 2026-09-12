# Restoration + Positive Letter-DTW Window Encoder

Branch: `agent/restoration-positive-dtw-window-encoder`

## Research hypothesis

Train every manuscript line independently. A window embedding should retain the
physical stroke information inside the window while also being learnable toward
a stable character identity. Corresponding shapes in different manuscripts are
never explicitly paired during training.

The branch is now trained in two stages rather than asking a randomly
initialized encoder to solve restoration and alignment simultaneously:

```
Stage A: L = 1.0 * L_restoration
Stage B: L = 1.0 * L_positive_letter_DTW + 0.05 * L_restoration
```

No negative transcripts, image-image pair contrastive loss, cross-attention,
context-consistency loss, or Transformer context are active. Stage-B character
discrimination comes from full-alphabet competition inside the DTW cost matrix.

## Architecture

The branch is now trained in two explicit stages.

### Stage A — restoration pretraining

```
full 128x1024 RGB line
        |
        | overlapping windows: 128x32, stride 16
        v
W_0 ... W_62
        |
        | SAME local CNN encoder for every window
        v
P_0 ... P_62                     128-D primitive tokens
        |
        | sequence decoder (shared per token)
        v
R_0 ... R_62                     1x128x32 soft stroke maps
        |
        v
L_rest = L_pixel + 0.5 L_edge + 0.5 L_dice
```

There is no DTW during Stage A. Its only job is to learn a local encoder whose
primitive vectors preserve stroke information. The old single full-height Conv2d
projection is retained only for historical checkpoint compatibility.

The new local encoder is a small CNN hierarchy applied independently to each
128x32 window:

```
3x128x32
 -> Conv blocks
 -> 32 channels
 -> 64 channels /2
 -> 96 channels /2
 -> 128 channels /2
 -> 128 channels /2
 -> adaptive pooling
 -> P_i in R^128
```

The line is therefore a sequence of 63 windows, producing a sequence of 63
primitive vectors and a sequence of 63 reconstructions. There is still no
Transformer/BiLSTM/context in this experiment.

### Stage B — DTW alignment from the pretrained encoder

Stage B initializes the exact same CNN encoder and decoder from the Stage-A
checkpoint. The encoder remains trainable.

```
W_i
 |
 v
pretrained local CNN
 |
 v
P_i
 |\
 | \----> restoration decoder ----> 0.05 * L_rest
 |
 +-------> identity semantic path: L_i = P_i
             |
             v
      full-alphabet character competition
             |
             v
      transcript letter cost matrix
             |
             v
      monotonic soft-DTW curriculum
             |
             v
             L_DTW
```

No trainable fully connected semantic adapter exists in the default new run.

### Why the restoration change was necessary

The old diagnostic checkpoint showed that the failure already existed in the
primitive representation. On diagnostic pair 132:

- side 1 primitive effective rank: ~3.36 / 128;
- side 1 stroke-to-primitive similarity correlation: ~0.014;
- side 1 restoration MAE: ~0.498;
- side 2 primitive effective rank: ~2.91 / 128;
- side 2 stroke-to-primitive correlation: ~0.280;
- side 2 restoration MAE: ~0.522.

Therefore the old residual semantic MLP was not the only issue: the primitive
encoder itself had not learned a reliable stroke geometry.

### Restoration loss

The decoder reconstructs a soft foreground/stroke map, not RGB:

```
L_restoration =
    1.0 * foreground-weighted L1
  + 0.5 * edge loss
  + 0.5 * soft Dice loss
```

Dice prevents a blurry/background-dominated reconstruction from receiving an
artificially acceptable pixel loss.

### DTW cost

For each visual window, Stage B first compares it against the complete frozen
Arabic character inventory. If `z_i` is the normalized primitive and `e_c`
is a frozen character vector:

```
logit(i,c) = cosine(z_i, e_c) / temperature
cost(i,c)  = -log softmax_c(logit(i,c))
```

The transcript selects the relevant character columns from this full-alphabet
cost matrix. This gives letter discrimination without adding negative
transcripts.

### DTW curriculum

The DTW path is recomputed on every forward pass. It was never cached. The old
problem was early path lock-in: gamma was already 0.05 from epoch 1 and both
vertical/horizontal steps cost only 0.02.

The new alignment defaults are:

```
gamma:              0.50 -> 0.05 over 10 epochs
vertical penalty:   0.05
horizontal penalty: 0.30
position prior:     0.15
alphabet temp:      0.10
```

When the number of available image windows is at least the number of transcript
letters (`T >= L`), horizontal transitions are disabled entirely. They are
unnecessary in that case because diagonal + vertical transitions can reach the
endpoint while assigning at least one window to every letter.

This directly prevents the old degeneracy where one window could absorb a long
run of transcript characters. If `L > T`, horizontal transitions remain
available because they are mathematically required, but they keep the larger
penalty.

### Epoch diagnostics

Stage B analyzes the same fixed validation line after every epoch and stores:

```
Weights/<JOB_NAME>/epoch_diagnostics/
  history.csv
  epoch_001/
    window_letter_cosine.csv
    window_letter_training_cost.csv
    window_letter_training_cost_dtw.png
    primitive_window_cosine.png
    metrics.json
  ...
```

The most important fields are:

- `training_cost_mean_abs_delta_prev`: is the actual DTW cost matrix changing?
- `dtw_path_jaccard_prev`: how much did the hard diagnostic route change?
- `hard_dtw_max_letters_same_window`: should be 1 whenever T >= L;
- `mean_dtw_path_cosine` and `mean_top1_margin`: is grounding improving?
- `primitive_effective_rank`: is the encoder collapsing?
- `stroke_primitive_similarity_correlation`: does primitive geometry reflect
  visual stroke geometry?
- `patch_weight_relative_delta_prev`: is the local encoder actually updating?
- `dtw_restoration_grad_cosine`: are DTW and restoration gradients compatible?

A correct DTW route does not need to keep changing forever. The desired pattern
is meaningful change early in training followed by stabilization as the
window-letter costs become more discriminative.

## Training

### Stage A: pretrain restoration

```bash
JOB_NAME=restore_seq2seq_pretrain_s16 \
RESTORATION_TRAINING_STAGE=pretrain \
RESTORATION_LOCAL_ENCODER=cnn_seq2seq \
RESTORATION_SEMANTIC_ADAPTER=identity \
RESTORATION_EPOCHS=10 \
RESTORATION_NUM_SAMPLES=6000 \
RESTORATION_LEARNING_RATE=1e-4 \
sbatch scripts/train_restoration_positive_dtw_2x4090.sbatch
```

Stage A uses:

```
DTW weight         = 0
restoration weight = 1.0
```

### Stage B: initialize from Stage A and train alignment

```bash
JOB_NAME=restore_seq2seq_dtw_s16 \
RESTORATION_TRAINING_STAGE=align \
RESTORATION_LOCAL_ENCODER=cnn_seq2seq \
RESTORATION_SEMANTIC_ADAPTER=identity \
RESTORATION_EPOCHS=20 \
RESTORATION_NUM_SAMPLES=6000 \
RESTORATION_LEARNING_RATE=1e-4 \
RESTORATION_EPOCH_PROBE=1 \
sbatch scripts/train_restoration_positive_dtw_2x4090.sbatch \
  --weights "$PWD/Weights/restore_seq2seq_pretrain_s16/model_latest.pth"
```

Stage B defaults to:

```
DTW weight         = 1.0
restoration weight = 0.05
```

The restoration term in Stage B is an anti-forgetting regularizer, not the main
objective.

## Image-only evaluation

The evaluator never needs the transcript or text codebook.

```bash
WEIGHTS="$PWD/Weights/restore_seq2seq_dtw_s16/model_latest.pth" \
REPRESENTATION=primary \
bash scripts/eval_yelda_restoration_positive_dtw.sh
```

For the default identity semantic path, `L_i = P_i`. Therefore `primary` and
`local` are the same underlying feature in the new experiment, and `joint`
does not add new information. Evaluate `primary` as the canonical output.
The three-way representation comparison remains useful only for historical
residual-MLP checkpoints.

Historical comparison commands:

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
