# Direct connected-subword supervision

The optional direct synthetic mode removes Span-DTW from synthetic training by
using renderer-derived horizontal intervals for every connected Arabic subword.
The original connected-subword Span-DTW command remains unchanged as an
ablation.

For every labeled subword interval, training:

1. maps the interval to overlapping image windows;
2. pools contextual CNN+BiLSTM features inside the interval;
3. matches the visual vector to the AraBERT subword vector with symmetric
   multi-positive InfoNCE;
4. applies interval-localization cross-entropy over all windows; and
5. penalizes highly similar windows outside every occurrence of that subword.

Duplicate subword strings are positives rather than false negatives. Interval
weights are reversed automatically for the logical right-to-left Arabic window
sequence.

## Train from scratch

Run from the repository root on the login node. The wrapper first builds
`DataSet/Synthetic_Arabic/subword_boxes/`, then delegates to the existing
multi-GPU Slurm launcher:

```bash
JOB_ID=cnn_connected_subword_direct_synthetic \
DATA_DIR="$PWD/DataSet/Synthetic_Arabic" \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=128 \
EPOCHS=35 \
bash scripts/train/run_connected_subword_direct_synthetic.sh
```

The checkpoint records:

```text
direct_subword_supervision=true
synthetic_alignment_backend=renderer_subword_intervals_no_dtw
```

The default loss weights are:

```text
DIRECT_SUBWORD_REGION_WEIGHT=1.0
DIRECT_SUBWORD_LOCALIZATION_WEIGHT=1.0
DIRECT_SUBWORD_OUTSIDE_WEIGHT=0.25
DIRECT_SUBWORD_TEMPERATURE=0.07
DIRECT_SUBWORD_OUTSIDE_MARGIN=0.25
DIRECT_SUBWORD_OUTSIDE_TOP_K=8
```

## Rebuild sidecars manually

```bash
python scripts/data/build_connected_subword_boxes.py \
  --data-dir "$PWD/DataSet/Synthetic_Arabic" \
  --font "$PWD/Fonts/Arslan_Wessam_B.ttf" \
  --font-size 90 \
  --padding 20 \
  --canvas-width 1024 \
  --canvas-height 128 \
  --overwrite
```

The builder reproduces the renderer geometry, estimates intervals with reshaped
RTL prefix widths, and snaps boundaries to nearby low-ink columns.

## Real-data continuation

Real manuscript records currently provide line transcripts but not trustworthy
connected-subword intervals. Real continuation therefore still uses the existing
Span-DTW objective:

```bash
JOB_ID=cnn_connected_subword_real \
SYNTHETIC_WEIGHTS="$PWD/Weights/cnn_connected_subword_direct_synthetic/model_latest.pth" \
bash scripts/train/run_connected_subword_real.sh
```

Removing Span-DTW from real continuation requires visual proposals,
connected-component grouping, or similarity-based CTC.
