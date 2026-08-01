# AlignmentProject

Arabic manuscript image–text and image–image alignment experiments.

## Connected-subword experiment

The experiment is isolated from the established baselines:

- `agent/connected-subword-cnn` — CNN+BiLSTM visual encoder.
- `agent/connected-subword-vit` — ViT visual encoder.

The Arabic transcript is represented as:

```text
connected run → <SUBWORD_BOUNDARY> → connected run → <SPACE> → next word
```

Each connected run may consume several consecutive image windows. The explicit
`<SUBWORD_BOUNDARY>` marks disconnected runs inside one word, `<SPACE>` marks a
new word, and the ordinary free `<BLANK>` transition remains available for
background or unused windows.

## Required two-stage training

Do not initialize this experiment from an existing real-data checkpoint. Train a
fresh connected-subword model on synthetic Arabic first, then initialize the
real-data stage from that new synthetic checkpoint.

Both launchers submit their own Slurm jobs. Run them from the repository root on
the login node and do not prefix them with `sbatch`.

### Stage 1 — synthetic training from scratch

```bash
JOB_ID=cnn_connected_subword_synthetic \
bash scripts/train/run_connected_subword_synthetic.sh
```

The default synthetic configuration matches the previous synthetic-to-real
workflow: 8,000 samples, 35 epochs, two GPUs, effective global batch 128, and the
zero-shot synthetic profile. Stage 1 rejects `PRETRAINED_WEIGHTS` and
`SYNTHETIC_WEIGHTS`, so the visual alignment model cannot accidentally load an
old checkpoint.

On the ViT branch use a ViT-specific job name:

```bash
JOB_ID=vit_connected_subword_synthetic \
bash scripts/train/run_connected_subword_synthetic.sh
```

### Stage 2 — real-data continuation

After Stage 1 finishes, use only its newly created checkpoint:

```bash
JOB_ID=cnn_connected_subword_real \
SYNTHETIC_WEIGHTS="$PWD/Weights/cnn_connected_subword_synthetic/model_latest.pth" \
bash scripts/train/run_connected_subword_real.sh
```

For ViT:

```bash
JOB_ID=vit_connected_subword_real \
SYNTHETIC_WEIGHTS="$PWD/Weights/vit_connected_subword_synthetic/model_latest.pth" \
bash scripts/train/run_connected_subword_real.sh
```

The default real configuration remains 30 epochs with 6,000 augmented real
samples per epoch.

## Evaluation

Qualitative/Smith–Waterman evaluation:

```bash
WEIGHTS="$PWD/Weights/<real_job_id>/model_latest.pth" \
bash Evaluation/evaluate_connected_subword.sh
```

Transcript-supervised pair, retrieval, and word-correspondence evaluation:

```bash
WEIGHTS="$PWD/Weights/<real_job_id>/model_latest.pth" \
bash Evaluation/evaluate_transcript_connected_subword.sh
```

## Baseline branches

The original approaches remain unchanged:

- `agent/training-speed-optimization` — CNN+BiLSTM character-span baseline.
- `agent/use-vit-encoder` — ViT character-span baseline.
