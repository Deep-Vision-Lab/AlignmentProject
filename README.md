# AlignmentProject

Arabic manuscript image–text and image–image alignment experiments.

## Connected-subword experiment

Two isolated experiment branches implement Arabic connected-component text
units without modifying the established model branches:

- `agent/connected-subword-cnn` — CNN+BiLSTM visual encoder.
- `agent/connected-subword-vit` — ViT visual encoder.

The text line is converted into:

```text
connected run → <SUBWORD_BOUNDARY> → connected run → <SPACE> → next word
```

Each connected run can consume several consecutive visual windows. The explicit
subword boundary consumes one or two transition windows, `<SPACE>` separates complete
words, and the ordinary free `<BLANK>` transition remains available for page
background or unused windows.

Example:

```text
الرحمن الرحيم
↓
ا, <SUBWORD_BOUNDARY>, لر, <SUBWORD_BOUNDARY>, حمن,
<SPACE>, ا, <SUBWORD_BOUNDARY>, لر, <SUBWORD_BOUNDARY>, حيم
```

The tokenizer follows Unicode Arabic joining behavior; it does not split by a
fixed character count.

### Train

Run from the repository root on the login node. The script submits its own Slurm
job, so do not prefix it with `sbatch`.

```bash
JOB_ID=cnn_connected_subword_real \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_bilstm_real_aug/model_best.pth" \
bash scripts/train/run_connected_subword_finetune.sh
```

On the ViT branch, use a ViT checkpoint:

```bash
JOB_ID=vit_connected_subword_real \
PRETRAINED_WEIGHTS="$PWD/Weights/vit_real_aug/model_best.pth" \
bash scripts/train/run_connected_subword_finetune.sh
```

### Evaluate

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate_connected_subword.sh
```

Transcript-supervised pair, retrieval, and word-correspondence evaluation:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate_transcript_connected_subword.sh
```

## Baseline branches

The original approaches remain unchanged:

- `agent/training-speed-optimization` — CNN+BiLSTM character-span baseline.
- `agent/use-vit-encoder` — ViT character-span baseline.
