# AlignmentProject run commands

All architecture, loss, optimization, augmentation, and evaluation defaults are organized in `Parameters.py`.
The training CLI intentionally accepts only the dataset path and optional pretrained weights.

## 1. Train ViT from scratch

Single GPU:

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
python train.py --dataset "$PWD/DataSet/ArabicDataset"
```

Two GPUs with `torchrun`:

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset "$PWD/DataSet/ArabicDataset"
```

No `--weights` argument means **train from scratch**.
The output directory is:

```text
Weights/<experiment_name>_scratch/
```

## 2. Fine-tune ViT

Single GPU:

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
python train.py \
  --dataset "$PWD/DataSet/ArabicDataset" \
  --weights "$PWD/Weights/vit_synthetic/model_latest.pth"
```

Two GPUs:

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
torchrun --standalone --nproc_per_node=2 train.py \
  --dataset "$PWD/DataSet/ArabicDataset" \
  --weights "$PWD/Weights/vit_synthetic/model_latest.pth"
```

Supplying `--weights` automatically means **fine-tuning**. Fine-tuning uses `finetune_epochs` and `finetune_learning_rate` from `Parameters.py`.
The output directory is:

```text
Weights/<experiment_name>_finetune/
```

## 3. Needleman-Wunsch evaluation on the real dataset

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
python Evaluation/eval_img_align_nw_real.py \
  --weights "$PWD/Weights/vit_baseline_finetune/model_latest.pth" \
  --data-dir "$PWD/DataSet/ArabicDataset" \
  --dataset-type real \
  --batch \
  --n-samples 100 \
  --real-split test \
  --feature contextual \
  --score-mode auto \
  --threshold 0.0 \
  --gap -0.30 \
  --output-dir "$PWD/Results/Evaluation/NW/Real/vit_baseline"
```

## 4. Smith-Waterman evaluation on the real dataset

```bash
conda activate manucripts_align
cd /home/ahmedmas/BGU-Lab/AlignmentProject
python Evaluation/eval_img_align_sw.py \
  --weights "$PWD/Weights/vit_baseline_finetune/model_latest.pth" \
  --data-dir "$PWD/DataSet/ArabicDataset" \
  --dataset-type real \
  --batch \
  --n-samples 100 \
  --real-split test \
  --feature contextual \
  --score-mode auto \
  --threshold 0.45 \
  --gap -0.30 \
  --output-dir "$PWD/Results/Evaluation/SW/Real/vit_baseline"
```

## 5. Compare pre-Transformer and post-Transformer features

For the contextual/post-Transformer representation use:

```text
--feature contextual
```

For the local/pre-Transformer patch representation use:

```text
--feature local
```

Run the same evaluation twice with only `--feature` changed so the two representations are directly comparable.

## 6. Experiment branches

All experimental branches are created from the cleaned `agent/use-vit-encoder` baseline:

```text
agent/use-vit-encoder
├── agent/vit-contextual-hard-negatives       # Plan A
├── agent/vit-monotonic-cross-attention       # Plan B
├── agent/vit-monotonic-optimal-transport     # Plan C
└── agent/vit-synthetic-span-localization     # Plan D (later)
```

Do not branch Plan B from Plan A or Plan C from Plan B. They must remain independent comparisons against the same ViT baseline.
