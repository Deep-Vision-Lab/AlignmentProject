# Generic full-quality training

The branch now has only two training entry points:

- `scripts/train/run_span_d3tw_full_quality.sh`
- `train.py`

The shell launcher submits itself to Slurm. Inside the allocated node it runs
`train.py` directly for one GPU or through `torchrun` for multiple GPUs.

## Synthetic Arabic, 10,000 samples, two GPUs

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=10000 \
NUM_GPUS=2 \
BATCH_SIZE=8 \
JOB_ID=synthetic_arabic_fullquality_10k_ddp2 \
bash scripts/train/run_span_d3tw_full_quality.sh
```

Default synthetic directory resolution:

1. `DataSet/Synthetic_Arabic_10000`
2. `DataSet/Synthetic_Arabic`

The selected directory must contain at least 10,000 `img1_*.png` files.

## Real Arabic data with augmentation, two GPUs

```bash
DATASET_TYPE=real \
DATA_DIR=DataSet/ArabicDataset \
REAL_AUGMENT=1 \
REAL_TRAIN_SAMPLES_PER_EPOCH=10000 \
NUM_GPUS=2 \
BATCH_SIZE=8 \
JOB_ID=real_arabic_fullquality_augmented_ddp2 \
bash scripts/train/run_span_d3tw_full_quality.sh
```

The real manifest keeps its validation and test splits unchanged. Only the
training split is augmented and optionally repeated to the requested number of
samples per epoch.

## Initialize from previous weights

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=10000 \
NUM_GPUS=2 \
BATCH_SIZE=8 \
PRETRAINED_WEIGHTS=Weights/OLD_JOB/model_latest.pth \
JOB_ID=synthetic_from_pretrained_ddp2 \
EPOCHS=30 \
bash scripts/train/run_span_d3tw_full_quality.sh
```

## Resume a complete checkpoint

```bash
DATASET_TYPE=synthetic \
NUM_GPUS=2 \
RESUME=Weights/OLD_JOB/checkpoint_latest.pth \
JOB_ID=synthetic_resumed_ddp2 \
bash scripts/train/run_span_d3tw_full_quality.sh
```

Do not set `PRETRAINED_WEIGHTS` and `RESUME` together.

## One GPU

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=10000 \
NUM_GPUS=1 \
BATCH_SIZE=16 \
bash scripts/train/run_span_d3tw_full_quality.sh
```

## Four GPUs with the same global batch size of 16

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=10000 \
NUM_GPUS=4 \
BATCH_SIZE=4 \
CPUS_PER_TASK=32 \
MEMORY=128G \
bash scripts/train/run_span_d3tw_full_quality.sh
```

## Monitor the job

```bash
squeue -u "$USER"
tail -f "$(ls -t out/align_full_*.out | head -1)"
```

The log should report `world_size`, `per_gpu_batch`, `global_batch`, the selected
dataset type, augmentation state, and the number of batches assigned to each
rank.
