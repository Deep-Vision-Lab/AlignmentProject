# ViT multi-GPU training

Primary branch: `agent/use-vit-encoder`

The ViT model now uses the same generic full-quality training pipeline as the
optimized CNN branches. The only difference is the visual-model constructor;
dataset loading, Span-D3TW losses, AMP, distributed sampling, NCCL setup,
validation aggregation, checkpointing, resume, and W&B remain in `train.py`.

## Two-GPU synthetic training

```bash
git fetch origin
git switch agent/use-vit-encoder
git pull --ff-only origin agent/use-vit-encoder

DATASET_TYPE=synthetic \
NUM_SAMPLES=8000 \
NUM_GPUS=2 \
BATCH_SIZE=32 \
JOB_ID=synthetic_arabic_vit_fullquality_8000_gpu2 \
bash scripts/train/run_vit_span_d3tw_full_quality.sh
```

The launcher requests two RTX 4090 GPUs by default and starts one torchrun
process per GPU. `BATCH_SIZE` is the per-GPU batch size, so the example above
uses a global batch size of 64.

## Backward-compatible shortcut

```bash
bash run_train_vit.sh \
  synthetic_arabic_vit_fullquality_8000_gpu2 \
  DataSet/Synthetic_Arabic
```

The shortcut accepts:

```text
bash run_train_vit.sh [job_id] [data_dir] [pretrained_weights]
```

## Resume a two-GPU run

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=8000 \
NUM_GPUS=2 \
BATCH_SIZE=32 \
JOB_ID=synthetic_arabic_vit_fullquality_8000_gpu2 \
RESUME="$PWD/Weights/synthetic_arabic_vit_fullquality_8000_gpu2/checkpoint_latest.pth" \
bash scripts/train/run_vit_span_d3tw_full_quality.sh
```

## Real-dataset training

```bash
DATASET_TYPE=real \
DATA_DIR="$PWD/DataSet/ArabicDataset" \
REAL_AUGMENT=1 \
REAL_TRAIN_SAMPLES_PER_EPOCH=10000 \
NUM_GPUS=2 \
BATCH_SIZE=32 \
JOB_ID=real_arabic_vit_fullquality_gpu2 \
bash scripts/train/run_vit_span_d3tw_full_quality.sh
```

## ViT architecture overrides

```bash
VIT_INPUT_HEIGHT=128
VIT_LAYERS=4
VIT_HEADS=4
VIT_MLP_DIM=512
VIT_DROPOUT=0.10
VIT_MAX_TOKENS=256
```

`VIT_HEADS` must divide the configured embedding/vector size.

## Multi-GPU runtime defaults

```bash
NUM_GPUS=2
DDP_STATIC_GRAPH=1
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
TORCH_COMPILE_VISUAL=0
```

`NCCL_P2P_DISABLE=1` is retained because some BGU RTX 4090 GPU pairs fail with
direct CUDA peer access. Shared-memory NCCL communication remains enabled.

## Expected startup output

A correct two-GPU run prints values similar to:

```text
world_size=2
per_gpu_batch=32
global_batch=64
device=cuda:0
```

Each torchrun rank sees one isolated physical GPU as its local `cuda:0` device.
Only rank 0 writes W&B data and checkpoints.

The saved checkpoint records `visual_encoder_type=vit`, all ViT architecture
values, `world_size`, per-GPU/global batch sizes, NCCL/DDP configuration, and
the existing training-profile parameters. This allows the shared evaluator to
reconstruct the correct ViT model instead of loading the CNN/BiLSTM encoder.
