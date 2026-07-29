# Canonical model branches

Only these two branches are active development branches:

- `agent/training-speed-optimization` — CNN + BiLSTM visual encoder
- `agent/use-vit-encoder` — ViT visual encoder

They intentionally share the same project tree, datasets, preprocessing,
Span-D3TW losses, negative sampling, DDP implementation, SLURM resources,
NCCL workarounds, AMP, validation, checkpoint format, evaluation scripts,
zero-shot preprocessing, tests, and runtime optimizations.

The only active branch-specific file is:

```text
model_backend.py
```

That file selects the visual model, applies model-specific preparation, and
adds the model architecture fields to the otherwise shared checkpoint config.

## Shared multi-GPU training command

Run the same command on either branch:

```bash
DATASET_TYPE=synthetic \
NUM_SAMPLES=8000 \
NUM_GPUS=2 \
BATCH_SIZE=32 \
JOB_ID=my_experiment \
bash scripts/train/run_model_full_quality.sh
```

For the real dataset:

```bash
DATASET_TYPE=real \
DATA_DIR="$PWD/DataSet/ArabicDataset" \
REAL_AUGMENT=1 \
REAL_TRAIN_SAMPLES_PER_EPOCH=10000 \
NUM_GPUS=2 \
BATCH_SIZE=32 \
JOB_ID=my_real_experiment \
bash scripts/train/run_model_full_quality.sh
```

`BATCH_SIZE` is per GPU. With two GPUs and `BATCH_SIZE=32`, the global batch is
64. The launcher retains the optimized RTX 4090 defaults:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
DDP_STATIC_GRAPH=1
```

## Shared real-dataset evaluation command

Both branches use the same evaluation entry points under `Evaluation/`,
including:

```bash
bash Evaluation/run_real_dataset_evaluations.sh
```

Checkpoints record `visual_encoder_type`, so shared evaluation reconstructs the
correct visual model.

## Synchronization rule

When shared training, evaluation, preprocessing, loss, optimization, test, or
script code changes, apply the identical change to both canonical branches.
Only model implementation/configuration changes belong exclusively in
`model_backend.py` or the selected encoder implementation.
