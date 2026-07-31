# AlignmentProject

Arabic manuscript image–text and image–image alignment experiments.

## Supported commands

There are only two public commands. Run both from the repository root on the login node.

### 1. Fine-tune the active branch model

```bash
JOB_ID=cnn_bilstm_real_aug_6k_stride8_30ep_stable \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_bilstm/model_latest.pth" \
bash scripts/train/run_real_finetune.sh
```

The active branch determines the model:

- `agent/training-speed-optimization` → CNN+BiLSTM
- `agent/use-vit-encoder` → ViT

The launcher submits its own Slurm job. Do not call it with `sbatch`.

### 2. Evaluate a trained checkpoint

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate.sh
```

The evaluator submits one Slurm job, reconstructs the model configuration from the checkpoint, and evaluates both `high_match` and `medium_match` by default.

## Internal files

Everything else under `scripts/train/` and the Python files under `Evaluation/` are implementation modules used by the two commands above. They are not alternative launchers and should not be run directly.

In particular:

- `scripts/train/train_model.py` installs the active branch backend.
- `scripts/train/train_optimized.py` owns the optimized training runtime.
- `scripts/train/run_rank_isolated.sh` isolates one CUDA device per distributed rank.
- `Evaluation/eval_img_align_sw.py` is the internal evaluation engine called by `Evaluation/evaluate.sh`.

## Current real-data defaults

- 6,000 augmented samples per epoch
- 30 epochs
- effective global batch size 64
- window size 32, stride 8, 125 visual windows
- stitching disabled
- infeasible Span-DTW positives filtered
- JAX persistent compilation cache enabled
