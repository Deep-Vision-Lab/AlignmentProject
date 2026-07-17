# Multi-GPU DDP training commands

This guide contains the terminal commands for the two-GPU full-quality Arabic alignment training on branch `multi_gpu_ddp`.

The default configuration uses:

- two RTX 4090 GPUs;
- `BATCH_SIZE=8` per GPU;
- global batch size `8 × 2 = 16`;
- full image-text loss on both lines;
- four active transcript negatives;
- local hard-negative loss;
- compositional image-image loss;
- differentiable contextual order loss;
- real-data binarization;
- real-data augmentation and smart RTL line stitching.

## 1. Enter the repository

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject_clone
```

## 2. Download and switch to the multi-GPU branch

```bash
git fetch origin
git checkout multi_gpu_ddp
git pull origin multi_gpu_ddp
```

Confirm the current branch:

```bash
git branch --show-current
```

Expected output:

```text
multi_gpu_ddp
```

## 3. Activate the Conda environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manucripts_align
```

Check the important packages:

```bash
python - <<'PY'
import torch
import jax
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible Torch GPUs:", torch.cuda.device_count())
print("jax:", jax.__version__)
print("transformers:", transformers.__version__)
PY
```

## 4. Check that the real dataset exists

```bash
ls -lh DataSet/ArabicDataset/dataset_manifest.jsonl
```

Check the dataset directories:

```bash
find DataSet/ArabicDataset -maxdepth 2 -type d | head -30
```

## 5. Optional DataLoader test

Run this before submitting the long job:

```bash
DATASET_TYPE=real \
DATA_DIR=DataSet/ArabicDataset \
REAL_VALIDATE_PATHS=1 \
REAL_BINARIZE=1 \
REAL_AUGMENT=1 \
BATCH_SIZE=2 \
python - <<'PY'
from AugmentedRealDataLoader import build_dataloaders

train_loader, valid_loader, test_loader = build_dataloaders(
    "DataSet/ArabicDataset"
)

batch = next(iter(train_loader))

print("images1:", tuple(batch["images1"].shape))
print("images2:", tuple(batch["images2"].shape))
print("text1:", batch["texts1"][0])
print("text2:", batch["texts2"][0])
print("negatives1:", batch["neg_texts1"][0])
print("train batches:", len(train_loader))
print("valid batches:", len(valid_loader))
print("test batches:", len(test_loader))
PY
```

Expected image shape for this test:

```text
(2, 3, 128, 1024)
```

## 6. Two-GPU Torch, NCCL, JAX, and DLPack preflight

This command must run inside an allocation containing two GPUs:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train/check_ddp_jax_devices.py
```

Expected final message:

```text
DDP/JAX preflight passed.
```

An optional short interactive Slurm allocation is:

```bash
salloc \
  --partition=rtx4090 \
  --gpus=rtx_4090:2 \
  --cpus-per-task=16 \
  --mem=96G \
  --time=00:30:00
```

After the allocation starts, activate the environment again and run the preflight:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manucripts_align

cd /home/ahmedmas/BGU-Lab/AlignmentProject_clone

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/train/check_ddp_jax_devices.py
```

Leave the interactive allocation when finished:

```bash
exit
```

## 7. Start the default two-GPU training

```bash
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

The default job name is:

```text
align_ddp2
```

The default model output directory is:

```text
Weights/real_arabic_fullquality_span3_augmented_ddp2/
```

## 8. Start with a custom job name

```bash
JOB_ID=real_arabic_ddp_experiment_01 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 9. Keep the global batch size equal to 16

The batch size is per GPU. For two GPUs, use:

```bash
BATCH_SIZE=8 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

This gives:

```text
8 samples per GPU × 2 GPUs = global batch size 16
```

## 10. Reduce memory usage

Use four samples per GPU:

```bash
BATCH_SIZE=4 \
LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH=4 \
IMAGE_PAIR_MAX_SAMPLES_PER_BATCH=4 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

This gives a global batch size of eight.

## 11. Train for a custom number of epochs

```bash
EPOCHS=220 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 12. Initialize from model weights

This loads the model and text encoder weights but starts a new optimizer and scheduler:

```bash
PRETRAINED_WEIGHTS="Weights/OLD_JOB/model_latest.pth" \
JOB_ID=real_arabic_ddp_from_pretrained \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 13. Resume a complete checkpoint

This restores:

- epoch number;
- image model;
- text encoder projection;
- optimizer;
- scheduler;
- AMP scaler.

```bash
RESUME="Weights/OLD_JOB/checkpoint_latest.pth" \
JOB_ID=real_arabic_ddp_resumed \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

Do not set `PRETRAINED_WEIGHTS` and `RESUME` together.

## 14. Train only with high-match real pairs

```bash
REAL_DATASET_LABELS=high_match \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

The default uses:

```text
high_match,medium_match
```

## 15. Disable smart line stitching

Keep the other augmentations but disable connecting two Arabic lines:

```bash
REAL_AUG_STITCH_PROB=0 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 16. Disable every real-data augmentation

Binarization remains enabled, but training augmentation is disabled:

```bash
REAL_AUGMENT=0 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 17. Disable image binarization

```bash
REAL_BINARIZE=0 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

The recommended default is Otsu binarization:

```bash
REAL_BINARIZE=1 \
REAL_BINARIZE_METHOD=otsu \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

Use a fixed threshold instead:

```bash
REAL_BINARIZE=1 \
REAL_BINARIZE_METHOD=fixed \
REAL_BINARIZE_THRESHOLD=180 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 18. Disable W&B logging

```bash
USE_WANDB=0 \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

Use offline W&B mode:

```bash
WANDB_MODE=offline \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

Use a custom W&B project:

```bash
WANDB_PROJECT=alignment-project-ddp \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 19. Recommended full launch with an explicit job ID

```bash
JOB_ID=real_arabic_fullquality_span3_augmented_ddp2 \
BATCH_SIZE=8 \
EPOCHS=180 \
REAL_DATASET_LABELS=high_match,medium_match \
REAL_BINARIZE=1 \
REAL_BINARIZE_METHOD=otsu \
REAL_AUGMENT=1 \
REAL_AUG_STITCH_PROB=0.25 \
USE_WANDB=1 \
WANDB_MODE=online \
bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh
```

## 20. Show your queued and running jobs

```bash
squeue -u "$USER"
```

Refresh every two seconds:

```bash
watch -n 2 'squeue -u "$USER"'
```

Show detailed information for one job:

```bash
scontrol show job JOB_ID_NUMBER
```

Replace `JOB_ID_NUMBER` with the numeric Slurm job ID.

## 21. Follow the training log

The output pattern is:

```text
out/align_ddp2_JOBID.out
```

List recent output files:

```bash
ls -lt out/align_ddp2_*.out | head
```

Follow the newest DDP log:

```bash
tail -f "$(ls -t out/align_ddp2_*.out | head -1)"
```

Follow a specific job log:

```bash
tail -f out/align_ddp2_JOB_ID_NUMBER.out
```

Search for batch timing:

```bash
grep -E 'batch=|forward=|backward=|time=' out/align_ddp2_JOB_ID_NUMBER.out
```

Show only the latest timing lines:

```bash
grep -E 'batch=|forward=|backward=|time=' \
  out/align_ddp2_JOB_ID_NUMBER.out | tail -20
```

Search for errors:

```bash
grep -iE 'error|exception|traceback|oom|out of memory|nccl|xla' \
  out/align_ddp2_JOB_ID_NUMBER.out
```

## 22. Inspect GPU use for a running job

First find the node:

```bash
squeue -j JOB_ID_NUMBER -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'
```

On clusters that permit overlapping job steps, run:

```bash
srun \
  --jobid=JOB_ID_NUMBER \
  --overlap \
  nvidia-smi
```

Refresh GPU use every two seconds:

```bash
srun \
  --jobid=JOB_ID_NUMBER \
  --overlap \
  watch -n 2 nvidia-smi
```

Exit `watch` with `Ctrl+C`.

## 23. Check Slurm resource statistics

```bash
sstat \
  -j JOB_ID_NUMBER.batch \
  --format=JobID,AveCPU,AveRSS,MaxRSS,Elapsed
```

After the job finishes:

```bash
sacct \
  -j JOB_ID_NUMBER \
  --format=JobID,JobName,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

## 24. Cancel a job

```bash
scancel JOB_ID_NUMBER
```

Cancel every job owned by the current user only when that is intended:

```bash
scancel -u "$USER"
```

## 25. Inspect generated checkpoints

```bash
ls -lh Weights/real_arabic_fullquality_span3_augmented_ddp2/
```

Expected files include:

```text
model_latest.pth
checkpoint_latest.pth
```

Inspect their sizes:

```bash
du -h Weights/real_arabic_fullquality_span3_augmented_ddp2/*
```

## 26. Confirm that both GPUs were configured

Search the log for the DDP startup lines:

```bash
grep -E 'world_size=|per_gpu_batch=|global_batch=|device_per_rank=' \
  out/align_ddp2_JOB_ID_NUMBER.out
```

Expected values:

```text
world_size=2
per_gpu_batch=8
global_batch=16
```

## 27. Compare this branch with the augmentation branch

```bash
git diff --stat origin/real_data_augmentation..origin/multi_gpu_ddp
```

List the DDP-only commits:

```bash
git log \
  --oneline \
  origin/real_data_augmentation..origin/multi_gpu_ddp
```

## 28. Return to the single-GPU augmentation branch

```bash
git checkout real_data_augmentation
git pull origin real_data_augmentation
```

Return to the multi-GPU branch:

```bash
git checkout multi_gpu_ddp
git pull origin multi_gpu_ddp
```

## 29. Fast copy-paste route

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject_clone

git fetch origin
git checkout multi_gpu_ddp
git pull origin multi_gpu_ddp

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manucripts_align

bash scripts/train/run_span_d3tw_full_quality_real_augmented_ddp.sh

squeue -u "$USER"
```

Then follow the newest log:

```bash
tail -f "$(ls -t out/align_ddp2_*.out | head -1)"
```
