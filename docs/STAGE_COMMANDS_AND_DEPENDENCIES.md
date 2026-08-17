# AlignmentProject — Revised Stage Commands and Dependencies

This is the operational companion to `docs/EXPERIMENT_MASTER_PLAN.md`.

The old standalone canonical-real fine-tuning/evaluation stages have been removed. The new order is:

`build/freeze Bridge V2 -> synthetic train -> zero-shot evaluations -> Bridge pre-eval -> Bridge fine-tune -> post-evaluations -> final all-real`

The same commands are used on CNN+BiLSTM, ViT, and DINOv3 branches; `model_backend.py` selects the architecture.

---

# Phase A — Create the dataset before any model run

## A1. Pull one canonical branch

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git fetch origin
git switch agent/training-speed-optimization
git pull --ff-only origin agent/training-speed-optimization
```

Create Bridge V2 only once. The resulting frozen dataset is shared by all three architecture branches.

## A2. Submit the CPU dataset build

```bash
bash scripts/slurm/submit_bridge_v2_dataset.sh
```

To force a complete rebuild:

```bash
REBUILD_BRIDGE=1 bash scripts/slurm/submit_bridge_v2_dataset.sh
```

The default target is:

```text
DataSet/RealSyntheticBridge_v2
```

The default build uses:
- seed 42;
- all leakage-safe train anchors;
- four guaranteed negatives per anchor;
- negative normalized 3-gram exclusion;
- 1-3 positive shared islands plus distractors;
- stored positive alignment masks.

Monitor the dataset job:

```bash
squeue -u "$USER" -o "%.18i %.38j %.2t %.10M %.55R"
```

After it completes successfully, run the validator explicitly:

```bash
python scripts/data/smoke_test_real_synthetic_bridge.py \
  --data-dir DataSet/RealSyntheticBridge_v2
```

Do **not** submit any model job until this command passes.

---

# Phase B — Full model pipeline

Switch to the architecture branch you want to train and pull it.

CNN / CNN+BiLSTM:

```bash
git switch agent/training-speed-optimization
git pull --ff-only origin agent/training-speed-optimization
```

ViT:

```bash
git switch agent/use-vit-encoder
git pull --ff-only origin agent/use-vit-encoder
```

DINOv3 ConvNeXt:

```bash
git switch agent/use-dinov3-convnext
git pull --ff-only origin agent/use-dinov3-convnext
export DINOV3_REPO_DIR=/path/to/local/dinov3
export DINOV3_WEIGHTS=/path/to/authorized/dinov3_convnext_tiny_weights
```

## Recommended one-command model submission

```bash
RUN_PREFIX=my_main_run \
SYNTH_EPOCHS=20 \
BRIDGE_EPOCHS=15 \
BRIDGE_LR=1e-6 \
FINAL_THRESHOLD=0.50 \
bash scripts/slurm/submit_full_research_pipeline.sh
```

The submitter first validates that the frozen Bridge V2 dataset already exists. It does **not** create the dataset for you. If the dataset is missing or invalid, it exits before submitting GPU jobs.

Model dependency chain:

```text
S1 synthetic training
  -> S2 synthetic qualitative real evaluation
  -> S3 synthetic quantitative real evaluation
  -> S4 Bridge V2 pre-finetune evaluation
  -> S5 Bridge V2 fine-tuning (15 epoch max, best validation checkpoint)
  -> S6 post-Bridge qualitative real evaluation
  -> S7 post-Bridge quantitative real evaluation
  -> S7b post-Bridge Bridge-specific evaluation
  -> S8 final complete-real evaluation
```

Every model stage uses `afterok`. If one stage fails, downstream stages remain blocked.

The job ledger is written to:

```text
logs/research_pipeline_<RUN_PREFIX>.jobs
```

---

# Individual stage commands

## S1 — Synthetic pretraining

```bash
JOB_ID=myrun_synth \
EPOCHS=20 \
LEARNING_RATE=1e-4 \
NUM_NEGATIVES=10 \
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
bash scripts/train/run_branch_fixed63_synthetic.sh
```

Expected checkpoint:

```text
Weights/myrun_synth/checkpoint_latest.pth
```

## S2 — Qualitative zero-shot real evaluation

```bash
WEIGHTS=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
RUN_TAG=myrun_s2_synth_qualitative \
bash scripts/eval/run_stage_qualitative.sh
```

## S3 — Quantitative zero-shot real evaluation

```bash
CHECKPOINT=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
RUN_TAG=myrun_s3_synth_quantitative \
bash scripts/eval/run_stage_quantitative.sh
```

## S4 — Evaluate Bridge V2 before fine-tuning

```bash
CHECKPOINT=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
BRIDGE_DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
RUN_TAG=myrun_s4_bridge_pretrain \
bash scripts/eval/run_stage_bridge_eval.sh
```

This measures the augmentation task before the model learns from it.

## S5 — Direct fine-tuning on Bridge V2

```bash
JOB_ID=myrun_bridge_v2 \
DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
PRETRAINED_WEIGHTS=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
EPOCHS=15 \
LEARNING_RATE=1e-6 \
NUM_NEGATIVES=10 \
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
bash scripts/train/run_real_synthetic_bridge.sh
```

Important:
- this starts directly from the synthetic S1 checkpoint;
- there is no standalone real fine-tuning stage in between;
- Bridge positive/negative sampling is balanced 50/50;
- generic whole-line positive sequence ranking is disabled for V2;
- shared-island-aware bridge ranking remains active;
- validation runs every epoch;
- later stages use:

```text
Weights/myrun_bridge_v2/checkpoint_best_val.pth
```

not the final epoch simply because it is last.

## S6 — Post-Bridge qualitative real evaluation

```bash
WEIGHTS=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s6_post_bridge_qualitative \
bash scripts/eval/run_stage_qualitative.sh
```

## S7 — Post-Bridge quantitative real evaluation

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s7_post_bridge_quantitative \
bash scripts/eval/run_stage_quantitative.sh
```

Then repeat the Bridge-specific evaluation:

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
BRIDGE_DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
RUN_TAG=myrun_s7_bridge_posttrain \
bash scripts/eval/run_stage_bridge_eval.sh
```

## S8 — Final complete-real evaluation

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s8_final_all_real \
FINAL_THRESHOLD=0.50 \
bash scripts/eval/run_stage_final_all_real.sh
```

This evaluates every real manifest relationship and produces the frozen final summary. Do not tune the model after S8.

---

# Monitoring

```bash
squeue -u "$USER" -o "%.18i %.38j %.2t %.10M %.55R"
```

```bash
cat logs/research_pipeline_<RUN_PREFIX>.jobs
```

```bash
sacct -j <jobids> --format=JobID,JobName%42,State,Elapsed,ExitCode
```

Dependency inspection:

```bash
scontrol show job <JOB_ID> | grep -E 'JobId=|JobState=|Dependency='
```

`afterok:<jobid>` means the next stage becomes eligible only when the previous stage exits successfully.
