# AlignmentProject — Stage Commands and SLURM Dependencies

This file is the operational companion to `docs/EXPERIMENT_MASTER_PLAN.md`.

The commands below are identical on the CNN+BiLSTM, ViT, and DINOv3 branches. `model_backend.py` selects the architecture. Do not manually substitute one branch's checkpoint into another branch.

## 1. Pull the branch and inspect the backend

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git fetch origin
git switch <BRANCH>
git pull --ff-only origin <BRANCH>
python - <<'PY'
import model_backend
print(model_backend.MODEL_NAME)
PY
```

Canonical branches:

```text
agent/training-speed-optimization   # CNN / CNN+BiLSTM
agent/use-vit-encoder               # ViT
agent/use-dinov3-convnext           # DINOv3 ConvNeXt
```

For the DINO branch set the local official repository before submission:

```bash
export DINOV3_REPO_DIR=/path/to/local/dinov3
```

## 2. Recommended command: submit the complete dependency chain

```bash
bash scripts/slurm/submit_full_research_pipeline.sh
```

This submits S1 -> S2 -> S3 -> S4 -> S5 -> S6 -> S7 -> S8 -> S9 -> S10 -> S11 with `afterok` dependencies. Every downstream job waits until the previous job exits successfully. A failure blocks the remainder of the chain.

Useful overrides:

```bash
RUN_PREFIX=my_cnn_run \
SYNTH_EPOCHS=20 \
REAL_EPOCHS=5 \
BRIDGE_EPOCHS=8 \
FINAL_THRESHOLD=0.50 \
bash scripts/slurm/submit_full_research_pipeline.sh
```

DINO example:

```bash
DINOV3_REPO_DIR=$HOME/BGU-Lab/dinov3 \
RUN_PREFIX=dinov3_main \
bash scripts/slurm/submit_full_research_pipeline.sh
```

The submitter writes a job-id ledger under:

```text
logs/research_pipeline_<RUN_PREFIX>.jobs
```

Monitor the chain:

```bash
squeue -u "$USER" -o "%.18i %.38j %.2t %.10M %.55R"
```

Accounting after jobs start/finish:

```bash
cat logs/research_pipeline_<RUN_PREFIX>.jobs
sacct -j <comma-separated-job-ids> --format=JobID,JobName%42,State,Elapsed,ExitCode
```

---

# Individual stage commands

These are useful for rerunning one stage manually. The automatic submitter is preferred for the full experiment.

## S1 — Synthetic pretraining

```bash
JOB_ID=myrun_synth \
EPOCHS=20 \
LEARNING_RATE=1e-4 \
NUM_NEGATIVES=10 \
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
bash scripts/train/run_branch_fixed63_synthetic.sh
```

Meaning: trains the active branch visual encoder on `DataSet/AugmentedArabicDataset63` using synthetic image-text supervision.

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

Meaning: saves real-image pair heatmaps and Smith-Waterman paths for high, medium, low, and no-shared examples.

## S3 — Quantitative zero-shot real evaluation

```bash
CHECKPOINT=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
RUN_TAG=myrun_s3_synth_quantitative \
bash scripts/eval/run_stage_quantitative.sh
```

Meaning: runs held-out real localization/bbox metrics plus the fixed positive-vs-no-shared threshold sweep.

## S4 — Canonical real fine-tuning, no augmentation

```bash
JOB_ID=myrun_real \
PRETRAINED_WEIGHTS=$PWD/Weights/myrun_synth/checkpoint_latest.pth \
EPOCHS=5 \
LEARNING_RATE=2e-6 \
bash scripts/train/run_stage_real_finetune.sh
```

Meaning: adapts the synthetic checkpoint to canonical train-safe real high/medium pairs with augmentation disabled.

Expected checkpoint:

```text
Weights/myrun_real/checkpoint_latest.pth
```

## S5 — Qualitative evaluation after real fine-tuning

```bash
WEIGHTS=$PWD/Weights/myrun_real/checkpoint_latest.pth \
RUN_TAG=myrun_s5_real_qualitative \
bash scripts/eval/run_stage_qualitative.sh
```

## S6 — Quantitative evaluation after real fine-tuning

```bash
CHECKPOINT=$PWD/Weights/myrun_real/checkpoint_latest.pth \
RUN_TAG=myrun_s6_real_quantitative \
bash scripts/eval/run_stage_quantitative.sh
```

## S7 — Build and audit RealSyntheticBridge V2

Small smoke corpus first, when developing the generator:

```bash
OVERWRITE=1 \
MAX_ANCHORS=50 \
OUTPUT_DIR=$PWD/DataSet/RealSyntheticBridge_v2_smoke \
bash scripts/data/build_real_conditioned_synthetic_bridge.sh
```

Full corpus:

```bash
OVERWRITE=1 \
MAX_ANCHORS=0 \
OUTPUT_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
bash scripts/data/build_real_conditioned_synthetic_bridge.sh
```

Meaning: creates one real anchor group with one 1-3-island synthetic positive, its alignment mask, and guaranteed synthetic negatives. The build command runs the smoke validator automatically.

## S8 — Evaluate augmentation before fine-tuning on it

```bash
CHECKPOINT=$PWD/Weights/myrun_real/checkpoint_latest.pth \
BRIDGE_DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
RUN_TAG=myrun_s8_bridge_pretrain_eval \
bash scripts/eval/run_stage_bridge_eval.sh
```

Meaning: evaluates the real-fine-tuned model on bridge positives/negatives before bridge training, and reruns dataset integrity checks.

## S9 — Fine-tune on bridge augmentation

```bash
JOB_ID=myrun_bridge_v2 \
DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
PRETRAINED_WEIGHTS=$PWD/Weights/myrun_real/checkpoint_latest.pth \
EPOCHS=8 \
LEARNING_RATE=7.5e-7 \
bash scripts/train/run_real_synthetic_bridge.sh
```

Meaning: balanced 50/50 bridge positive/negative fine-tuning, with real-image/synthetic-text ranking restricted to the actual shared islands. The bridge runtime preserves:

```text
Weights/myrun_bridge_v2/checkpoint_best_val.pth
```

## S10 — Post-bridge qualitative and quantitative evaluation

```bash
WEIGHTS=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s10_post_bridge_qualitative \
bash scripts/eval/run_stage_qualitative.sh
```

Then:

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s10_post_bridge_quantitative \
bash scripts/eval/run_stage_quantitative.sh
```

Optionally repeat the bridge-specific evaluation:

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
BRIDGE_DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
RUN_TAG=myrun_s10_bridge_posttrain \
bash scripts/eval/run_stage_bridge_eval.sh
```

## S11 — Final complete-real evaluation

```bash
CHECKPOINT=$PWD/Weights/myrun_bridge_v2/checkpoint_best_val.pth \
RUN_TAG=myrun_s11_final_all_real \
FINAL_THRESHOLD=0.50 \
bash scripts/eval/run_stage_final_all_real.sh
```

Meaning: evaluates every manifest row from every real relationship label and produces `final_summary.json`. This stage is final: do not tune the model from its results.

---

# How SLURM dependency chaining works

The full submitter uses this pattern:

```bash
J1=$(sbatch --parsable stage1_script.sh)
J2=$(sbatch --parsable --dependency=afterok:${J1} stage2_script.sh)
J3=$(sbatch --parsable --dependency=afterok:${J2} stage3_script.sh)
```

`afterok:<jobid>` means the next job is eligible to run only if the dependency finished with exit code 0.

Useful dependency inspection:

```bash
scontrol show job <JOB_ID> | grep -E 'JobId=|JobState=|Dependency='
```

Cancel the entire remaining experiment if necessary:

```bash
scancel <job1> <job2> <job3> ...
```

Do not replace `afterok` with `afterany` for the scientific pipeline. `afterany` would continue even after failed training/evaluation and could create misleading downstream results.
