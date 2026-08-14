# CNN Training Plan: Stage 1 -> Joint Real Discrimination

This is the canonical CNN experiment after the failed absolute no-shared and transcript-group ranking ablations.

## Goal

Start from the clean Stage-1 synthetic checkpoint and teach the real dataset once under the final interpretation:

1. image-text Span-DTW supervision,
2. local relative hard-negative discrimination,
3. positive image-image span contrastive supervision,
4. positive-vs-no-shared sequence-ranking supervision,
5. clean original real images plus fresh online augmentations from code.

Do **not** use the pre-generated `ArabicDatasetRealAug10K` dataset for this experiment.

## Data policy

- Source dataset: `DataSet/ArabicDataset` only.
- `NUM_SAMPLES=0`: use the full canonical manifest.
- Split complete `pair_id` groups, not individual rows.
- Default split: 80% train / 10% validation / 10% test.
- No-shared rows whose pair/page touches validation or test are excluded from training.
- Validation and test are clean only.
- Training exposure is balanced 50/50 positive vs no-shared so sequence ranking is active reliably.
- Within each class, training views are 1 clean : 2 online-augmented (~33/67).
- Effective epoch length: 6x the clean positive+no-shared training pool by default.
- Online augmentation comes from `RealDataAugmentation.py` / `AugmentedRealDataLoader.py`.
- Initial experiment uses appearance/ink/scan augmentation; line stitching stays disabled.

## Stage 0 - update the canonical CNN branch

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git fetch origin
git switch agent/training-speed-optimization
git pull --ff-only origin agent/training-speed-optimization
git rev-parse --short HEAD
```

## Stage 1 - pretrained model

Reuse the existing Stage-1 synthetic checkpoint:

```bash
ls -lh Weights/cnn_bilstm_augmented_fixed63_27k/model_latest.pth
```

Do not retrain Stage 1 unless reproduction is required.

## Stage 2 - joint real pilot: 5 epochs

```bash
JOB_ID=cnn_joint_real_from_stage1_v1 \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_bilstm_augmented_fixed63_27k/model_latest.pth" \
EPOCHS=5 \
LEARNING_RATE=1e-5 \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=64 \
REAL_TRAIN_FRACTION=0.80 \
REAL_VALID_FRACTION=0.10 \
REAL_CLEAN_VIEWS_PER_CYCLE=1 \
REAL_AUG_VIEWS_PER_CYCLE=2 \
REAL_EFFECTIVE_EPOCH_MULTIPLIER=6 \
NUM_NEGATIVES=10 \
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
bash scripts/train/run_stage1_joint_real_discrimination.sh
```

### Stage-2 training check

```bash
LOG=$(ls -t out/cnn_joint_real_from_stage1_v1_*.out | head -1)
grep -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' "$LOG" | head -20
grep 'sequence_batch' "$LOG" | head -20
grep 'sequence_batch' "$LOG" | tail -20
```

Expected data recipe:

```text
split=0.80/0.10/0.10
positive_ratio=0.500
no_shared_ratio=0.500
clean_ratio=0.333
augmented_ratio=0.667
online_augmentation=True
```

## Stage 2 evaluation

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1 \
N_SAMPLES=20 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

The fixed sweep evaluates thresholds `0.40 0.50 0.60 0.65 0.70` and reports score/path/matched AUROCs.

Phase-3 structural reference:

```text
path_steps AUC       = 0.6800
matched_fraction AUC = 0.6725
positive path steps  = 26.45
negative path steps  = 16.25
```

## Stage 2 automatic gate

```bash
python scripts/eval/check_real_discrimination_gate.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1
```

Default continuation gate requires one threshold where:

- `steps_AUC > 0.6800` **or** `matched_AUC > 0.6725`,
- positive mean path steps >= 8,
- positive-minus-negative mean path-step gap >= 2.

The script writes `gate_result.json` and exits nonzero if the gate fails.

## Stage 3 - 10-epoch continuation, only after Stage-2 gate passes

Start from the 5-epoch pilot checkpoint and lower the learning rate:

```bash
JOB_ID=cnn_joint_real_from_stage1_v1_cont10 \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_joint_real_from_stage1_v1/model_latest.pth" \
EPOCHS=10 \
LEARNING_RATE=5e-6 \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=64 \
REAL_TRAIN_FRACTION=0.80 \
REAL_VALID_FRACTION=0.10 \
REAL_CLEAN_VIEWS_PER_CYCLE=1 \
REAL_AUG_VIEWS_PER_CYCLE=2 \
REAL_EFFECTIVE_EPOCH_MULTIPLIER=6 \
NUM_NEGATIVES=10 \
SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE=4 \
bash scripts/train/run_stage1_joint_real_discrimination.sh
```

### Stage-3 training check

```bash
LOG=$(ls -t out/cnn_joint_real_from_stage1_v1_cont10_*.out | head -1)
grep -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' "$LOG" | head -20
grep 'sequence_batch' "$LOG" | head -20
grep 'sequence_batch' "$LOG" | tail -20
```

## Stage 3 evaluation

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1_cont10/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1_cont10 \
N_SAMPLES=20 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

## Stage 3 gate

```bash
python scripts/eval/check_real_discrimination_gate.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1_cont10
```

## Stage 4 - larger final evaluation, only after Stage-3 gate passes

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1_cont10/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1_cont10_full \
N_SAMPLES=100 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

Final summary/check:

```bash
echo '=== PILOT 5-EPOCH ==='
python scripts/eval/summarize_real_discrimination.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1

echo '=== CONTINUATION 10-EPOCH ==='
python scripts/eval/summarize_real_discrimination.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1_cont10

echo '=== FINAL LARGE EVAL ==='
python scripts/eval/summarize_real_discrimination.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1_cont10_full

python scripts/eval/check_real_discrimination_gate.py \
  Results/Evaluation/Representation_Diagnostics/cnn_joint_real_from_stage1_v1_cont10_full \
  --no-fail
```

## Overnight automatic SLURM dependency chain

If the 5-epoch pilot is already running, do not modify the working tree while it runs. Fetch the helper directly from the remote branch into `/tmp` and submit the chain:

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject

git fetch origin

git show \
  origin/agent/training-speed-optimization:scripts/slurm/submit_joint_real_overnight_pipeline.sh \
  > /tmp/submit_joint_real_overnight_pipeline.sh

TRAIN5_JOB_ID=$(squeue -h -u "$USER" -n cnn_joint_real_from_stage1_v1 -o '%A' | head -1)

bash /tmp/submit_joint_real_overnight_pipeline.sh "$TRAIN5_JOB_ID"
```

The helper submits:

```text
5-epoch pilot (already running)
    -> afterok: training/config check + branch sync
    -> afterok: 20-pair evaluation sweep
    -> afterok: structural gate
       -> if PASS: 10-epoch continuation
           -> afterok: training/config check
           -> afterok: 20-pair evaluation sweep
           -> afterok: structural gate
              -> if PASS: 100-pair final sweep
                   -> afterok: final summaries/check
```

If either gate fails, its exit code is nonzero and the downstream `afterok` jobs are not released.

Inspect the whole queued chain with:

```bash
squeue -u "$USER" -o '%.18i %.32j %.2t %.10M %.40R'
```

Inspect finished jobs with:

```bash
sacct -u "$USER" --starttime today \
  --format=JobID,JobName,State,Elapsed,ExitCode
```

## Final selection

Compare at least:

1. `cnn_bilstm_phase3` old baseline,
2. `cnn_joint_real_from_stage1_v1` 5-epoch pilot,
3. `cnn_joint_real_from_stage1_v1_cont10` if the gate released it.

Select primarily on path-steps AUROC, matched-fraction AUROC, positive-vs-no-shared path-length separation, and preservation of substantial positive paths. SW-score AUROC is secondary. Do not choose a model merely because negative cosine is lower.

## Historical ablations not to resume

Do not use these as pretrained checkpoints for this clean experiment:

```text
cnn_phase4_no_shared_negatives
cnn_bilstm_real_final_no_shared_neg
cnn_bilstm_real_ranking_v1
```

The new objective deliberately avoids absolute repulsion of all visually different Arabic windows. It learns relative local discrimination plus coherent sequence discrimination instead.
