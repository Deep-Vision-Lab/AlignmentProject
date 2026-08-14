# CNN Training Plan: Stage 1 -> Joint Real Discrimination

This is the canonical next CNN experiment after the failed absolute no-shared and transcript-group ranking ablations.

## Goal

Start from the clean Stage-1 synthetic checkpoint and teach the real dataset once under the final interpretation:

1. image-text Span-DTW supervision,
2. local hard-negative discrimination,
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
- Training exposure ratio: 1 clean view : 2 online-augmented views (~33/67).
- Effective epoch length: 6x the clean training pool by default.
- Therefore an underlying row receives roughly two clean and four fresh augmented exposures per epoch on average.
- Online augmentation uses `RealDataAugmentation.py` / `AugmentedRealDataLoader.py`; no offline augmented dataset is read.
- Initial run uses appearance/ink/scan augmentation only; line stitching stays disabled for a clean first ablation.

## Stage 0 - update the canonical CNN branch

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git fetch origin
git switch agent/training-speed-optimization
git pull --ff-only origin agent/training-speed-optimization
git rev-parse --short HEAD
```

Use the commit printed in the current project instructions as the expected head.

## Stage 1 - pretrained model

Reuse the existing synthetic Stage-1 checkpoint; do not retrain Stage 1 unless reproduction is required.

```bash
ls -lh Weights/cnn_bilstm_augmented_fixed63_27k/model_latest.pth
```

Canonical Stage-1 checkpoint:

```text
Weights/cnn_bilstm_augmented_fixed63_27k/model_latest.pth
```

Optional Stage-1 reproduction only:

```bash
JOB_ID=cnn_bilstm_augmented_fixed63_27k \
bash scripts/train/run_augmented_synthetic_27k_fixed63.sh
```

## Stage 2 - joint real pilot (5 epochs)

This replaces the old separate real/non-augmented then augmented-real curriculum.

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

### Startup checks

```bash
LOG=$(ls -t out/cnn_joint_real_from_stage1_v1_*.out | head -1)

grep -E 'Joint real training dataset|Joint real objective installed|objective=sequence_ranking' "$LOG" | head -20
```

Expected data line should report approximately:

```text
split=0.80/0.10/0.10
clean_ratio=0.333
augmented_ratio=0.667
online_augmentation=True
```

It must show the canonical `DataSet/ArabicDataset`, not `ArabicDatasetRealAug10K`.

### Training diagnostics

```bash
grep 'sequence_batch' "$LOG" | head -20
grep 'sequence_batch' "$LOG" | tail -20
```

Desired direction:

- positive path fraction stays healthy,
- no-shared path fraction decreases relative to positive,
- `hard_frac_gap` increases,
- `hard_score_gap` increases,
- sequence ranking loss decreases,
- no representation-wide collapse.

## Stage 2 evaluation - run immediately after the 5-epoch pilot

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1 \
N_SAMPLES=20 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

This runs thresholds `0.40 0.50 0.60 0.65 0.70` on the same fixed positive and no-shared diagnostic manifests and prints score/path/matched AUROCs.

Reference to beat from the old Phase-3 model at `T=0.65`:

```text
path_steps AUC       = 0.6800
matched_fraction AUC = 0.6725
positive path steps  = 26.45
negative path steps  = 16.25
```

### Pilot gate

Continue only if the new model does **not** collapse positive paths and improves the structural discrimination signal. Preferred gate:

- `path_steps AUC > 0.680`, or
- `matched_fraction AUC > 0.6725`,
- positive mean path length clearly above no-shared,
- positive paths remain substantial (not ~1-5 windows as in failed runs).

If this gate fails, stop and keep the Stage-1 checkpoint plus Phase-3 baseline; do not add more epochs blindly.

## Stage 3 - continuation only after a successful pilot

Start from the pilot checkpoint, keep the same data recipe/objectives, and reduce the learning rate.

```bash
JOB_ID=cnn_joint_real_from_stage1_v1_cont \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_joint_real_from_stage1_v1/model_latest.pth" \
EPOCHS=5 \
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

## Stage 3 evaluation

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1_cont/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1_cont \
N_SAMPLES=20 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

If the 20-pair fixed diagnostic improves, run a larger evaluation if the manifests contain enough rows:

```bash
CHECKPOINT="$PWD/Weights/cnn_joint_real_from_stage1_v1_cont/model_latest.pth" \
RUN_NAME=cnn_joint_real_from_stage1_v1_cont_full \
N_SAMPLES=100 \
bash scripts/eval/run_real_discrimination_sweep.sh
```

## Stage 4 - final selection

Compare at least:

1. `cnn_bilstm_phase3` (old baseline),
2. `cnn_joint_real_from_stage1_v1` (5-epoch pilot),
3. `cnn_joint_real_from_stage1_v1_cont` (only if continuation gate passed).

Select the checkpoint based primarily on:

- path-steps AUROC,
- matched-fraction AUROC,
- positive vs no-shared path-length separation,
- SW score AUROC as a secondary metric,
- preservation of meaningful positive path length.

Do not select a model merely because mean cosine is lower for negatives.

## Important historical ablations (do not resume)

Do not use these as pretrained weights for the clean experiment:

```text
cnn_phase4_no_shared_negatives
cnn_bilstm_real_final_no_shared_neg
cnn_bilstm_real_ranking_v1
```

The absolute negative objective collapsed useful similarity structure, and the transcript-group ranking objective compressed both positive and negative paths.

## Why the new curriculum is different

The new run does not ask every visually different Arabic window to have a low absolute cosine. Instead it keeps relative hard-negative discrimination while training sequence coherence:

```text
same/correct local correspondence > hard confusing correspondence + margin
true positive sequence            > no-shared sequence + margin
```

This permits isolated Arabic stroke similarities while requiring the true pair to form the stronger coherent alignment.
