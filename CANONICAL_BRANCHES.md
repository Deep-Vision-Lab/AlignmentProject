# Canonical model branches

Three architecture branches are maintained for controlled comparisons:

- `agent/training-speed-optimization` — CNN window encoder; `USE_BILSTM=0/1` controls CNN-only vs CNN+BiLSTM.
- `agent/use-vit-encoder` — patch-projection + Transformer visual encoder.
- `agent/use-dinov3-convnext` — Meta DINOv3 ConvNeXt-Tiny window encoder with optional BiLSTM context.

The branches intentionally share datasets, preprocessing, Arabic text policy, Span-DTW/image-text objectives, negative sampling, DDP runtime, validation/checkpoint format, Bridge V2 augmentation, and Smith-Waterman evaluation. Only the visual architecture should change in controlled comparisons.

The complete protocol lives in:

- `docs/EXPERIMENT_MASTER_PLAN.md`
- `docs/MODEL_AND_CODE_STAGES.md`
- `docs/STAGE_COMMANDS_AND_DEPENDENCIES.md`

## First rule: freeze Bridge V2 before models

Create and validate `DataSet/RealSyntheticBridge_v2` **before submitting CNN, ViT, or DINOv3 model training**. Build it once with the fixed seed and reuse the exact same dataset across all architecture branches.

Submit the one-time CPU build with:

```bash
bash scripts/slurm/submit_bridge_v2_dataset.sh
```

After the job succeeds, validate explicitly:

```bash
python scripts/data/smoke_test_real_synthetic_bridge.py \
  --data-dir DataSet/RealSyntheticBridge_v2
```

The model pipeline refuses to submit GPU jobs if the frozen Bridge V2 manifest/metadata are missing or the smoke test fails.

## Revised curriculum

The former standalone canonical-real fine-tuning/evaluation block is no longer part of the active protocol.

Active order:

```text
D0 freeze Bridge V2 dataset
S1 synthetic pretraining
S2 qualitative zero-shot real evaluation
S3 quantitative zero-shot real evaluation
S4 Bridge V2 pre-finetune evaluation
S5 direct Bridge V2 fine-tuning from S1
S6 post-Bridge qualitative real evaluation
S7 post-Bridge quantitative + Bridge-specific evaluation
S8 final complete-real evaluation
```

Submit S1-S8 from any canonical architecture branch with:

```bash
BRIDGE_EPOCHS=15 BRIDGE_LR=1e-6 \
bash scripts/slurm/submit_full_research_pipeline.sh
```

All model dependencies use `afterok`.

## CNN compatibility

New CNN experiments default to ResNet-18. Historical ResNet-34 checkpoints remain supported.

```bash
# CNN only
CNN_BACKBONE=resnet18 USE_BILSTM=0 ...

# CNN + BiLSTM
CNN_BACKBONE=resnet18 USE_BILSTM=1 ...
```

## Synthetic training

Use the active branch backend through:

```bash
JOB_ID=my_fixed63_run \
bash scripts/train/run_branch_fixed63_synthetic.sh
```

The branch-aware runtime enters `training_runtime/entrypoint.py`; do not use legacy direct `train.py` launchers for controlled architecture comparisons.

## RealSyntheticBridge V2 design

The frozen corpus contains, per leakage-safe real anchor group:

- one genuine real manuscript anchor;
- one synthetic positive containing 1, 2, or 3 ordered shared transcript islands;
- guaranteed-unrelated synthetic distractors before, between, and/or after shared islands;
- one `128 x 1024` alignment mask with white shared regions and black distractor regions;
- four synthetic negatives by default;
- no negative or positive distractor may share a complete normalized word with the real anchor;
- no negative or positive distractor may share a normalized character trigram with the real anchor.

The builder writes `images/`, `texts/`, `masks/`, `dataset_manifest.jsonl`, and `metadata.json` and runs the Bridge smoke test.

## Direct Bridge V2 adaptation

Bridge V2 now replaces the old intermediate real-only adaptation stage. Fine-tuning starts directly from the synthetic S1 checkpoint:

```bash
PRETRAINED_WEIGHTS=/absolute/path/to/synthetic/checkpoint_latest.pth \
DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
JOB_ID=my_bridge_v2 \
EPOCHS=15 \
LEARNING_RATE=1e-6 \
bash scripts/train/run_real_synthetic_bridge.sh
```

Canonical Bridge settings:

- maximum 15 epochs;
- validation every epoch;
- use `checkpoint_best_val.pth` for downstream evaluation;
- balanced 50/50 Bridge positive/no-shared rows;
- 10 image-text negatives per positive, 4 active hardest Span-DTW negatives;
- AraBERT backbone frozen;
- bridge-specific real-image/synthetic-text ranking restricted to actual shared islands;
- generic whole-positive-line sequence ranking disabled because V2 positives intentionally contain distractors;
- masks retained for diagnostics/future ablations but no separate mask loss in the baseline.

If validation is still improving and the best checkpoint occurs at epoch 15, continuation to 20 total epochs is a later controlled extension rather than silently changing the first protocol.

## DINOv3 ConvNeXt setup

For S1 from-scratch DINO training:

```bash
export DINOV3_REPO_DIR=/absolute/path/to/dinov3
export DINOV3_WEIGHTS=/absolute/path/to/the-authorized-convnext-tiny-checkpoint
```

The DINO backbone is frozen by default for the first controlled comparison. Unfreezing is a separate ablation.

## Shared evaluation

The common positive/no-shared threshold sweep remains:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pth \
RUN_NAME=my_run_discrimination \
bash scripts/eval/run_real_discrimination_sweep.sh
```

Thresholds are `0.40, 0.50, 0.60, 0.65, 0.70`.

The final frozen all-real evaluation is:

```bash
CHECKPOINT=/absolute/path/to/checkpoint_best_val.pth \
RUN_TAG=my_final_run \
bash scripts/eval/run_stage_final_all_real.sh
```

It evaluates every canonical manifest row, reporting `high_match`, `medium_match`, `low_match`, and `no_shared_content` separately; `low_match` remains its own ambiguous/partial class.

## Synchronization rule

Shared data, training, loss, evaluation, preprocessing, documentation, and launcher changes must remain synchronized across the three architecture branches. Architecture-specific construction belongs in `model_backend.py`, the selected encoder implementation, or the branch-specific `docs/MODEL_AND_CODE_STAGES.md`.
