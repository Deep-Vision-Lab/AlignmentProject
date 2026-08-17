# Canonical model branches

Three architecture branches are maintained for controlled comparisons:

- `agent/training-speed-optimization` — CNN window encoder. New runs default to
  ResNet-18; historical ResNet-34 checkpoints remain supported. Use
  `USE_BILSTM=0` for CNN-only and `USE_BILSTM=1` for CNN + BiLSTM.
- `agent/use-vit-encoder` — pure patch-projection + Transformer visual encoder.
- `agent/use-dinov3-convnext` — Meta DINOv3 ConvNeXt-Tiny window encoder, with
  optional BiLSTM sequence context through `USE_BILSTM=0/1`.

The branches intentionally share datasets, preprocessing, frozen Arabic language
backbone, trainable shared-space text projection head, Span-D3TW/image-text
objectives, negative sampling, DDP runtime, SLURM resources, validation/checkpoint
format, and real Smith-Waterman diagnostics. Architecture-specific construction
lives behind `model_backend.py` plus the selected encoder implementation.

The complete controlled curriculum is documented in:

- `docs/EXPERIMENT_MASTER_PLAN.md`
- `docs/MODEL_AND_CODE_STAGES.md`
- `docs/STAGE_COMMANDS_AND_DEPENDENCIES.md`

Submit the whole dependency chain from any canonical branch with:

```bash
bash scripts/slurm/submit_full_research_pipeline.sh
```

## CNN backbone compatibility

New CNN experiments use:

```bash
CNN_BACKBONE=resnet18
```

The strong historical Stage-1/R0/R1/R2 checkpoints were built with the modified
ResNet-34. Launchers/evaluation resolve those checkpoints as `resnet34` rather than
trying to load them into ResNet-18. This lets us compare the new smaller backbone
without invalidating prior results.

CNN-only and CNN+BiLSTM are modes of the same branch, not separate source branches:

```bash
# CNN only
CNN_BACKBONE=resnet18 USE_BILSTM=0 ...

# CNN + BiLSTM
CNN_BACKBONE=resnet18 USE_BILSTM=1 ...
```

## Branch-aware fixed-63 synthetic training

Use the same public branch-aware launcher on the active architecture branch:

```bash
JOB_ID=my_fixed63_run \
bash scripts/train/run_branch_fixed63_synthetic.sh
```

The launcher enters `training_runtime/entrypoint.py`, so `model_backend.py` in the
checked-out branch controls model construction. Do not use a legacy direct
`train.py` launcher for architecture comparisons.

## RealSyntheticBridge V2

The bridge corpus is generated once on CPU and reused across architectures. No
Arabic rendering occurs in the GPU training loop.

Build/validate it with:

```bash
BRIDGE_DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
bash scripts/data/prepare_real_synthetic_bridge_v2.sh
```

Default per leakage-safe real anchor group:

- one genuine real manuscript anchor;
- one synthetic positive containing 1, 2, or 3 shared transcript islands in their
  original order;
- guaranteed-unrelated synthetic distractor content before, between, or after the
  shared islands;
- one `128 x 1024` alignment mask: white shared regions and black unaligned regions;
- four synthetic negative lines by default;
- no negative or positive distractor may share a complete normalized word with the
  real anchor;
- no negative or positive distractor may share a normalized character trigram with
  the real anchor;
- isolated letters and bigrams may repeat, keeping negatives realistic rather than
  creating an alphabet-level shortcut.

The builder writes `images/`, `texts/`, `masks/`, `dataset_manifest.jsonl`, and
`metadata.json`, then runs `scripts/data/smoke_test_real_synthetic_bridge.py`.

Fine-tune the active architecture with:

```bash
PRETRAINED_WEIGHTS=/absolute/path/to/checkpoint_latest.pth \
DATA_DIR=$PWD/DataSet/RealSyntheticBridge_v2 \
JOB_ID=my_bridge_v2 \
bash scripts/train/run_real_synthetic_bridge.sh
```

Bridge V2 training keeps the AraBERT backbone frozen while preserving the
trainable projection/normalization/special embeddings that map text into the shared
representation. Full image-text supervision remains valid for each line's own
transcript. The bridge-specific real-image/synthetic-text ranking term is restricted
to the actual shared islands. The older generic whole-positive-line sequence ranking
is disabled because the positive intentionally contains unaligned distractors.
Alignment masks are propagated for diagnostics/future ablations; the canonical V2
baseline does not add a separate mask loss.

Training is balanced 50/50 positive/no-shared and preserves
`checkpoint_best_val.pth` whenever bridge validation improves.

## DINOv3 ConvNeXt setup

The DINO branch uses Meta's official DINOv3 repository locally. Set:

```bash
export DINOV3_REPO_DIR=/absolute/path/to/dinov3
export DINOV3_WEIGHTS=/absolute/path/to/the-authorized-convnext-tiny-checkpoint
```

The original DINOv3 weights are required for the initial foundation-model training
run. A later AlignmentProject checkpoint contains the complete DINO state, so its
evaluation/fine-tuning only needs the local official architecture source.

The DINO backbone is frozen by default for the first controlled experiment:

```bash
DINOV3_FREEZE_BACKBONE=1
```

Unfreezing is a separate ablation, not mixed into the first comparison.

## Shared real evaluation

The fixed positive/no-shared sweep remains the common discrimination diagnostic:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pth \
RUN_NAME=my_run_discrimination \
bash scripts/eval/run_real_discrimination_sweep.sh
```

It measures the same deterministic positive/no-shared real pairs at thresholds
`0.40, 0.50, 0.60, 0.65, 0.70`.

The final frozen all-real stage is:

```bash
CHECKPOINT=/absolute/path/to/checkpoint_best_val.pth \
RUN_TAG=my_final_run \
bash scripts/eval/run_stage_final_all_real.sh
```

It evaluates every canonical manifest row, reporting `high_match`, `medium_match`,
`low_match`, and `no_shared_content` separately and high+medium vs no-shared binary
discrimination. `low_match` remains a separate ambiguous/partial class.

## Synchronization rule

Shared data, loss, training, evaluation, preprocessing, documentation, and script
changes must be kept synchronized across the three architecture branches.
Architecture-specific changes belong in `model_backend.py`, the selected encoder
implementation, or the branch-specific `docs/MODEL_AND_CODE_STAGES.md`.
