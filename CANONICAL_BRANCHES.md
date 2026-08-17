# Canonical model branches

Three architecture branches are maintained for controlled comparisons:

- `agent/training-speed-optimization` — CNN window encoder. New runs default to
  ResNet-18; historical ResNet-34 checkpoints remain supported. Use
  `USE_BILSTM=0` for CNN-only and `USE_BILSTM=1` for CNN + BiLSTM.
- `agent/use-vit-encoder` — pure patch-projection + Transformer visual encoder.
- `agent/use-dinov3-convnext` — Meta DINOv3 ConvNeXt-Tiny window encoder, with
  optional BiLSTM sequence context through `USE_BILSTM=0/1`.

The branches intentionally share datasets, preprocessing, frozen Arabic text
encoder, Span-D3TW/image-text objectives, negative sampling, DDP runtime, SLURM
resources, validation/checkpoint format, and real Smith-Waterman diagnostics.
Architecture-specific construction lives behind `model_backend.py` plus the selected
encoder implementation.

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

Use the same public command on the active architecture branch:

```bash
JOB_ID=my_fixed63_run \
bash scripts/train/run_augmented_synthetic_27k_fixed63.sh
```

The wrapper validates the 27k fixed-63 corpus and enters
`training_runtime/entrypoint.py`, so the checked-out branch actually controls model
construction. Do not use a legacy direct `train.py` launcher for architecture
comparisons.

## Offline real-conditioned synthetic bridge

The bridge corpus is generated once on CPU. No Arabic rendering occurs in the GPU
training loop.

```bash
sbatch scripts/data/build_real_conditioned_synthetic_bridge.sbatch
```

Default per leakage-safe real anchor:

- 1 positive synthetic line containing an exact contiguous span from its transcript;
- 4 negative synthetic lines from other training transcripts;
- negatives are rejected if they share a normalized word of length >=3 or a
  normalized character 4-gram with the full anchor transcript.

The builder automatically runs:

```bash
python scripts/data/smoke_test_real_synthetic_bridge.py \
  --data-dir DataSet/RealSyntheticBridge_v1
```

before declaring the offline corpus ready.

Fine-tune a checkpoint from the currently checked-out architecture with:

```bash
PRETRAINED_WEIGHTS=/absolute/path/to/model_latest.pth \
JOB_ID=my_bridge_v1 \
bash scripts/train/run_real_synthetic_bridge.sh
```

Bridge v1 uses four complementary signals:

- real image <-> genuine real transcript;
- synthetic image <-> its exactly known synthetic transcript;
- positive/negative real-image <-> synthetic-image sequence discrimination;
- **direct real-image <-> synthetic-text sequence ranking**: text from a positive
  rendered sample must form a stronger/longer local path in the real anchor than
  text from a guaranteed-negative rendered sample.

The direct term detaches the text embeddings, so its gradients update the visual
representation rather than moving the semantic target space. It deliberately
penalizes a coherent negative text sequence, not every isolated negative character,
because unrelated Arabic lines can still contain legitimate repeated letters.
Training is 50/50 positive/no-shared and preserves `checkpoint_best_val.pth` whenever
validation improves.

Useful bridge knobs include:

```bash
BRIDGE_CROSS_TEXT_WEIGHT=0.10
BRIDGE_CROSS_TEXT_THRESHOLD=0.50
BRIDGE_CROSS_TEXT_PATH_MARGIN=0.10
BRIDGE_CROSS_TEXT_POSITIVE_FLOOR=0.20
BRIDGE_CROSS_TEXT_NEGATIVE_CEILING=0.15
```

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

## Shared real discrimination evaluation

The no-PNG fixed-manifest sweep remains the common scientific comparison:

```bash
CHECKPOINT=/absolute/path/to/checkpoint.pth \
RUN_NAME=my_run_discrimination \
bash scripts/eval/run_real_discrimination_sweep.sh
```

It measures the same positive/no-shared real pairs at thresholds
`0.40, 0.50, 0.60, 0.65, 0.70`.

## Synchronization rule

Shared data, loss, training, evaluation, preprocessing, and script changes must be
kept synchronized across the three architecture branches. Architecture-specific
changes belong in `model_backend.py` or the selected encoder implementation.
