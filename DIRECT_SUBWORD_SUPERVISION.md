# Direct connected-subword supervision

The optional direct synthetic mode removes Span-DTW from synthetic training by
using validated renderer-derived horizontal intervals for every connected Arabic
subword. The original connected-subword Span-DTW command remains available as
an ablation and remains the continuation method for real manuscript lines that
do not have trusted subword intervals.

For every labeled subword interval, training:

1. loads both synthetic lines;
2. pools local CNN windows with geometric-overlap and foreground-ink weights;
3. matches local visual regions to AraBERT subword classes with duplicate-neutral
   bidirectional supervised contrastive learning;
4. applies soft focal BCE to partially and fully overlapping local windows;
5. applies smaller contextual-region and contextual-localization losses so the
   BiLSTM learns sequence context without replacing local visual supervision;
6. keeps an attention-style interval auxiliary loss and a hard outside-region
   margin loss; and
7. reports window IoU, center error, and boundary errors during training.

## Balanced multi-source loader

The direct launcher now loads exactly 3,000 samples from each of:

```text
DataSet/Synthetic_Arabic_1
DataSet/Synthetic_Arabic_2
DataSet/Synthetic_Arabic_3
DataSet/Synthetic_Arabic_4
```

The resulting dataset contains 12,000 samples before the deterministic 60/20/20
train, validation, and test split. `SYNTHETIC_REQUIRE_FULL_PER_DIR=1` makes the
job fail rather than silently using fewer than 3,000 samples from any source.

The training split receives online box-safe augmentations:

- contrast and brightness changes;
- autocontrast;
- mild blur;
- mild stroke thickening or erosion; and
- Gaussian scan noise.

These transformations never crop, rotate, translate, horizontally jitter, or
change the image dimensions. Validation and test images remain clean. This is
necessary because the direct objective relies on exact renderer-derived subword
coordinates.

`ZERO_SHOT_PROFILE` therefore remains disabled in direct mode. Its geometric
transformations would invalidate fixed interval coordinates.

## Preview the augmentations

Create contact-sheet PNGs for all four sources before training:

```bash
python scripts/data/preview_synthetic_augmentations.py \
  --data-root "$PWD/DataSet" \
  --profile box-safe \
  --samples-per-source 2 \
  --augmentations 4 \
  --output-dir "$PWD/Results/AugmentationPreview"
```

The script writes one preview sheet per source and full-resolution variants under:

```text
Results/AugmentationPreview/Synthetic_Arabic_*/img1_*/
```

To inspect the geometric zero-shot augmentation used by the Span-DTW experiment
instead, replace `--profile box-safe` with `--profile zero-shot`.

## Recommended fully offline train-and-evaluate pipeline

Run from the repository root on the login node. The command submits one Slurm
training job that prepares all four sources' sidecars inside the compute job,
then immediately queues calibrated synthetic evaluation with an
`afterok:<training_job_id>` dependency.

```bash
JOB_ID=cnn_connected_subword_direct_multisource \
DATA_ROOT="$PWD/DataSet" \
HF_HOME="$PWD/.hf_cache" \
SYNTHETIC_SAMPLES_PER_DIR=3000 \
EPOCHS=35 \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=128 \
LEARNING_RATE=1e-4 \
DIRECT_SUBWORD_BOX_SAFE_AUGMENT=1 \
DIRECT_SUBWORD_AUGMENT_PROBABILITY=0.85 \
DIRECT_SUBWORD_CLEAN_PROBABILITY=0.15 \
DIRECT_SUBWORD_NOISE_STD_MAX=10 \
THRESHOLDS=0.60,0.70,0.80,0.85,0.90 \
CALIBRATION_START_INDEX=1 \
CALIBRATION_SAMPLES=100 \
HOLDOUT_START_INDEX=101 \
HOLDOUT_SAMPLES=100 \
bash scripts/train/run_connected_subword_direct_pipeline.sh
```

The pipeline prints both Slurm job IDs. Monitor them with:

```bash
squeue -u "$USER" -o "%.18i %.42j %.10T %.10M %.10l %.25R"
```

The checkpoint is written to:

```text
Weights/cnn_connected_subword_direct_multisource/model_latest.pth
```

The current synthetic Smith-Waterman evaluator accepts one dataset root, so the
pipeline evaluates on `Synthetic_Arabic_1` by default after training on all four.
Override this with `EVAL_DATA_DIR=/path/to/another/source`.

The final calibrated holdout summary is written to:

```text
Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/
  cnn_connected_subword_direct_multisource_calibrated_holdout/final_summary.json
```

## Custom source list

Override the default four directories with a comma-separated list:

```bash
SYNTHETIC_DATA_DIRS="/data/set_a,/data/set_b,/data/set_c,/data/set_d" \
SYNTHETIC_SAMPLES_PER_DIR=3000 \
JOB_ID=cnn_connected_subword_direct_custom \
bash scripts/train/run_connected_subword_direct_pipeline.sh
```

Each source must contain `images/` and `texts/`. In direct mode, each source also
receives its own sibling `subword_boxes/` directory generated inside Slurm.

## Default direct objective settings

```text
DIRECT_SUBWORD_REGION_WEIGHT=1.0
DIRECT_SUBWORD_CONTEXT_REGION_WEIGHT=0.15
DIRECT_SUBWORD_LOCALIZATION_WEIGHT=1.0
DIRECT_SUBWORD_CONTEXT_LOCALIZATION_WEIGHT=0.25
DIRECT_SUBWORD_ATTENTION_WEIGHT=0.10
DIRECT_SUBWORD_OUTSIDE_WEIGHT=0.25
DIRECT_SUBWORD_TEMPERATURE=0.07
DIRECT_SUBWORD_BCE_TEMPERATURE=0.10
DIRECT_SUBWORD_SIMILARITY_THRESHOLD=0.20
DIRECT_SUBWORD_FOCAL_GAMMA=1.5
DIRECT_SUBWORD_POSITIVE_BOOST=2.0
DIRECT_SUBWORD_OUTSIDE_MARGIN=0.25
DIRECT_SUBWORD_OUTSIDE_TOP_K=8
DIRECT_SUBWORD_USE_INK_WEIGHTING=1
DIRECT_SUBWORD_INK_FLOOR=0.05
```

## Real-data continuation

```bash
JOB_ID=cnn_connected_subword_real \
SYNTHETIC_WEIGHTS="$PWD/Weights/cnn_connected_subword_direct_multisource/model_latest.pth" \
bash scripts/train/run_connected_subword_real.sh
```

Real continuation still uses Span-DTW because the current real-data manifest does
not contain trusted connected-subword boxes.
