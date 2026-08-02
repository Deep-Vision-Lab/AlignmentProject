# Direct connected-subword supervision

The optional direct synthetic mode removes Span-DTW from synthetic training by
using validated renderer-derived horizontal intervals for every connected Arabic
subword. The original connected-subword Span-DTW command remains available as
an ablation and remains the continuation method for real manuscript lines that
do not have trusted subword intervals.

For every labeled subword interval, training now:

1. loads both synthetic lines even though the separate image-pair loss is off;
2. pools **local CNN windows** with geometric-overlap and foreground-ink weights;
3. matches local visual regions to AraBERT subword classes with duplicate-neutral
   bidirectional supervised contrastive learning;
4. applies soft focal BCE to every partially or fully overlapping local window;
5. applies smaller contextual-region and contextual-localization losses so the
   BiLSTM learns sequence context without replacing local visual supervision;
6. keeps an attention-style interval auxiliary loss and a hard outside-region
   margin loss; and
7. reports window IoU, center error, and boundary errors during training.

The sidecar builder hashes the renderer configuration, rewrites stale sidecars,
validates interval bounds/order/ink coverage, and saves sample overlays under
`DataSet/Synthetic_Arabic/subword_boxes/overlays/`.

## Recommended train-and-evaluate pipeline

Run from the repository root on the login node. This command submits training
and immediately queues calibrated synthetic evaluation with
`afterok:<training_job_id>`, so evaluation starts only when training succeeds.

```bash
JOB_ID=cnn_connected_subword_direct_v2 \
DATA_DIR="$PWD/DataSet/Synthetic_Arabic" \
HF_HOME="$PWD/.hf_cache" \
NUM_SAMPLES=8000 \
EPOCHS=35 \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=128 \
LEARNING_RATE=1e-4 \
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
Weights/cnn_connected_subword_direct_v2/model_latest.pth
```

The final calibrated holdout summary is written to:

```text
Results/Evaluation/CNN_BiLSTM_ConnectedSubword_Direct/Synthetic/
  cnn_connected_subword_direct_v2_calibrated_holdout/final_summary.json
```

## Train only

```bash
JOB_ID=cnn_connected_subword_direct_v2 \
DATA_DIR="$PWD/DataSet/Synthetic_Arabic" \
NUM_GPUS=2 \
EFFECTIVE_GLOBAL_BATCH_SIZE=128 \
EPOCHS=35 \
bash scripts/train/run_connected_subword_direct_synthetic.sh
```

## Queue evaluation manually after an existing training job

```bash
TRAIN_JOB_ID=<training-job-id>

WEIGHTS="$PWD/Weights/cnn_connected_subword_direct_v2/model_latest.pth" \
DATA_DIR="$PWD/DataSet/Synthetic_Arabic" \
RUN_TAG=cnn_connected_subword_direct_v2_calibrated_holdout \
DEPENDENCY_JOB_ID="$TRAIN_JOB_ID" \
THRESHOLDS=0.60,0.70,0.80,0.85,0.90 \
CALIBRATION_START_INDEX=1 \
CALIBRATION_SAMPLES=100 \
HOLDOUT_START_INDEX=101 \
HOLDOUT_SAMPLES=100 \
bash Evaluation/evaluate_connected_subword_direct_synthetic.sh
```

## Default objective settings

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

`ZERO_SHOT_PROFILE` must remain disabled in this mode because its geometric
transformations would invalidate fixed interval coordinates.

## Real-data continuation

```bash
JOB_ID=cnn_connected_subword_real \
SYNTHETIC_WEIGHTS="$PWD/Weights/cnn_connected_subword_direct_v2/model_latest.pth" \
bash scripts/train/run_connected_subword_real.sh
```

Real continuation still uses Span-DTW because the current real-data manifest does
not contain trusted connected-subword boxes.
