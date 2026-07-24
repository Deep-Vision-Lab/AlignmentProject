# Training Speed Optimization

Branch: `agent/training-speed-optimization`  
Base: `multi_gpu_ddp`

This branch keeps the original `train.py` and launcher available. The optimized
path uses:

```bash
bash scripts/train/run_span_d3tw_optimized.sh
```

## Phase status

- [x] Phase 0 — reproducible baseline/optimized benchmark scripts
- [x] Phase 1 — one authoritative safe span configuration
- [x] Phase 2 — CUDA/NVTX component profiling and JSON reports
- [x] Phase 3 — transcript/path preload, scratch staging, spawn DataLoader workers
- [x] Phase 4 — one visual forward for both lines, larger CNN chunks, channels-last, TF32
- [x] Phase 5 — persistent frozen AraBERT surface cache and batched text encoding
- [x] Phase 6 — bucketed batched JAX Span-DTW and batched hardest-negative scoring
- [x] Phase 7 — one positive hard path per selected line reused by local/pair/order losses
- [x] Phase 8 — quality-preserving and scheduled faster auxiliary-loss modes
- [x] Phase 9 — coalesced text-gradient all-reduce, DDP static graph, fused Adam
- [x] Phase 10 — gradient accumulation with DDP `no_sync()`
- [x] Phase 11 — AMP retained, TF32/channels-last enabled, FP32 DP costs
- [x] Phase 12 — atomic/throttled checkpoints and lightweight W&B epoch logging
- [x] Phase 13 — regression tests for blank DP, batch DP, cache, config, accumulation
- [ ] Phase 14 — cluster performance numbers (requires an RTX 4090 SLURM run)
- [x] Phase 15 — isolated branch rollout; no direct changes to `multi_gpu_ddp`

Phase 14 is operational rather than a code change. The scripts are committed,
but throughput and GPU-memory numbers must be generated on the BGU cluster.

## Important semantic configuration

The optimized launcher enforces:

```text
MAX_TEXT_TOKEN_CHARS=2
MAX_TEXT_SPAN_CHARS=2
MAX_WINDOWS_PER_SPAN=3
SPAN_INCLUDE_SPACE_CONTEXT=0
SPAN_ALLOW_CHARACTER_SPACE_SURFACES=0
SPAN_USE_BLANK_TRANSITIONS=1
```

`<SPACE>` remains a real transcript character. `<BLANK>` consumes one image
window without advancing the transcript.

## Main speed changes

### Frozen Arabic text backbone

AraBERT output is cached by unique visible surface. Trainable projection,
`<SPACE>` and `<BLANK>` vectors are applied after cache retrieval, so gradients
remain valid. Core and context surfaces from all positive/negative texts are
batched and deduplicated.

### Batched JAX bridge

Equal-shaped dense transition tensors are grouped and sent through one DLPack
Torch→JAX transfer. Positive samples, no-gradient negative scoring and selected
hardest negatives use batched calls. The blank-aware recurrence is unchanged.

### Visual model

Both paired line batches are concatenated and passed through the visual model
once. The results are split afterward. This avoids two separate ResNet/BiLSTM
launch sequences per training batch.

### Positive path reuse

Selected positive hard paths are computed once per line and reused by:

- local hard-negative loss;
- image-image region construction;
- image-pair contrastive loss;
- order consistency loss.

Blank regions are excluded from character-region objectives.

### Effective batch 128

Default optimized settings are:

```text
microbatch per GPU = 32
GPU count          = 2
accumulation        = 2
effective batch    = 128
```

This is safer than allocating 64 samples on each 24 GB RTX 4090.

## Recommended first checks

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git checkout agent/training-speed-optimization
git pull origin agent/training-speed-optimization
conda activate manucripts_align

python -m py_compile \
  arabic_span_text_encoder.py \
  span_alignment_loss.py \
  jax_span_dtw.py \
  training_optimizations.py \
  scripts/train/train_optimized.py

pytest -q \
  tests/test_blank_span_dtw.py \
  tests/test_batched_span_dtw.py \
  tests/test_span_surface_cache.py \
  tests/test_optimized_training_config.py \
  tests/test_gradient_accumulation.py
```

## Submit the optimized training run

From scratch:

```bash
JOB_ID=synthetic_arabic_8k_optimized_gpu2 \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
bash scripts/train/run_span_d3tw_optimized.sh
```

From model-only pretrained weights:

```bash
PRETRAINED_WEIGHTS="$PWD/Weights/<existing-run>/model_latest.pth" \
JOB_ID=synthetic_arabic_8k_optimized_gpu2 \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
bash scripts/train/run_span_d3tw_optimized.sh
```

Do not resume an optimizer checkpoint that predates the trainable blank
parameter unless its optimizer state is known to be compatible.

## Quality-preserving versus faster scheduling

Default:

```bash
OPTIMIZATION_MODE=quality
```

This preserves the configured auxiliary-loss frequencies. The speedup comes
from batching, caching and reuse.

Optional faster mode:

```bash
OPTIMIZATION_MODE=fast \
IMAGE_PAIR_EVERY_N_BATCHES=2 \
bash scripts/train/run_span_d3tw_optimized.sh
```

Scheduled losses are multiplied by their interval to preserve their expected
average contribution. Validation must be compared before using this as the main
research configuration.

## Matched benchmark

```bash
bash scripts/benchmark/submit_training_speed_comparison.sh
```

After both jobs finish:

```bash
python scripts/benchmark/summarize_training_speed.py \
  --baseline-log out/align_base_<STAMP>_<JOBID>.out \
  --optimized-json logs/performance/benchmark_optimized_<STAMP>_epoch_001.json
```

The optimized run writes one JSON performance report per epoch under:

```text
logs/performance/
```

## Profiling

```bash
PROFILE_TRAINING=1 \
PROFILE_MAX_BATCHES=30 \
ENABLE_NVTX=1 \
USE_WANDB=0 \
EPOCHS=1 \
JOB_ID=optimized_profile \
bash scripts/train/run_span_d3tw_optimized.sh
```

Profiling synchronizes CUDA events around components and therefore should not be
used to estimate final raw throughput. Use it to locate bottlenecks, then run a
second benchmark with `PROFILE_TRAINING=0`.

## Useful tuning controls

```text
CNN_CHUNK_SIZE=1024
SPAN_BACKBONE_BATCH_SIZE=512
SPAN_FEATURE_CACHE_SIZE=8192
DATALOADER_NUM_WORKERS=2
DATALOADER_PREFETCH=2
USE_CHANNELS_LAST=1
ALLOW_TF32=1
USE_FUSED_ADAM=1
TORCH_COMPILE_VISUAL=0
FULL_CHECKPOINT_EVERY_N_EPOCHS=5
MODEL_WEIGHTS_EVERY_N_EPOCHS=2
```

`TORCH_COMPILE_VISUAL` remains disabled by default because PyTorch 2.0 compile
startup can be expensive and must be benchmarked on the cluster before becoming
a default.

## Acceptance checklist before merging

- [ ] Static compilation passes.
- [ ] Regression tests pass.
- [ ] One 30-batch profile completes with finite losses and gradients.
- [ ] No DLPack fallback is reported.
- [ ] JAX batched-call count is much smaller than aligned-item count.
- [ ] Surface cache hit rate rises during epoch 1.
- [ ] No CUDA OOM at microbatch 32, accumulation 2.
- [ ] Feature-concentration output retains correct `<SPACE>`/`<BLANK>` behavior.
- [ ] Validation loss is comparable to the `multi_gpu_ddp` baseline.
- [ ] Throughput improvement is measured before merging.
