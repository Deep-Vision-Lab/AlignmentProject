# Training Speed Optimization

Branch: `agent/training-speed-optimization`  
Base: `multi_gpu_ddp`  
Draft PR: `#5`

The original `train.py` and launcher remain unchanged and available. The
optimized path is opt-in:

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
- [x] Phase 13 — static checks and focused CPU regression tests
- [ ] Phase 14 — RTX 4090 cluster throughput, memory, and convergence measurements
- [x] Phase 15 — isolated branch and draft-PR rollout; `multi_gpu_ddp` is untouched

Phase 14 requires the BGU SLURM environment and two RTX 4090 GPUs. The scripts
needed to perform it are committed. No performance number should be claimed
before those jobs finish.

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
remain valid. Core and context surfaces from all positive and negative texts are
batched and deduplicated. The frozen surface cache remains valid across epochs.

### Batched JAX bridge

Equal-shaped dense transition tensors are grouped and sent through one DLPack
Torch-to-JAX transfer. Positive samples, no-gradient negative scoring, and the
selected hardest negatives use batched calls.

JAX bucket batch dimensions are padded to a stable multiple, controlled by:

```text
SPAN_DTW_BATCH_BUCKET_SIZE=8
```

This reduces recompilation when the number of samples in a text-length bucket
changes slightly between batches. Gradient-enabled batched calls now return
both per-sample costs and gradients from one recurrence evaluation.

### Fast hard decoder

The old hard path decoder copied one scalar cost from CUDA to CPU inside the
innermost DP loop, causing many device synchronizations. The optimized decoder:

1. constructs all transition costs on the GPU;
2. transfers each complete cost tensor once;
3. runs the discrete DP and backtrace on CPU/NumPy.

It preserves blank transitions and is regression-tested against the reference
decoder.

### Visual model

Both paired line batches are concatenated and passed through the visual model
once. The results are split afterward. This avoids two separate ResNet/BiLSTM
launch sequences per training batch.

Additional controls include:

```text
CNN_CHUNK_SIZE=1024
USE_CHANNELS_LAST=1
ALLOW_TF32=1
CUDNN_BENCHMARK=1
TORCH_COMPILE_VISUAL=0
```

`TORCH_COMPILE_VISUAL` remains disabled by default because PyTorch 2.0 compile
startup must be measured on the cluster first.

### Positive path reuse

Selected positive hard paths are computed once per line and reused by:

- local hard-negative loss;
- image-image region construction;
- image-pair contrastive loss;
- order consistency loss.

Blank regions are excluded from character-region objectives.

### Input pipeline

- transcript strings and file paths are prepared once;
- the dataset is staged to `$SLURM_SCRATCH_DIR` by default;
- JAX-compatible workers use the `spawn` multiprocessing context;
- pinned memory, prefetching, and non-blocking copies remain enabled.

### Effective batch 128

Default optimized settings are:

```text
microbatch per GPU = 32
GPU count          = 2
accumulation       = 2
effective batch    = 128
```

This is safer than allocating 64 samples on each 24 GB RTX 4090.

## Local checks

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject
git checkout agent/training-speed-optimization
git pull origin agent/training-speed-optimization
conda activate manucripts_align

python -m py_compile \
  Parameters.py \
  DataSet.py \
  arabic_span_text_encoder.py \
  span_alignment_loss.py \
  jax_span_dtw.py \
  jax_batch_bucketing.py \
  fast_hard_alignment.py \
  training_optimizations.py \
  scripts/train/train_optimized.py

PYTHONPATH="$PWD" pytest -q \
  tests/test_blank_span_dtw.py \
  tests/test_fast_hard_alignment.py \
  tests/test_batched_span_dtw.py \
  tests/test_span_surface_cache.py \
  tests/test_optimized_training_config.py \
  tests/test_gradient_accumulation.py
```

The draft PR also runs these checks through
`.github/workflows/optimization-checks.yml`.

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

The entrypoint resolves the offline AraBERT cache from these locations:

1. explicit `HF_HOME`;
2. `<project>/.hf_cache`;
3. `<project>_clone/.hf_cache`;
4. `~/.cache/huggingface`;
5. explicit `TRANSFORMERS_CACHE`.

## Quality-preserving versus faster scheduling

Default:

```bash
OPTIMIZATION_MODE=quality
```

This preserves configured auxiliary-loss frequencies. The speedup comes from
batching, caching, path reuse, fewer bridge calls, and fewer CUDA
synchronizations.

Optional faster mode:

```bash
OPTIMIZATION_MODE=fast \
IMAGE_PAIR_EVERY_N_BATCHES=2 \
bash scripts/train/run_span_d3tw_optimized.sh
```

Scheduled losses are multiplied by their interval to preserve their expected
average contribution. Validation must be compared before using this as the main
research configuration.

## Matched cluster benchmark

```bash
bash scripts/benchmark/submit_training_speed_comparison.sh
```

This submits matched baseline and optimized one-epoch runs. After both jobs
finish:

```bash
python scripts/benchmark/summarize_training_speed.py \
  --baseline-log out/align_base_<STAMP>_<JOBID>.out \
  --optimized-json logs/performance/benchmark_optimized_<STAMP>_epoch_001.json
```

The optimized run writes one JSON performance report per epoch under:

```text
logs/performance/
```

## Component profiling

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
used to estimate final raw throughput. Use it to identify bottlenecks, then run
a second measurement with `PROFILE_TRAINING=0`.

## Useful tuning controls

```text
CNN_CHUNK_SIZE=1024
SPAN_BACKBONE_BATCH_SIZE=512
SPAN_FEATURE_CACHE_SIZE=8192
SPAN_DTW_BATCH_BUCKET_SIZE=8
DATALOADER_NUM_WORKERS=2
DATALOADER_PREFETCH=2
DATALOADER_MP_CONTEXT=spawn
USE_CHANNELS_LAST=1
ALLOW_TF32=1
USE_FUSED_ADAM=1
TORCH_COMPILE_VISUAL=0
FULL_CHECKPOINT_EVERY_N_EPOCHS=5
MODEL_WEIGHTS_EVERY_N_EPOCHS=2
```

## Acceptance checklist before merging

- [x] Python and shell static checks pass.
- [x] Blank-transition tests pass.
- [x] Reference/fast hard-decoder equivalence tests pass.
- [x] Torch/JAX batched-DTW equivalence and gradient tests pass.
- [x] Frozen-surface cache and trainable-projection gradient tests pass.
- [x] Configuration guardrail tests pass.
- [x] Gradient-accumulation tests pass.
- [ ] One 30-batch RTX 4090 profile completes with finite losses and gradients.
- [ ] No DLPack fallback is reported on the cluster.
- [ ] JAX batched-call count is much smaller than aligned-item count.
- [ ] Surface cache hit rate rises during epoch 1.
- [ ] No CUDA OOM at microbatch 32, accumulation 2.
- [ ] Feature-concentration output retains correct `<SPACE>`/`<BLANK>` behavior.
- [ ] Validation loss is comparable to the `multi_gpu_ddp` baseline.
- [ ] Throughput improvement is measured before merging.
