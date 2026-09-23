# Compact no-position architecture experiment

This is a **combined architecture experiment**, not an attribution ablation.
An improvement cannot identify which individual architectural change caused it.
The default legacy architecture remains available. `base-vit-arch` is untouched.

## Executed graph

```
XML four-sided crop -> grayscale original intensities -> scan augmentation (online only)
-> direct bilinear 1024x128 resize -> gray normalization (0.449, 0.226)
-> 63 explicit 128x32 windows, stride 16, physically extracted LTR
-> trainable ImageNet ResNet18 pooled 512
-> Linear(512,128) -> Dropout(local_dropout=0.10) -> reverse tokens for Arabic RTL
-> L [B,T,128]
-> bidirectional Transformer(128, 5 layers, 1 head, FFN512, GELU, pre-LN, final LN)
-> C [B,T,128]
concat(L,C) [B,T,256] -> Linear(256,128) -> one final LayerNorm -> h
h -> L2 normalization -> z -> existing full-alphabet NLL / Soft-DTW
h[valid] -> existing SIGReg
```

No local projection/wrapper LayerNorm, no input/fusion dropout, and no
Transformer positions, relative biases, RoPE, coordinate embeddings, or causal
mask. The same dropout realization of L feeds both branches. Transformer
dropout remains separately configurable (`TINY_VIT_DROPOUT`, default 0.0).
`LOCAL_DROPOUT=0` is an explicit comparison. Dropout does not reduce dimension;
the learned linear projection does. Its starting probability is not an optimum.

Without positions, the attention block has no explicit order/distance signal.
Output indices, the unchanged Arabic ordering, monotonic DTW and its separate
position prior still carry sequence constraints. Repeated letters alone do
not establish that positions are harmful.

The grayscale model has 12,260,672 trainable visual parameters:

| Component | Parameters |
|---|---:|
| ResNet18 backbone (gray conv1, no classifier) | 11,170,240 |
| Local linear projection | 65,664 |
| Transformer, including final LN | 991,616 |
| Single fusion linear + final LN | 33,152 |
| Frozen text codebook + normalization | 524,544 (0 trainable) |

The new Transformer is **fresh**, not pretrained DeiT-Tiny. Its layers are
independently initialized. Cached ImageNet backbone initialization, including
the existing RGB-filter mean for gray conv1, is retained. No download is made.
Full-model `--weights` and optimizer resume are rejected for this variant.
Evaluation reconstructs the explicit variant and checks dimensions strictly.
Legacy metadata defaults to the legacy graph, never this experiment. Saved
codebook state, deterministic seed/mapping and state hash preserve identity.

## Objective (default new launcher)

`L = 1.0 * positive_letter_DTW(z) + 0.20 * SIGReg(h[valid])`.
Negative transcripts are inactive. No new loss was added. Full-alphabet
competition precedes transcript-column gathering. Training recurrence,
normalization by T+L, penalties, gamma schedule, codebook policy and SIGReg
mathematics are unchanged. The launcher retains Adam, LR 2e-5, scan augmentation
and the reviewed coefficients; explicit environment overrides are recorded.
Gamma start/end 0.05, vertical penalty 0.05, horizontal penalty 0.30,
DTW prior 0.15 and competition temperature 0.10 are defaults, not inferred from
the Transformer position setting.

## Monitoring contract

Each completed epoch has three distinct records:

- `train_online`: changing weights, training dropout, scan augmentation.
- `train_eval`: final epoch weights, full actual training membership, deterministic.
- `val_eval`: the same weights/settings, full actual validation membership.

Real online repeat/scan wrappers and the epoch sampler are not monitoring
populations. Original loader membership is saved in `split_manifest.json` and
hashed into checkpoints. Known source-line, page/pair and augmentation-parent
identities must be disjoint; failures stop instead of reshuffling. Pre-rendered
augmentation records are retained and identified in saved provenance, not
silently called pristine originals. Unsupported datasets without recoverable
record identities fail clearly.

Both clean passes use FP32, `eval()`, no backward/updates, frozen BN, and a
fixed reference gamma (default configured final gamma; `MONITOR_GAMMA` override).
Modes and Python/NumPy/Torch RNG states are restored. Rank zero evaluates the
unwrapped model, with SIGReg's distributed statistics explicitly disabled;
results and failures are broadcast to all ranks. Evaluation does not use a
padded distributed sampler. No test images are inferred or selected routinely.
Legacy subprocess test-preview hooks are bypassed when monitoring is enabled.

Per-line DTW is count-weighted over eligible lines (both pair sides when
present), with empty/invalid reasons and counts. Online DTW reporting is also
line-count weighted, while its optimized total remains labeled as the
batch-weighted stochastic optimization objective. SIGReg is not a per-line
quantity: evaluation uses fixed seeded sketches on fixed-order batches, then
token-count weights the batch statistics. Batch size and the partial last
batch are recorded. Do not compare its magnitude across different population
policies as though it were an additive line loss.

The main generalization gap is `val_eval_DTW - train_eval_DTW`, not validation
minus online loss. A training decrease with worsening validation suggests
overfitting; poor results on both can have many causes. One smoke epoch and
raw DTW magnitude do not diagnose model quality or establish localization.

## Spatial evidence

Fixed previews use saved train/validation IDs and include the source image,
exact model input, own transcript, training cell/effective matrices, and hard
path CSV with logical/physical/source coordinates. They are separate from
full-split metrics. No scaled Soft-DTW gradients are advertised as probabilities.

Image-pair masks use the existing pair evaluator and spatial metric core.
Both source-line identities must belong to the evaluated split. Consecutive
window support uses actual model windows mapped through inverse geometry,
never an unrelated source-pixel 32/16 grid. IoU and support are evaluated
together. Missing masks are unavailable, not accuracy zero.

XML text-fragment boxes are a separate diagnostic. They are eligible only
when concatenated cleaned RTL box labels exactly equal the full own-line
transcript. Mismatches are unavailable, not fuzzy-matched labels. Multi-letter
fragment boxes are not character-level GT. Eligible intervals report IoU,
precision/recall/F1, center/boundary error, support and counts. A crop box or
an image-image overlap score is not proof of individual letter correctness.

## Commands

Run from the repository root in the existing environment. Names must be new;
the launcher refuses to overwrite a weights directory. No historical launcher
or checkpoint is replaced.

Short CPU smoke (two optimization batches total, explicitly labeled subsets):

```bash
JOB_NAME=resnet18_128d_5l_1h_no_pos_smoke_NEW \
RESTORATION_EPOCHS=2 EXPERIMENT_BATCH_SIZE=1 EXPERIMENT_SMOKE_BATCHES=1 \
MONITOR_MAX_RECORDS=5 MONITOR_BATCH_SIZE=2 MONITOR_PREVIEWS=3 \
DATALOADER_NUM_WORKERS=0 OMP_NUM_THREADS=2 \
bash scripts/train_resnet18_128d_5l_1h_no_pos.sh
```

Full run, with full train/validation evaluation after every epoch:

```bash
JOB_NAME=resnet18_128d_5l_1h_no_pos_full_NEW \
NPROC=2 RESTORATION_EPOCHS=30 EXPERIMENT_SMOKE_BATCHES=0 \
MONITOR_MAX_RECORDS=0 MONITOR_BATCH_SIZE=16 MONITOR_PREVIEWS=3 \
bash scripts/train_resnet18_128d_5l_1h_no_pos.sh
```

Or submit the dedicated launcher (not submitted by implementation):

```bash
sbatch scripts/train_resnet18_128d_5l_1h_no_pos_2x4090.sbatch
```

Standalone checkpoint re-evaluation of full **saved** train/validation membership:

```bash
python scripts/evaluate_epoch_checkpoint.py \
  --weights Weights/RUN/model_best_validation_dtw.pth \
  --manifest Results/Monitoring/RUN/split_manifest.json \
  --output Results/Monitoring/RUN/standalone_full.json --device auto
```

Explicit final held-out test diagnostic, separately invoked after review:

```bash
python -m Evaluation.eval_point3_hard_paths \
  --dataset DataSet/ArabicDataset --weights Weights/RUN/model_best_validation_dtw.pth \
  --split test --image-preprocessing training --n-samples 10 --device cpu \
  --output-dir Results/Evaluation/RUN/final_test_point3
```

## Artifacts

`Results/Monitoring/RUN/` contains CSV/JSON history, complete and selected ID
manifests, preview IDs, separate clean-DTW/total/raw-SIGReg/weighted-SIGReg/gap
curves, and epoch/split previews and available spatial results/curves.
`MONITOR_OUTPUT` may specify a new output directory explicitly.

`Weights/RUN/` contains `model_latest.pth`, per-epoch snapshots,
`model_best_validation_dtw.pth`, and the separately named
`model_best_validation_total.pth`. The latter is not the selection criterion
for the primary best checkpoint. Periodic/final optimizer snapshots remain
`checkpoint_latest.pth`. Subset-selected best weights are explicitly marked as
subset-selected; do not call them full-validation selections.

Useful checks:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider \
  tests/test_compact_experiment.py tests/test_evaluation_checkpoint_contract.py \
  tests/test_resnet18_packed_windows.py tests/test_restoration_positive_dtw.py
RUN_GLOO_MONITOR_TEST=1 OMP_NUM_THREADS=2 python -m pytest -q \
  -p no:cacheprovider tests/test_compact_experiment.py -k two_rank
```
