# AlignmentProject — simple alignment core

A standalone PyTorch pipeline for weakly supervised Arabic manuscript alignment.
This refactor lives on `refactor/simple-alignment-core`; historical implementations
remain available in Git. It starts **new experiments**, not silent migrations of
old checkpoints, optimizer states, or held-out populations.

## Read the pipeline

| File | Responsibility |
| --- | --- |
| `dataset.py` | One synthetic/real/bridge dataset, XML crops, tensorization, scan augmentation |
| `dataloader.py` | Seeded group-safe train/validation/test views and batching |
| `cnn_encoder.py` | Simple CNN or trainable ImageNet ResNet18 window encoder |
| `transformer_encoder.py` | Independently initialized bidirectional window-token layers |
| `model.py` | Explicit windows → local → context → single-linear fusion → unit vectors |
| `text_embedding.py` | Frozen deterministic character directions and Arabic letter cleaning |
| `dtw.py` | Alphabet competition, Soft-DTW and hard monotonic path |
| `losses.py` | Positive DTW, optional explicit-negative margin and pre-L2 SIGReg |
| `parameters.py` | One configuration dataclass; no architecture environment variables |
| `train.py` | Training, validation every epoch, strict checkpoint loading/saving |
| `evaluate.py` | Selected-split loss or image-only shared regions |

Runtime dependencies: Python ≥3.10, PyTorch, matching torchvision, NumPy, Pillow,
and Matplotlib; pytest for tests. JAX, W&B and Hugging Face are not needed by the
new runtime. Historical environment/requirements files also contain dependencies
for preserved data-building tools; they are not a minimal runtime installation.

## Model and objective

Default input is grayscale `[B,1,128,1024]`, normalized by mean **0.449**, std
**0.226**. Native `linesImages/line_N` images receive the strict XML four-sided
crop, including 5%/minimum-2-pixel safety margins and the builder's vertical
padding correction. Direct bilinear resize follows; no artificial canvas,
heuristic second crop, inversion, or binarization. Non-native sources use their
explicit bbox, if supplied, otherwise the complete source. `--crop xml` requires
native geometry; `--crop none` explicitly bypasses it. `--no-grayscale` selects RGB
ImageNet normalization. Binarization is opt-in only.

Full-height 128×32 windows at stride 16 produce **63** tokens. The ResNet18
backbone produces 512 dimensions, followed by `Linear(512,128) → Dropout(.10)`.
Physical windows are encoded left-to-right, then reversed into Arabic RTL order.
The same local tensor feeds both context and fusion. Context has five pre-norm
GELU Transformer layers, one head, FFN 512, final LayerNorm, and no positions.
Fusion is `Linear(256,128) → LayerNorm → L2`. `fused_pre_l2` is exposed separately
for SIGReg. Other CNN/preset choices are explicit CLI options, not backend patches.
No-position attention has no explicit distance/order signal; DTW still uses
ordered indices and its separate position prior.

The default objective is `positive_DTW + 0.20 * SIGReg`. Negative margin weight is
zero. The negative-loss function accepts explicit per-line negatives; the training
CLI deliberately rejects a nonzero negative weight because it has no negative
transcript source. No image–image loss is added.

DTW retains full-alphabet NLL before gathering transcript columns, temperature
0.10, gamma 0.05, vertical penalty 0.05, horizontal penalty 0.30, position prior
0.15, horizontal suppression when windows ≥ letters, and `T+L` normalization.
NFKC Arabic-letter cleaning drops whitespace, marks, digits, punctuation and
tatweel from **DTW supervision**, not from stored transcripts. The codebook retains
the seed-1234 Unicode mapping and normalized fixed-LayerNorm directions. Space has
a valid vector; PAD is zero. Vectors are approximately, not exactly, orthogonal.

SIGReg uses valid **pre-L2** fused vectors and the preserved Gaussian-weighted
characteristic-function statistic. Training DDP uses global moments/shared random
directions. Validation runs on rank zero with collectives disabled and seeded
sketches. Reported SIGReg is a token-weighted batch-population statistic, so its
value depends on batch policy; it is not an additive per-line metric. DTW averages
use actual valid-line counts, including both sides of paired samples. Empty
transcripts are counted as skipped, not zero-DTW observations.

## Data and splits

`--dataset-type auto|synthetic|real|real_synthetic` selects parsing, not another
Dataset class. Synthetic uses `images/img{1,2}_N.png` and `texts/text{1,2}_N.txt`.
Real uses `dataset_manifest.jsonl` with A/B sides. Independent mode flattens unique
image/text **associations**; paired mode keeps both sides. Native own-side
transcripts are authoritative; generic explicit filenames need not share stems.
Copied-page annotations are retained and grouped together using source-page hashes.
For paired real data, connected page components keep every related page in one split.
Bridge V3 uses `dataset_manifest.jsonl` or `anchor_index.jsonl`; each item preserves
its real/positive-synthetic images, transcripts, optional positive mask, and
`anchor_id`. Negative bridge rows are outside this positive relationship view.

Defaults are 80/10/10, seed 42. For 60/20/20 use
`--train-ratio .6 --val-ratio .2 --test-ratio .2`. Ratios apply to **groups**, so
line percentages can differ. `random` still keeps repeated group IDs atomic;
`predefined` requires manifest `split` fields and rejects crossing groups. Group
leakage is an error. Splits are saved and replayed from checkpoint IDs; evaluation
does not regenerate them. This manifest population differs from the historical
all-page census and must not be presented as the same held-out population.

Only training uses geometry-preserving brightness, contrast, blur, Gaussian noise
and sparse speckle augmentation. Validation/test are deterministic. Pre-rendered
augmented source images remain augmented; disabling transforms does not undo them.

## Training

Run from the repository root in the activated PyTorch environment:

```bash
python train.py --dataset DataSet/ArabicDataset --run-name experiment1

torchrun --standalone --nproc_per_node=2 train.py \
  --dataset DataSet/ArabicDataset --run-name experiment2

python train.py --dataset DataSet/Synthetic63 --dataset-type synthetic \
  --run-name synthetic_test
```

Pretrained ResNet18 loads **cached weights only**, from `Pretrained/torch/checkpoints`
or the local Torch cache. Nothing is downloaded. Use `--no-cnn-pretrained` only for
an explicitly random-backbone experiment. Transformer/projection/fusion are fresh.

Short CPU smoke run (explicit subset, not scientific validation):

```bash
OMP_NUM_THREADS=2 python train.py --dataset DataSet/ArabicDataset \
  --run-name smoke_NEW --epochs 1 --batch-size 1 --num-workers 0 \
  --max-batches 1 --device cpu
```

Every completed epoch prints **Train loss** and **Validation loss**. No test images
are iterated in training. CUDA uses BF16 autocast for the visual forward; DTW and
SIGReg remain float32. CPU uses float32. Rank-zero validation is full-split without
DistributedSampler duplication. Online DDP training uses standard sampler padding
when needed; online totals are not clean end-of-epoch train-evaluation metrics.

Outputs are `Weights/<run_name>/checkpoint_latest.pt`, `checkpoint_best.pt`,
`history.json`, and `split_manifest.json`. Best means **validation total objective**,
recorded explicitly in metadata, not the old best-validation-DTW criterion. A smoke
cap labels both passes and checkpoint selection as subset measurements. Existing run
directories are rejected unless an explicit compatible `--resume` checkpoint is used.
New-format checkpoints save model, frozen text state, optimizer, full config, epoch,
best validation score, split IDs/hash, metrics and initialization provenance. Old
`.pth` checkpoints fail clearly rather than loading partially.

## Evaluation

```bash
python evaluate.py --checkpoint Weights/experiment1/checkpoint_best.pt \
  --dataset DataSet/ArabicDataset --split test

python evaluate.py --checkpoint Weights/experiment1/checkpoint_best.pt \
  --dataset DataSet/ArabicDataset --split val --mode shared_regions \
  --record-indices 0 1 --output Results/val_pair_NEW

python evaluate.py --checkpoint Weights/experiment1/checkpoint_best.pt \
  --mode shared_regions --image-a /path/to/line_a.png --image-b /path/to/line_b.png \
  --output Results/explicit_pair_NEW
```

Use `--split train` for in-sample loss/pair diagnostics and `--split val` for
calibration. Explicit images use the checkpoint's grayscale/resize/crop policy;
`--crop-override none` records a deliberate full-source ablation. The evaluator
loads the same standalone model strictly and runs eval/no-grad.

Shared regions use local affine Smith–Waterman: both images may have unmatched
beginnings/ends. Default rewards are cosine minus the maximum of 0.6, row median
+0.05, and column median +0.05. `--score-mode raw` uses cosine−threshold instead.
Gap opening/extension defaults are 0.2/0.05. Five distinct positively matched
windows on each side are required; `--min-windows 4` is an explicit alternative.
At most one missing window between supported anchors is filled; `--max-gap 0`
requires strict consecutive support. Separate regions remain separate. Greedy
extraction prevents crossing/reuse but is not globally optimal. Background
correction can suppress broad/repeated true matches; all settings are uncalibrated.

Inspect `cosine_heatmap.png` (fixed −1…1 scale), `alignment_scores_heatmap.png`
(separate transformed rewards), both full-width originals/model inputs,
`line_a_mask.png`, `line_b_mask.png`, and overlays. Masks are source-sized,
full-height, with white predicted shared regions. Physical window coordinates are
inverted through the crop/resize geometry; 32 model pixels are not assumed to be
32 source pixels. `cosine.npy`, `alignment_scores.npy`, `correspondences.csv`, and
`metadata.json` retain numerical evidence and rejection reasons. Empty predictions
are valid. `--gt-a`/`--gt-b` source-size masks are scored only after predictions;
geometry mismatches fail, and missing annotations are unavailable, not zero accuracy.
Region overlap does not establish character-level alignment correctness.

## Slurm and verification

```bash
sbatch scripts/train.sbatch --dataset DataSet/ArabicDataset --run-name cluster_NEW
sbatch scripts/evaluate.sbatch --checkpoint Weights/cluster_NEW/checkpoint_best.pt \
  --dataset DataSet/ArabicDataset --split val

OMP_NUM_THREADS=2 python -m pytest -q
python -m compileall -q .
```

The thin wrappers use the existing `rtx4090`/`rtx_4090` resource names, two GPUs
for torchrun training and one for evaluation. They contain no architecture logic.
`scripts/data/` is preserved unchanged; some historical builders already refer to
unavailable legacy utilities and are not part of the new runtime/test contract.
Other historical documents and environment files are archival, not current commands.
Pytest discovery is confined to `tests/`, not ignored datasets or vendored libraries.
On this checkout, recursive compile-all also traverses large ignored dataset/cache
trees, including a Python 3.13 environment that Python 3.10 cannot compile.
`python -m compileall -q *.py tests scripts` is a focused source-only check; a failure
in that external environment is not a successful full-tree compilation.
