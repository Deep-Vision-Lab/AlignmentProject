# Image-only shared regions between two Arabic manuscript lines

Public entry point: `python -m Evaluation.eval_shared_regions --help`.
This is an evaluation-only diagnostic, not a new training objective. It does not
read transcripts, predict characters, or use annotation masks to construct a
match. Native XML is used only by the existing deterministic image crop.

## Runtime contract

The existing strict checkpoint loader reconstructs the architecture from
`model_config`; the text model is not instantiated. The default representation
is normalized fused features, with `--representation local` and `context` as
diagnostic alternatives. The compact checkpoint is ResNet18 -> 128-D local ->
5-layer/1-head no-position Transformer -> 256-to-128 fusion. Model evaluation
disables dropout and freezes BatchNorm running statistics; no backward or
optimizer step occurs.

The existing checkpoint contract and Point-2 prepared-image encoder supply
preprocessing, tensorization, valid masks, and physical indices. For the current
checkpoint: strict XML crop with checkpoint margins -> grayscale -> direct
bilinear 1024x128 -> normalization `(0.449, 0.226)`, 32-pixel full-height windows,
stride 16, 63 valid windows, logical RTL order. There is no second crop,
binarization, white canvas, or augmentation.

The raw matrix is the dot product of explicitly L2-normalized selected vectors.
`cosine.npy` preserves that float32 matrix without clipping, centering, or
thresholding. Its heatmap always uses [-1, 1]. Rows are line 1 and columns line 2;
tick labels are logical indices, which increase in RTL reading order for Arabic.
Invalid/fully artificial-padding windows are excluded; `summary.json` preserves
the mapping from matrix rows/columns to logical and physical windows.

## Alignment and support

Affine-gap **local Smith-Waterman** allows both lines to have unmatched starts
and ends. Ordinary Needleman-Wunsch is global; ordinary subsequence DTW matches
a complete query against a portion of the other line. Neither directly expresses
two partially shared lines in the same way.

The separate `smith_waterman_affine` API in `sw_core.py` leaves the historical
constant-gap evaluator unchanged. Diagonal steps receive a reward; a run of k
gap steps costs `gap_open + (k-1)*gap_extend`. Local scores restart at zero.
There is **no image-to-transcript DTW position prior** in this matcher.

Starting settings (not calibrated):

| Option | Default | Meaning |
|---|---:|---|
| `--score-mode` | `background` | Row/column median correction |
| `--cosine-threshold` | 0.60 | Absolute cosine floor |
| `--contrast-margin` | 0.05 | Required excess over both medians |
| `--gap-open` | 0.20 | First gap window cost |
| `--gap-extend` | 0.05 | Each additional gap window cost |
| `--min-windows` | 5 | Distinct positive physical anchors on **each** side; 4 is supported |
| `--max-internal-gap` | 1 | Maximum valid but unsupported physical positions between anchors |
| `--min-region-score` | 0 | Trimmed region objective must exceed this value |
| `--max-candidates` | 128 | Explicit greedy search cap, reported if reached |

Background reward is
`C[i,j] - max(threshold, median(C[i,:])+margin, median(C[:,j])+margin)`.
`--score-mode raw` instead uses `C - threshold`. Rewards are saved separately as
`alignment_scores.npy/csv`; they are not cosine similarities or probabilities.
Background correction rejects a uniformly high matrix, but can suppress broad
or repeated genuine matches. Raw mode can accept unrelated high-background
features. Neither mode is an accuracy guarantee.

Each traceback is split at large unsupported physical gaps. Each piece must
independently satisfy support on both sides and a positive trimmed objective
including its actual gap costs. Only positive diagonal correspondences count;
gaps and nonpositive diagonals do not inflate support. Internal filling cannot
cross invalid/padding positions. `--strict-consecutive` sets the gap allowance
to zero. Unsupported leading/trailing positions are never filled.

Repeated extraction is greedy. Accepted row/column spans and crossing quadrants
are excluded, preventing window reuse and crossing regions. Rejected positive
cells are suppressed to allow other candidates to be examined. This procedure
can miss a better joint set of regions; it is **not globally optimal**.

Masks union actual supported window footprints, plus explicitly permitted tiny
internal fills, mapped with the shared inverse crop/resize/padding geometry.
Rasterization rounds interval starts down and ends up. White spans the complete
**original source image height**; black is everything else. No distant interval
is filled just because it lies inside a larger traceback. Distinct regions keep
separate IDs and colors. Because stride 16 windows overlap at width 32, their
source footprints can touch even across separate IDs; binary masks cannot
resolve gaps smaller than this footprint resolution.

Five windows is an evaluation support criterion, not a universal minimum word
length. One-to-one diagonal matching plus gaps does not model arbitrary duration
differences. If visual review demonstrates width/duration failures, a later
local-DTW/common-subsequence comparison is reasonable; no such second algorithm
is silently mixed into this evaluator.

## Commands

Run from the repository root, with the project Python environment active:

```bash
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PRETRAINED_LOCAL_ONLY=1
WEIGHTS=Weights/resnet18_128d_5l_1h_no_pos_21634461/model_best_validation_dtw.pth
MEMBERSHIP=Results/Monitoring/resnet18_128d_5l_1h_no_pos_21634461/split_manifest.json

# Validation: both zero-based indices refer to val_eval in the SAVED manifest.
# 0.15 is an exploratory setting, not a validated operating threshold.
python -m Evaluation.eval_shared_regions --weights "$WEIGHTS" \
  --manifest "$MEMBERSHIP" --split validation --record-indices 14 4 \
  --cosine-threshold 0.15 --device cpu

# Training pair: in-sample diagnostic, not used to select thresholds.
python -m Evaluation.eval_shared_regions --weights "$WEIGHTS" \
  --manifest "$MEMBERSHIP" --split train --record-indices 673 401 \
  --cosine-threshold 0.15 --device cpu

# Two explicit native images, with the SAME checkpoint preprocessing.
python -m Evaluation.eval_shared_regions --weights "$WEIGHTS" \
  --image1 DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000004/A/linesImages/line_01.png \
  --image2 DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000003/B/linesImages/line_05.png \
  --cosine-threshold 0.15 --device cpu

# A validation negative proxy; label affects reporting ONLY, never matching.
python -m Evaluation.eval_shared_regions --weights "$WEIGHTS" \
  --manifest "$MEMBERSHIP" --split validation --record-indices 14 0 \
  --cosine-threshold 0.15 --pair-label negative \
  --label-source 'dataset_manifest_full_pairs:no_shared_content:unverified_proxy' --device cpu

# Same evaluator on one RTX4090, using the existing cluster resource names.
mkdir -p out
sbatch scripts/eval_shared_regions_1x4090.sbatch --weights "$WEIGHTS" \
  --manifest "$MEMBERSHIP" --split validation --record-indices 14 4 \
  --cosine-threshold 0.15
```

For external images without native XML, explicitly add
`--preprocessing no-xml`. This disables XML cropping only; the checkpoint's
intensity, resize, normalization, and runtime policies remain in force. The
override and both expected/effective preprocessing contracts are recorded.
It is **not** claimed equivalent to native training preprocessing. Without this
explicit override, a missing required XML crop fails loudly.

Saved-membership selection checks the exact file SHA256 against checkpoint
metadata when present and uses the existing source/parent/page overlap checks.
It never regenerates a split or falls back to test. The manifest contains root
paths; missing files fail rather than silently relocating the population.
Explicit-image selection records the paths but does not claim split membership.

To score independently drawn source-size masks, add
`--gt-mask1 /path/to/mask1.png --gt-mask2 /path/to/mask2.png`.
Nonzero pixels are foreground. Prediction completes before those masks are read.
Wrong dimensions fail; annotations are never resized. Missing annotations are
`unavailable`, not zero accuracy. Pixel and column IoU/precision/recall/F1,
region center/boundary errors, and existing IoU-plus-consecutive-window support
are reported. The reused region matcher selects the best prediction per GT
region, not a one-to-one instance AP metric. Geometric overlap support is named
separately from the positive embedding anchors. No result is character accuracy.
Negative pair labels or all-black GT additionally report false-positive mask
coverage. Their provenance must be checked independently.

## Outputs and calibration

Without `--output-dir`, every call creates a new timestamp/UUID directory under
`Results/Evaluation/SharedRegions/`. An explicit existing directory is rejected.
Start with `overview.png`, then the full-size `line1_overlay.png` and
`line2_overlay.png`, then `cosine_heatmap.png` and `alignment_score_heatmap.png`.
Also saved: original PNGs, exact model-input PNGs, source-size masks, raw/reward
NPY and CSV matrices, positive-correspondence CSV (logical/physical/source
coordinates), and `summary.json` (checkpoint hash, geometry, membership, settings,
complete region paths, support/fill counts, rejection reasons, metrics, caveats).
Empty predictions still produce all artifacts, black masks, and an explicit
no-match overview. Images and matrices are actual model inference, not generated
visual examples.

Calibration must use **validation only**, including independently verified
shared and unrelated pairs. Sweep threshold/margin/gaps with fixed support,
record positive localization against manual masks and negative false-positive
coverage, then freeze settings before any explicit final test evaluation.
Do not optimize visual smoothness or merely the fraction of pairs with a path.
Repeated handwriting, letters, and background can create false matches.

During implementation, a small validation-only sensitivity pilot uses four
`high_match` and four `no_shared_content` manifest proxy labels, all from
`pair_000004`. Four negatives reuse one line, so observations are correlated.
There are no manual source-size shared-region masks. The saved pilot is useful
for checking score sensitivity, **not sufficient for scientific calibration**;
no final threshold or alignment-accuracy claim is selected from it. The default
0.60 is deliberately retained as an uncalibrated starting parameter, even though
it returns empty predictions on the initially sampled trained-model pairs.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider \
  tests/test_shared_regions.py tests/test_sw_dense_region.py \
  tests/test_sw_real_evaluation.py tests/test_evaluation_checkpoint_contract.py \
  tests/test_cropped_1024_evaluation_geometry.py \
  tests/test_wide_side_padding_evaluation_geometry.py
bash -n scripts/eval_shared_regions_1x4090.sbatch
```

The new tests cover affine scores against independent tiny exhaustive search,
off-diagonal/empty/short/multiple/crossing/reused matches, high uniform background,
small/large/invalid gaps, RTL/packed/padded coordinate mapping, full-height source
masks, GT geometry rejection and prediction independence, membership/hash checks,
no-overwrite behavior, and real checkpoint feature repeatability with unchanged
parameters/BatchNorm buffers. The real-assets check skips explicitly if its
local checkpoint or saved manifest is missing; it never downloads anything.

## Observed local verification, 2026-09-24

Repository HEAD: `1485f71f8eac9e322fb9beb84d26646a4fab8342`, branch
`current-arch-model`. Existing cleanup edits and monitoring results were left
alone. Checkpoint: job 21634461 `model_best_validation_dtw.pth`, epoch 30, SHA256
`2f4c50082594bbcc1a5c32184a408ccd283e8dade029c9f42d0a5d58b0aae04e`.
Saved membership SHA256:
`f6e3cf471a91b81f9098173c1057cbd2bf9e927a9b156619e6673b0a23a01042`.
CPU inference succeeded; no CUDA device was available and no Slurm job was
submitted. The targeted suite above passed **58 tests**, including 24 new tests.

All sampled real inputs produced 63 valid vectors per side with 128 coordinates.
The real-assets test exercised all three representations twice, with exact
repeatability and unchanged parameter and BatchNorm state. Explicit-image and
saved-manifest invocations on the same pair saved identical cosine matrices.

Outputs are new directories under `Results/Evaluation/SharedRegions/`:

- `check_20260924_validation_t015`: validation indices 14/4, one five-anchor
  region, 8.77%/8.98% mask coverage. This is an unconfirmed candidate, not proof of
  matching text. Source dimensions are 1368x218 and 1625x308.
- `check_20260924_train_t015`: training indices 673/401, no accepted region.
- `check_20260924_explicit_raw_t015`: explicit native image paths, raw scoring,
  same validation pair and same cosine matrix.
- `check_20260924_*_background`: default-threshold 0.60 checks, empty predictions.
- `pilot_20260924_val_I_J`: eight validation-only cases at threshold 0.15.
- `validation_pilot_20260924.json`: all 112 pair/configuration measurements for
  two score modes and seven thresholds, from those actual trained-model matrices.

Pilot results below are **pair detections**, not region accuracy or calibrated
precision/recall. The negative coverage is the mean across all eight negative
line masks, including empty masks. Labels are unverified manifest proxies.

| Score mode | Threshold | Positive-label pairs detected / 4 | Negative-label pairs detected / 4 | Negative mask coverage |
|---|---:|---:|---:|---:|
| background | 0.10 | 4 | 4 | 17.68% |
| raw | 0.10 | 4 | 4 | 29.08% |
| either | 0.15 | 4 | 2 | 5.17% |
| either | 0.20 | 2 | 0 | 0% |
| either | 0.60 | 0 | 0 | 0% |

Background correction reduces false-positive extent here at threshold 0.10 but
does not eliminate negative detections. Threshold 0.20 reduces detections on
positive-label pairs as well. Empty masks at 0.60 are not evidence of a useful
operating point. Independent manual masks and a broader, independently reviewed
validation cohort are still needed before claiming credible localization.

### Existing training evidence (read only)

`Results/Monitoring/resnet18_128d_5l_1h_no_pos_21634461/history.csv` and
`history.json` contain clean end-of-epoch evaluations through epoch 30. Fixed
evaluation gamma is 0.05; competition temperature 0.1; DTW position prior 0.15;
positive weight 1; SIGReg weight 0.2; negatives inactive. The image-image matcher
does **not** inherit that DTW prior. DTW evaluates 811/832 training lines and
277/287 validation lines (21/10 skipped empty cleaned transcripts, zero invalid).
SIGReg uses all valid vectors in its fixed seeded batches, including lines with
empty transcripts; it is a population statistic, not a per-line additive loss.

| Epoch | Clean train DTW | Validation DTW | Validation minus train | Train SIGReg raw / weighted | Validation SIGReg raw / weighted |
|---|---:|---:|---:|---:|---:|
| 1 | 2.21000 | 2.25291 | 0.04291 | 235.29152 / 47.05830 | 230.16524 / 46.03305 |
| 6 | 2.18648 | 2.22609 | 0.03961 | 52.45217 / 10.49043 | 48.22420 / 9.64484 |
| 30 | 1.99554 | 2.03456 | 0.03902 | 50.30703 / 10.06141 | 45.82490 / 9.16498 |

The early online total drop is mostly the weighted SIGReg component:
50.31304 -> 1.72820 between epochs 1 and 6; online DTW is 2.19006 -> 2.19896.
The online pass is stochastic and uses changing weights; it is not the clean
generalization comparison. Clean DTW later improves on both splits with a small
positive gap, which does not demonstrate correct image-image alignment.

At epoch 30, image-pair mask metrics are unavailable: 342 eligible training and
33 validation pairs, **zero annotated pairs**. The separate XML PartOfWord
image-text diagnostic has 45 training lines/827 regions and 26 validation
lines/476 regions. Mean interval IoU is 0.48250/0.49249; F1 0.60020/0.61187;
center error 24.46/25.05 source pixels; boundary error 31.18/32.06 pixels;
IoU-plus-support success 0.30351/0.29832 (train/validation). These are eligible
text-fragment diagnostics, not character accuracy and not ground truth for this
new image-only matcher. They cannot replace the missing pair masks.
