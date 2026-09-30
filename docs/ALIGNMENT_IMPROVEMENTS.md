# Sequence negatives and visual localization

Implemented on the existing `refactor/simple-alignment-core` branch. No branches,
trained checkpoint files, optimizer defaults, crop policy, CNN/Transformer/fusion
architecture, character codebook, or SIGReg mathematics were changed.

## Training controls

The **single enable switch** is `--negative-dtw-weight`: 0 disables generation and
negative alignment; positive DTW remains identical. Positive and negative strings
are complete transcripts for the **same image**, not guessed per-window labels.
For each valid candidate, the loss is `relu(margin + D_positive - D_negative)`.
Both costs differentiate; hinges are averaged within each line, then across lines
with eligible negatives. DDP weights these counts independently of positive-line
counts, and reduces loss sums, ranking counts/margins and corruption diagnostics.
Lower sequence cost is better; ties are not correct rankings.

| CLI control | Default | Meaning |
|---|---:|---|
| `--negative-dtw-weight` | 0 | Enable with a positive weight; no separate boolean |
| `--negative-count` | 3 | Requested candidates per line; fewer may be returned |
| `--negative-margin` | .20 | Cost-ranking margin |
| `--negative-operations` | `substitute,adjacent,blocks,words,shift,shuffle` | Comma-separated mixture |
| `--negative-severity` | .35 | Fraction edited; local segments capped at 8 characters |
| `--negative-seed` | 42 | Stable SHA256 seed with sample identity and epoch |
| `--negative-warmup-epochs` | 0 | Positive-only training epochs before ranking |
| `--negative-curriculum-epochs` | 5 | Early larger substitutions → configured mixture/severity; 0 disables progression |

Every candidate uses the loss's `clean_letters` NFKC/Arabic-letter filtering.
Length is preserved. Bounded retries reject duplicates, identical normalized
strings, unsupported/empty strings, provided `equivalent_transcripts`, and strings
with the same collapsed repeated-letter sequence. This last rejection is deliberately
conservative for stretch-equivalent targets. Shortfalls and operation/rejection
counts are logged. No global Python hash or worker RNG controls text corruption.
Positive source text is never edited. Image augmentation is unchanged.

Training negatives use training records only and vary by epoch. Validation
negatives use epoch 0 with no curriculum and are fixed across epochs, including
warmup (validation measures the configured full objective). Test data are not
mined. Epoch statistics include `ranked_lines`, `ranking_candidates`,
`ranking_accuracy`, `negative_minus_positive_cost`, `corruption`, and
`skip_reasons`. All-infeasible global training batches raise before an optimizer
step. No-support ranking metrics are unavailable, not fabricated accuracy.

The inspected epoch-20 checkpoint `Weights/real_20260924_222148/checkpoint_best.pt`
uses **full_alphabet_nll**, temperature .1, gamma .05, vertical .05, horizontal
.30, position prior .15, positive weight 1, negative weight 0, SIGReg weight 0.
These are printed from the resolved config. Full-alphabet softmax is preserved
before gathering transcript columns. Historical DTW checkpoints retain their
Arabic inventory plus per-transcript extension for uncommon letters. To use an
explicit fixed training inventory, supply unique normalized `--alphabet-inventory`;
unknown targets are then reported as unsupported, not silently remapped. Negative
generation without an explicit inventory uses the fixed `ARABIC_LETTERS` set and
reports unsupported positives instead of inventing substitutions for them.

### Exact launcher commands (new runs; illustrative weights, not calibrated)

Run from the repository root. Choose unused job names; existing run directories
are rejected. The actual launcher is `scripts/train/train.sbatch`, with its
existing `rtx4090` partition and two `rtx_4090` GPUs. It already forwards CLI args.

```bash
# Unchanged positive-only experiment, negative generation completely OFF.
sbatch --job-name=alignment_positive_20261001a scripts/train/train.sbatch \
  --dataset DataSet/ArabicDataset --dataset-type real \
  --positive-dtw-weight 1 --sigreg-weight 0 --negative-dtw-weight 0

# Same settings and split seed, with sequence negatives (example weight .1).
sbatch --job-name=alignment_negative_20261001a scripts/train/train.sbatch \
  --dataset DataSet/ArabicDataset --dataset-type real \
  --positive-dtw-weight 1 --sigreg-weight 0 --negative-dtw-weight 0.1 \
  --negative-count 3 --negative-margin 0.2 \
  --negative-operations substitute,adjacent,blocks,words,shift,shuffle \
  --negative-severity 0.35 --negative-seed 42 \
  --negative-warmup-epochs 2 --negative-curriculum-epochs 5
```

Both retain batch 32, epochs 20, Adam LR 2e-5, weight decay 0, seed 42,
128×1024 images, 32-wide windows/16 stride, ResNet18, five-layer one-head tiny
Transformer, and the launcher's existing fusion settings. Loss settings are
examples for a controlled experiment, **not recommendations inferred from a gain**.

`--resume CHECKPOINT` requires the same configuration except total epochs and
restores optimizer plus saved per-rank RNG/loader states. Older checkpoints lacking
those states load strictly but warn that continuation is not bit-exact. Same
hardware/software/world size is still necessary for exact continuation.
`--finetune CHECKPOINT` starts a new run/optimizer with the saved splits and
permits explicit loss changes only; it stores the source checkpoint/config rather
than rewriting them. For this real checkpoint also pass `--dataset-type real`.

## Independent alignment experiments

DTW and position prior retain their existing defaults. `--position-prior 0` is an
independent ablation. `--alignment-objective ctc --position-prior 0` selects a
separate, complete-transcript CTC baseline, **not DTW with free skips**. It uses
full-alphabet logits from the same cosine codebook/temperature plus a constant
blank logit (`--ctc-blank-logit 0`). It checks `T >= L + adjacent_equal_label_pairs`;
impossible targets are reported. CTC is summed then divided by transcript length,
so its magnitude is **not comparable to DTW's T+L normalization**. CTC never allows
an empty/all-skip explanation for a nonempty target. Its emission spikes are not
word boundaries. The historical stats key `positive_dtw` holds the selected
positive sequence objective; `alignment_objective` identifies which one.

Do not combine negative training, CTC and position ablations for the first test:
change one thing at a time. Neither CTC nor negative training is enabled by default.

## Dimensions and padding audit

`D` is embedding dimension, `T` image windows, `L` transcript letters. Image-text
costs are `[T,L]`, image-image scores `[T_A,T_B]`, alphabet logits `[T,K]`.
Embedding dimension mismatches raise; sequences are never made square. Training
uses per-line strings and cost matrices, not padded transcript tensors. The
standalone optional `encode_batch` still returns a PAD mask; it is not used in
the sequence-loss loop.

Current image preprocessing directly resizes the real XML/bbox crop to the saved
H×W: **no batch image padding**. Real internal white spaces are retained. Collation
rejects unequal tensor sizes instead of adding hidden pixels. The model's optional
`valid_widths` and physical `token_valid` API supports explicitly padded callers:
only fully valid windows enter the CNN (including BatchNorm statistics); invalid
keys are masked in the Transformer; losses and evaluation filter valid vectors.
Partial boundary windows are excluded, so up to a window-minus-one tail pixels
may be unrepresented in this explicit-width mode. Invalid physical IDs are -1.
Features, masks and physical indices are reversed together for RTL. No mask is
allowed to reinterpret real white space as artificial padding or as a CTC blank.
There is no new variable-width batching/preprocessing policy.

## Shared-region decoding and letter evidence

All public inference paths use `evaluate.match_features` with implementation
version `local-stretch-1`. Cosines are unmodified and saved separately. Default
cosine rewards remain `C - max(.6,row_median+.05,column_median+.05)`; raw
`C-threshold` remains available. Median correction may suppress broad true matches.

The new local decoder permits bounded repeats (default maximum 3 windows per
same opposite window) and affine **true gaps**, with local starts/ends. A diagonal
adds full reward; a repeat adds half reward for one new window minus .05. Repeat
direction cannot flip without a diagonal; no cell can be revisited. Gap costs
are opening .2 and extension .05. Candidate endpoints are kept before greedy
noncrossing selection. Scores equal the sum of recorded traceback deltas and
evidence minus repeat/gap penalties. Endpoint budgeting and greedy selection
remain approximate; `candidate_limit_reached` exposes truncation.

Default support is 3 distinct positively matched physical windows per side,
configurable down/up via `--min-windows`. This is not a minimum word length.
Weak diagonal bridges of at most `max_gap=1` may fill between anchors. True skips
are not filled; distant groups are split; unsupported leading/trailing spans are
trimmed. Overlapping physical window footprints may touch even when IDs differ.
Masks are full-height on original source images using inverse crop/resize geometry.
`--decoder affine` retains the former cosine-only one-to-one matcher for comparison.

Letter evidence computes
`logsumexp_c(logpA[i,c] + logpB[j,c] - log(pi[c])) - acceptance_offset`.
It retains soft uncertainty with the saved character vectors and temperature.
New runs save a vocabulary-ordered add-one-smoothed **training-only** prior.
Older checkpoints explicitly use a uniform prior over configured inventory or
`ARABIC_LETTERS`. Priors are normalized, floored (default 1e-4), and renormalized;
flooring occurs before float32 conversion. Blank is excluded without renormalizing
away its mass. No evaluation transcript, mask, OCR, Quran lookup, or argmax label
enters image-only prediction. Default offset .1 is uncalibrated and, in the small
diagnostic below, far too permissive. This score is not a probability.

## Notebook controls and exports

Open `notebooks/model_evaluation.ipynb`. The **first code cell** controls paths,
`SPLIT` (now validation by default, fallback OFF), `NUM_SAMPLES`, `RANDOM_SEED`,
`REPRESENTATION`, `SIMILARITY_MODE`, `DECODER`, thresholds, support, repeats and
gaps. Section **7** compares cosine/evidence on identical deduplicated saved-val
pairs without rerunning the CNN/Transformer. Section **8** shows native negatives
and candidate-specific paired crops/score breakdowns. Section **9** optionally
reports fixed validation transcript ranking using checkpoint settings.

For independently checked pairs, use `ANNOTATION_MANIFEST` rows with the existing
A/B image paths and `alignment_mask_meta.manually_verified_label` set to `aligned`
or `unaligned`. These labels affect evaluation selection/scoring only, never the
matcher. The aggregate report separates verified-negative counts/FPR/coverage
from unreviewed manifest-negative metrics; absent verified labels stay unavailable.

For parameter changes use `session.settings.update(SETTINGS)` rather than rebuilding
the session. Cache keys include checkpoint identity, prior, representation, all
scoring/mask controls, and implementation version. Feature caching is independent.
Population evaluation deduplicates IDs, not notebook slots. Existing notebook
outputs are preserved and labeled archived; rerun to see current results.

Exports contain raw cosine/reward/evidence arrays, alphabet log probabilities,
vocabulary/prior, geometry, valid/logical/physical indices, accepted paths,
candidate acceptance/rejection breakdowns, exact settings and checkpoint SHA.
Unique output creation is retained; the duplicate session construction was removed.
Normal full-line figures retain masks/overlays; candidate crops are separate
diagnostic plots. True skips are dashed gray, weak evidence orange, repeats blue,
and strong matches green.

## Measured checks (CPU, 2026-10-01)

Checkpoint SHA256:
`397e36722cbfdac8319f1850a5a0fa7ddc5521498ffef538c8bc6a231dd62acc`.
Saved split: train 1698, validation 176, test 198 lines. Validation paired catalog:
4 manifest positives and 41 manifest negatives. **No independently hand-checked
negative annotations were available**. All localization masks used below were
automatically derived page-box/text-LCS masks, not character-level ground truth.

Measured two positives (`pair_000028:455`, `pair_000028:1580`) and two manifest
negatives (`pair_000028:172`, `pair_000028:183`). Same eight cached line features:

| Mode | Positive mean IoU | Positive mean Dice | Negative mean mask coverage | Negative any-region rate |
|---|---:|---:|---:|---:|
| Cosine + stretch | .4403 | .5956 | .1599 | 2/2 |
| Letter evidence + stretch, uniform prior | .3658 | .5352 | .6935 | 2/2 |

This is a tiny, single-page-pair diagnostic, **not a measured improvement or a
calibrated false-positive benchmark**. Letter evidence is visibly over-permissive
at the starting offset. Some manifest negatives have text-LCS overlap; labels need
manual review. Files/figures: `/tmp/alignment-comparison-kwrbzmp2/` (temporary).

Then ran one real training line and one fixed validation line, starting each mode
from identical checkpoint weights/RNG and the same Adam LR, with separate fresh
optimizers. No original checkpoint was overwritten. This is an execution smoke,
not positive-versus-negative efficacy evidence:

| Mode | Online positive cost | Validation positive cost | Validation ranking |
|---|---:|---:|---|
| Positive DTW only | 1.09146 | .640887 | disabled |
| DTW + .1 negative margin, 2 candidates | 1.09146 | .640887 | 2/2, mean Dneg-Dpos .60583 |
| DTW prior=0 only | 1.08620 | .636595 | disabled |
| Separate CTC | 3.03420 | 1.72158 | disabled |

Both training negatives already exceeded the margin (mean difference 1.21624), so
their hinge was zero; unchanged positive results are expected, not a gain.
All modes had finite backward flow into CNN and Transformer. Temporary smoke
checkpoints: `/tmp/alignment-real-smoke-maz1odpa/`.

Remaining experiments: full controlled training runs, calibrated validation sweep
with independently checked negatives, full held-out test evaluation after freezing
settings, and GPU/DDP runtime verification (CUDA unavailable here). CPU unit tests
cover corruption/normalization/equivalence/worker independence, ranking signs and
gradients, disabled/warmup paths, rectangular costs, padding exclusion, RTL and
boundary windows, stretched/weak/gapped/crossing/no-match decodes, score reconstruction,
GT independence, cache invalidation, checkpoint compatibility, and exact CPU resume.

Validation: **96 tests passed, 8 subtests passed** in the project
`manucripts_align` environment. A real two-process CPU/Gloo reduction test agrees
with the single-population gradient, including unequal positive/ranked counts.
The sandbox disallows the IPC sockets required by DataLoader/Gloo; these tests
passed after running with approved local IPC access. Python compilation, shell
syntax, notebook cell syntax, and `git diff --check` passed. The actual notebook
also executed headlessly on the real checkpoint (1 positive, 1 manifest negative,
4-pair comparison cap; no outputs/checkpoints overwritten).
