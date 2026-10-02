# Image-image evaluator implementation report

Historical first revision. The current implementation and fresh checkpoint
results are documented in [the whole-route review](path_quality_evaluation_report.md).

The trained architecture and production checkpoints were not modified. This change
fixes evaluation and the existing `notebooks/model_evaluation.ipynb`.

## 1. Files modified

- `evaluate.py`: central dispatch, local DP, physical segmentation, validation,
  diagnostics, shared route plotting and CLI settings.
- `evaluation_utils.py`: shared cached prediction, session delegation, diagnostic
  printing, metrics and visualization.
- `notebooks/model_evaluation.ipynb`: one configuration and prediction flow,
  aligned/unaligned samples, population evaluation, validation-only sweeps.
- `tests/test_local_repeat_evaluation.py`: 14 new evaluation regression tests.
- `tests/test_local_repeat_dtw.py`: explicit small support settings for repeat
  tests; updated rejection reason and allowed one-window fill expectation.
- `tests/test_evaluation_notebook.py`: validation-split tuning and shared-matcher
  consistency, retaining independent sample/cache checks.
- `docs/evaluation_pipeline_fix.md`: this report.

Existing notebook sample slots, optional recognition diagnostics and historical
outputs are retained. Historical output metadata is marked, and the notebook
explains that those outputs must be rerun to reflect the fixed implementation.
The existing opt-in split fallback remains visibly labelled as non-test evaluation.

## 2. Functions modified or added

Modified: `resolved_match_settings`, `match_features`, `_local_repeat_path`,
`local_repeat_regions`, standalone `evaluate_pair`, CLI `main`,
`EvaluationSession.predict_pair`, `compute_pair_metrics`, `plot_pair_alignment`,
and `plot_representation_comparison`.

Added: `mutual_top_k_anchors`, `_repeat_region_candidate`, `plot_match_routes`,
`predict_cached_pair`, and `print_candidate_diagnostics`.

Notebook matching, result caching and plotting now call repository functions.

## 3. Bugs found

- Local repeat acceptance ignored `min_windows` and checked smaller distinct
  support and pair minima instead.
- `max_gap` controlled fills but did not segment local repeat tracebacks.
- Weak short paths could generate secondary masks without region diagnostics.
- Positive notebook samples bypassed the central matcher, while session/negative/
  population evaluation used a separate configuration and stretch decoder.
- Every positive cell could restart the local DP.
- Reusing a cached result for a newly selected sample slot did not restore GT
  metadata to the new slot; the unified notebook flow exposed this cache bug.
- A split segment with a weak leading cell could lose its supported suffix under
  the strong-start rule. It now restarts at its first eligible anchor.
- Long visualization metrics could run past the figure; the panel now fits them.

## 4. Traced flow and exact fixes

The flow before modification was:

`token_valid-filtered model features -> cosine -> threshold/background rewards ->
local repeat DP -> whole traceback -> distinct support checks -> physical support
and fill masks`.

The notebook computed that path directly; session evaluation used `match_features`
and its default stretch decoder. GT scoring occurred outside the matcher, but the
notebook loaded sample GT before path selection.

The flow now is:

`valid features + valid physical identities -> match_features -> rectangular
cosine -> selected rewards -> eligible start anchors -> local repeat DP -> physical
anchor gap segmentation -> independent segment scoring/validation -> noncrossing
accepted regions -> supported windows + allowed valid fills -> source masks -> GT
loading/scoring/display`.

- For each segment, required A/B support and matched pairs are respectively
  `max(min_windows, min_distinct_windows_a/b)` and
  `max(min_windows, min_matched_pairs)`. Failure is
  `insufficient_distinct_support`, with actual and required counts.
- Consecutive matched anchors split when `abs(physical_next - physical_prev) - 1`
  exceeds `max_gap` on either line. Repeats have zero gap. Each segment is scored
  independently, resetting its start transition and excluding inter-segment costs.
- Both affine skip directions can follow each other, allowing unsupported windows
  on both lines to be represented as actual gaps, without calling them matches.
- Optional starts require mutual top-K reward ranking and positive reward, with an
  optional minimum starting reward. Rankings stay fixed when greedy suppression
  removes regions. Ordinary positive continuation cells need not be start anchors.
- When a split begins without a qualifying start, the original segment remains a
  rejected diagnostic. Its suffix can restart at the first eligible anchor, and
  must independently meet every support and objective requirement.
- Diagnostics separate matching rewards, affine gap penalties, increasing repeat
  run penalties, and final score. `start_anchor_mutual_top_k` and
  `start_anchor_eligible` separately report ranking and minimum-start-reward status.
- Mandatory filters are support, matched pairs, allowed physical gaps and positive
  objective. Enabled optional filters add explicit rejection reasons.
- Secondary-score ratios are also checked against the final best accepted region,
  so a stronger later region cannot invalidate the denominator silently.
- Fills can cover only small internal physical holes present in the valid physical
  map. They do not increase support counts or confidence. The full bounding span
  is never used as mask evidence. With `max_gap=1`, an explicitly allowed valid
  one-window skip can be filled; an invalid or large gap cannot be filled.
- `match_features` can filter full tokens using `token_valid_a/b`. It validates
  physical identities and rejects unfiltered zero vectors before cosine
  computation. Matrix dimensions are valid A windows × valid B windows.
- Session, cached positive samples, negatives, population, masks, metrics and
  exports share the same entry point. All resolved settings enter notebook and
  session cache identities; implementation version is `local-repeat-2`.
- Heatmaps retain their original matrices and share route overlays: main/secondary
  region widths, R/X labels, starts/ends, strong anchors, repeats and true gaps.

Density is the **minimum**, over A and B, of distinct supported physical windows
 divided by inclusive physical span. Fills do not count. Repeat fraction is repeat
transitions divided by matched pairs. True gaps count unsupported physical windows
between successive matched anchors across both lines, consistent with skipped
window counts rather than counting only gap runs.

## 5. Parameters and initial configuration

The notebook uses the requested initial values: threshold 0.50, background scoring,
contrast 0.05, all three support minima 5, physical gap limit 1, gap open/extend
0.20/0.05, repeat penalty 0.05, maximum consecutive repeats 3, strong starts enabled
and start top-K 3. It prints every resolved setting, including alignment mode,
similarity mode and decoder. Confidence filters remain disabled.

New central settings:

| Parameter | Initial notebook value |
| --- | --- |
| `alignment_mode` | `local_repeat_dtw` |
| `use_strong_start_anchor` | `True` |
| `start_top_k` | `3` |
| `start_min_reward` | `None` |
| `min_region_score` | `None` |
| `min_mean_reward` | `None` |
| `min_path_density` | `None` |
| `max_repeat_fraction` | `None` |
| `min_secondary_score_ratio` | `None` |

Existing repeat and distinct-support parameters are now exposed through
`match_features` and the CLI. `MAX_REJECTED_PATHS_TO_PLOT=5` limits visualization.
Legacy stretch/affine modes remain selectable for explicit comparisons and
existing callers. An explicitly selected alignment mode determines the decoder;
the notebook does not silently fall back to stretch. Parameter sweeps and algorithm
comparisons require the saved validation split.

## 6. Tests added

The new tests cover all nine requested cases plus independent objective accounting,
optional filters, start eligibility after splitting, first-strong-anchor trimming,
empty valid inputs, invalid settings and standalone checkpoint evaluation. The
consistency test executes the notebook's actual cached rescoring function with a
controlled accepted route, then compares it to session and population predictions,
including pairs, scores, physical support and masks. Existing tests verify frozen
model state, GT independence, sample slots, cached threshold changes, representation
options, metrics and exported artifacts.

## 7. Test results

**53 passed, 5 deselected**, with all notebook code cells compiling and
`git diff --check` passing. Command:

```sh
MPLCONFIGDIR=/tmp/alignment_mpl /opt/anaconda3/envs/AlignProj/bin/python -m pytest \
  tests/test_local_repeat_dtw.py tests/test_local_repeat_evaluation.py \
  tests/test_evaluation_notebook.py tests/test_evaluate.py \
  tests/test_alignment_improvements.py -q \
  -k 'not checkpoint_loss and not workers_do_not and not two_rank and not generation_epoch and not exact_resume'
```

The selected suite excludes training-focused tests and unrelated multiprocessing
checks. An initial broader run passed 35 tests but encountered two unrelated
OpenMP shared-memory failures in spawned-worker/distributed training tests under
the sandbox. This is not a claim that the entire repository suite passes.

Four existing saved-validation examples were compared against `HEAD:evaluate.py`
using the same frozen `real_sum_nogate_hybridneg` checkpoint and identical cosine
matrices. Two provided manifest negatives and the four-pair population entry point
were also evaluated. Session and cached predictions matched exactly. Plots were
rendered and visually checked. Results are in:

`Results/Evaluation/pipeline_fix_review/report.json`, per-pair candidate JSON,
raw score/physical-map NPZ files, and PNG figures.

## 8. Before/after short paths and retained text

| Saved validation pair | Old accepted support | New accepted support |
| --- | --- | --- |
| `pair_000028:1580` | `[3,3]`, `[20,16]` | `[8,7]`, `[5,5]` |
| `pair_000028:3638` | `[3,3]`, `[9,11]` | none |
| `pair_000028:455` | `[7,6]`, `[11,13]` | none |
| `pair_000028:1989` | `[5,5]`, `[10,9]` | `[9,9]` |

Sample 1 reproduces the reported old short-region score **0.478655** and old main
score **2.963787**. No accepted new region has support below 5. Short `[3,3]`
candidates are rejected for insufficient distinct support. Sample 1 retains two
independently supported segments at scores **1.965754** and **1.462855**; the other
retained example scores **3.236052**, has 15 matched pairs and no internal gap.
The former long accumulated spans are not assumed to be valid continuous regions.

## 9. Remaining concerns

- Two inspected positives have no accepted region with the initial settings;
  confidence and recall require a broader validation study.
- One of two inspected manifest negatives still yields `[7,6]` support. The code
  fixes logical acceptance and evidence handling; it does not calibrate accuracy.
- Greedy region selection is not globally optimal, and top-K alone is not an
  absolute confidence threshold. Optional confidence filters are intentionally
  disabled pending validation.
- LCS/box GT is automatically derived. No GT, transcript, label or alignment
  annotation is supplied to reward computation, route selection or masks.
- Existing notebook output remains historical until cells are rerun. Full-dataset
  population metrics and exhaustive notebook execution were not run.
