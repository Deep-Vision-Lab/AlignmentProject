# Whole-route alignment evaluator review

The evaluator and notebook were revised without training, changing the network,
or writing production weights. The frozen checkpoint was
`Weights/real_sum_nogate_hybridneg/checkpoint_best.pt`, SHA256
`5e658c7e23c0c17badcbec2b9a58948613b2dcc04c0c74c10d461be4e07f53b8`.
This report supersedes the first evaluator report, `evaluation_pipeline_fix.md`.

## 1. Files changed

- `alignment_candidates.py`: new image-only candidate statistics, validation,
  trimming, extension, dominance, selection, and merging.
- `evaluate.py`: central dispatch, competing DP tracebacks, checkpoint letter
  verification, physical assertions, route plots, CLI controls and crop boxes.
- `evaluation_utils.py`: session/cached prediction, post-prediction GT loading,
  local evaluation categories, category metrics, unique sampling, diagnostics.
- `notebooks/model_evaluation.ipynb`: initial settings, shared cached matching,
  unique sample queue, shared negative/population evaluation, refreshed outputs.
- `scripts/eval/review_path_quality.py`: reproducible frozen-checkpoint review,
  before/after localization, anchor ablation, and attributed notebook outputs.
- `tests/test_path_quality_evaluation.py`: 20 new regression tests.
- Existing local-repeat and notebook tests: updated for whole-route acceptance
  and explicit legacy comparisons; this report and the historical report link.

## 2. Functions changed

Core prediction: `resolved_match_settings`, `match_features`,
`_local_repeat_path`, `local_repeat_regions`, `checkpoint_letter_evidence`,
`predict_cached_pair`, `EvaluationSession.predict_pair`, `evaluate_pair`.
Candidate handling: `uniqueness_matrices`, `segment_path`,
`candidate_statistics`, `validate_candidate`, `extend_path`,
`extend_selected_regions`, `select_candidates`, `merge_regions`,
`decode_candidates`. Evaluation/display: `EvaluationSession.evaluate_pair`,
`load_pair_ground_truth`, `boxes_ground_truth`, `evaluation_category`,
`compute_pair_metrics`, `aggregate_summary`, `sample_random_pairs`,
`print_candidate_diagnostics`, `plot_match_routes`, `plot_pair_alignment`.
The notebook's `_rescore_cached` delegates to the shared prediction wrapper.

## 3. Bugs confirmed

The previous decoder rejected the positive 9×8 and 6×7 candidates because their
first cells were not mutual top-K, and rejected strong 4×4 routes under the
fixed five-window rule. Selection suppressed evidence before alternative routes
were fully evaluated. The notebook repeated its positive pool, making Sample 5
a duplicate. Negative-pair evaluation did not resolve the available fallback
local GT, despite shared subwords. Larger candidates could also win on length
despite weaker normalized evidence; pairwise dominance tolerance could compound
through a chain of increasingly weak supersets.

## 4. Exact solutions

| Requested fix | Implemented behavior |
| --- | --- |
| 1: first-cell gate | Any positive cell can start. Whole-route anchor count/fraction and positions determine reliability. |
| 2: trimming | Original, prefix, suffix and both-end trims compete. Trimmed cells and provenance are retained. |
| 3: adaptive support | Normal and short classes have separate support/evidence checks; short routes cannot bypass stronger filters. |
| 4: uniqueness | Competing row/column maxima exclude self, including ties and non-top-1 cells. Missing competitors produce `None` diagnostics. |
| 5: gaps | Tracebacks split on physical discontinuities before validation/masks. Invalid identities are excluded from skipped-window counts/density; large spatial holes still separate masks. Filled windows never become matched support. |
| 6: extension | Both ends can continue with positive incremental objective, bounded weak runs/repeats/gaps and monotonic geometry. Selected-core extension is revalidated and cannot overlap another selected region. |
| 7: merging | Ordered disjoint routes require small physical gaps, actual positive interior evidence and valid combined statistics. |
| 8: selection | Competing tracebacks and versions are validated before overlap selection; comparable supersets dominate contained routes. Supersets are processed first, preventing compounded tolerance. |
| 9: normalization | Full reward/penalty decomposition and normalized evidence are stored. Remaining competitors rank by `score_per_match × sqrt(min(support_A, support_B)) × density`, then normalized confidence; raw score is last. |
| 10: letter verification | Cosine proposals are verified on their own cells using fixed checkpoint alphabet distributions/prior. Configured thresholds can reject weak evidence. |
| 11: interpretation | Diagnostics distinguish visual candidates, accepted alignment candidates and rejected visual candidates. Acceptance does not establish correctness. |
| 12–13: local GT/metrics | After prediction, classify positive, partial overlap, true no overlap or unknown; report localization separately. |
| 14: sampling | Deduplicate, cap at availability, exhaust once, never wrap; existing slots survive exhaustion. |
| 15: one pipeline | Notebook/session/population/CLI/masks use `match_features`; plots read its stored routes. |
| 16: unequal lengths | Filter `token_valid` before rectangular cosine; assert valid physical identities and reading order. |
| 17: diagnostics | Every actual candidate stores the complete schema, including unavailable fields as `None`; all records are exported. |
| 18: heatmaps | Shared overlays label accepted A0… and top rejected R0…, anchors, endpoints, repeats, physical gaps, extensions and trims. GT masks require an explicit display option. |

The ranking utility is a deterministic support/confidence tradeoff, not a
calibrated probability. Larger comparable continuations win through dominance;
length alone cannot defeat substantially stronger evidence. Extension uses its
own continuation rules so a moderate tail need not beat its core's mean score.

## 5. New parameters and initial values

The notebook uses the requested initial configuration. Cosine threshold remains
**0.50**, background margin **0.05**, gaps **0.20/0.05**, repeat penalty **0.05**,
repeat run cap **3**, physical internal gap **1**. No threshold search was run.

| Controls | Values |
| --- | --- |
| `normal_min_distinct_a/b`, `normal_min_matched_pairs` | 5 / 5 / 5 |
| `allow_short_regions`, `short_min_distinct_a/b`, `short_min_matched_pairs` | True; 3 / 3 / 4 |
| `short_min_mean_cosine`, `short_min_median_cosine`, `short_min_mean_reward` | .72 / .72 / .15 |
| `short_min_path_density`, `short_max_internal_gap` | .80 / 0 |
| `use_path_anchor_check`, `anchor_top_k` | True / 3 |
| Long anchor count/fraction | 2 / .15 |
| Short anchor count/fraction, including short-class controls | 2 / .40 |
| `enable_path_extension`, minimum reward, weak-run cap, gap cap | True / 0 / 2 / 1 |
| `enable_region_merging`, `merge_max_gap_a/b` | True / 1 / 1 |
| `dominance_min_quality_ratio` | .90 |
| `candidate_similarity_mode`, notebook `verify_with_letter_evidence` | cosine / True |
| Mean-letter/positive-letter-fraction filters | None / None |
| Median bidirectional-margin filters, long/short | None / None |
| Score/mean reward/density/repeat/secondary confidence filters | None |
| Candidate capacity / rejected routes plotted | 512 / 5 |

Capacity limits raw competing tracebacks; trim/extension records can make the
final diagnostic pool larger than 512. None of the nine review examples reached
the raw capacity limit. Public matcher/CLI defaults now select local repeat-DTW
and cosine .50. Letter verification is opt-in in the generic API (which may have
no text encoder); the notebook enables it. Every setting is serialized and
included in matching cache keys.

## 6. Removed/deprecated parameters

`use_strong_start_anchor`, `start_top_k`, `start_min_reward` remain parseable for
old configurations but have **no prediction effect**, including the private DP
argument. The notebook no longer exposes them. Its `MIN_REGION_WINDOWS` control
is replaced by explicit normal/short settings. Legacy `min_windows` and distinct
support aliases remain for explicit legacy comparisons or adaptive minima set
to `None`; they cannot override the explicit normal minima. Affine and stretch
decoders remain available through explicit selection.

## 7. Tests added

The 20 new tests cover all 15 requested scenarios: weak first cell; trimming;
compact 4×4/6-match route; weak 3×3 rejection; ambiguous high cosine; letter
verification; weaker tails; large-gap splitting; one-window gap; dominance;
distant regions; rectangular padding exclusion; RTL identities; actual shared
notebook/session/population/CLI/mask parity; capped unique sampling. Additional
tests cover post-prediction categories/metrics, actual bridge evidence,
default adaptive settings, weak larger competitors, and dominance chains.

The entry-point parity test uses a frozen fixture model and deterministic
features, calls actual notebook helper source, runs population evaluation,
`evaluate_pair` and CLI argument parsing/saved-record selection, and compares
candidates, rejected routes and saved mask pixels.
Label/annotation changes cannot change prediction. No optimizer runs.

## 8. Test results

**73 passed, 5 deselected**, two dependency warnings, 5.66 seconds. The suite
includes `test_path_quality_evaluation`, `test_local_repeat_dtw`,
`test_local_repeat_evaluation`, `test_evaluation_notebook`, `test_evaluate`, and
`test_alignment_improvements`. Existing training/resume/multiprocessing tests
were deselected to honor the no-training scope. Notebook code compiles; the CLI
help and diff whitespace checks pass.

## 9. Results on each problematic notebook sample

Frozen-checkpoint **CPU**, saved **validation** membership. Previous evaluator
and new evaluator use identical extracted features/cosine. Before/after IoU
uses the same post-prediction box/LCS masks. Support is distinct matched windows,
not filled mask width. These annotations are automatically derived, not manual
character-level verification.

| Example / sample suffix | Previous accepted support | New accepted support | Mean A/B IoU before → after |
| --- | --- | --- | --- |
| Positive 1 / 1580 | 8×7; 5×5 | 9×8; 6×5 | .494 → .505 |
| Positive 2 / 3638 | none | 5×3; 3×3, both short | .000 → .265 |
| Positive 3 / 455 | none | 3×3 short; 7×7 normal | .000 → .369 |
| Positive 4 / 1989 | 9×9 | 10×9 | .671 → .683 |
| Manifest negative 1 / 981 | none | 3×4 short | .000 → .000 |
| Manifest negative 2 / 1887 | 6×5; 5×5 | 6×6; 5×5 | .116 → .110 |
| Shared-word negative / 1487 | none | 5×4 short | .000 → .112 |
| Manifest negative 4 / 2496 | none | 4×3 short | .000 → .243 |
| Manifest negative 5 / 2437 | 6×5 | 6×5 | .352 → .352 |

Artifacts: `Results/Evaluation/path_quality_review_v3/report.json`, nine
prediction figures, per-example complete candidate JSON, cosine/reward/letter
matrices, prediction masks and separate GT masks. The notebook imports those
real review outputs with explicit provenance and cleared execution counts; it
does not claim a full Run All execution. Sample 5 has no duplicate figure.

## 10. Missed positives recovered

Positive 1's formerly rejected **9×8** continuation is accepted, with a second
**6×5** route. Positive 3's rejected **6×7** pair set is contained in the accepted
**7×7** route. Positive 2 now has short detections, but the highlighted full
**4×4, six-match, cosine .804** route is **not recovered**: it has **zero** mutual
top-3 anchors. Another 4×4 route has two anchors in six matches (**.333**), below
the requested **.40**. Adaptive support removes the old five-window failure,
but the new requested anchor filter still excludes those complete routes.

An explicitly labelled anchor-disabled ablation holds all other settings fixed
and accepts a 4×4 route at A48–51/B32–35. This is diagnostic evidence, not a
deployment setting change; it does not prove recovery of the highlighted
A38–41/B32–35 route. No GT or test result was used to relax the filter.

## 11. Previous false-positive regions removed

**No complete previously accepted false-positive mask was eliminated** under
the requested initial thresholds. The difficult negative-2 route remains
accepted, with mean cosine **.812**, median bidirectional margin **−.054**,
mean letter evidence **1.740**, positive letter fraction **1.0**. Its high
cosine and positive letter evidence do not establish textual correctness.
Negative 5's previous mask remains. Negative 1 now has a short detection with
zero localization IoU, a remaining error. Optional margin/letter rejection is
implemented and tested, but setting thresholds to `None` deliberately does not
filter these ambiguous routes. Claiming they were removed would be incorrect.

## 12. Positive Sample 4 stability

The original aligned core remains covered. The selected route is **10×9**,
A53–62/B18–26, compared with previous A54–62/B18–26. Mean IoU improves
**.6712 → .6833**; mean precision **.8518**, recall **.7722**. This preserves the
successful localized region; it is not a claim of complete line alignment.

## 13. Duplicate sampling

Fixed. Requesting five with four eligible unique positives warns and displays
four; exhaustion does not replace a slot or refill the queue. Negative sampling
also deduplicates and caps. The refreshed notebook has four positive plots,
no fifth plot, and five distinct reviewed negative plots.

## 14. Local-overlap evaluation

All five coarse manifest negatives have nonempty box/LCS local annotations, so
the review classifies them as **partial overlap**, not true no overlap. Category
counts: positive **4**, partial overlap **5**, true no overlap **0**, unknown
**0**. Mean localization precision/recall: positives **.6448/.5877**; partial
overlap **.2239/.3451**. True-negative false-positive rate is **None**, because
this review contains no annotated true no-overlap pairs. Category assignment,
transcript comparison and GT loading occur after predictions/masks are fixed.

## 15. Remaining failures

The highlighted short route still lacks the configured global anchors;
ambiguous visual/letter evidence can pass optional-disabled confidence filters;
several local regions are missed or overextended. Short detections can also be
wrong, as negative 1 demonstrates. Candidate generation remains bounded and
greedy multi-region selection is not globally optimal. The nine examples are
too small to calibrate confidence or estimate a true-negative error rate.
Derived subword annotations are useful evaluation evidence, not manually
verified lexical truth. Validation calibration/manual annotations would be a
separate task; this change does not tune thresholds on these examples.

## 16. Model versus evaluator

The first-cell, support, suppression, gap, sampling and GT-category problems
were evaluator failures and are addressed. The remaining short-word rejection
is still an evaluator constraint, exposed by its exact anchor diagnostics.
The difficult false route's positive cosine and letter evidence, together with
negative alternative-match margins, point to ambiguous learned evidence. This
suggests a representation limitation and an uncalibrated decision rule; nine
pairs cannot isolate their contributions. Retraining is not assumed necessary
and was not performed. The model and production checkpoint remain unchanged.

Reproduce from the repository root with the project Python environment:

```sh
MPLCONFIGDIR=/tmp/alignment_mpl python scripts/eval/review_path_quality.py \
  --refresh-notebook
```

The review directory preserves `before_evaluate_v2.py` and uses it automatically
for the exact prior-evaluator comparison. `--before-source` can select another
preserved snapshot. The original before/after results are also retained in the
review JSON.
