# A–B false-positive reduction with protected positive examples

The A–B confidence preset was selected on **134 saved-validation aligned pairs and 300 saved-validation presumed negatives**. Model weights, CNN features, optional-space handling, word vocabulary, candidate generation and long-passage matching are unchanged. The search changes only word-confidence and corroboration settings. Matching still uses images/features and the fixed training vocabulary; evaluated transcriptions and masks are not matching inputs.

## Validation selection

The search proposed 238 presets. Before scoring the validation population, development regression guards rejected 212: each of the two previously reviewed examples could lose at most 0.05 absolute mask IoU, and the three-word example had to retain all its baseline word identities. The remaining 26 presets were compared by **maximum validation positive mask IoU subject to presumed-negative FPR ≤10%**, with recall and precision breaking ties. These are the best tested settings under that declared objective, not a claim of a global optimum.

| Validation metric | Previous preset | Selected A–B preset |
| --- | ---: | ---: |
| Positive IoU | 73.49% | 73.86% |
| Dice | 81.23% | 80.79% |
| Foreground precision | 75.97% | 76.56% |
| Foreground recall | 89.87% | 87.86% |
| Negative false detections | 69/300 | 30/300 |
| Presumed-negative FPR | 23.00% | 10.00% |

The full validation audit attributed 42 of the 69 false detections to word-only matches and 27 to matches containing a retained visual passage. The selected thresholds primarily remove weak word-only correspondences; long visual passage acceptance is unchanged. A strong long passage still permits word additions. Without such a passage, weaker sets require three distinct word hypotheses with sufficient combined character/window support. Exactly two words can instead establish a match when both satisfy the stronger lexical and visual checks below. Weak short overlaps can still be rejected.

| Confidence setting | Previous | Selected |
| --- | ---: | ---: |
| Maximum word lexical NLL | 1.8 | 1.6 |
| Minimum posterior for short words | 0.15 | 0.25 |
| Minimum whole-word visual cosine | 0.4 | 0.6 |
| Minimum distinct corroborating word types | 2 | 3 |
| Minimum combined word characters | 8 | 10 |

An additional fixed training example exposed a weakness of a hard three-word requirement. A second, validation-only refinement tested 15 confidence exceptions plus the disabled control; ten retained the declared development examples. It allows **exactly two independently strong words** when each has side-wise lexical NLL ≤0.55 and whole-word cosine ≥0.75. The existing combined character and physical-support requirements still apply. Exact validation ties favor stricter exception thresholds. The earlier test computation was interrupted before aggregate results were read; no test score selected either refinement.

The fixed training example contains two correctly aligned complete words (`الذين`, `كفروا`) plus an old weak extra fragment. Its IoU improves from **84.75% to 94.95%** after preserving the two strong words and removing the fragment. This example is a declared development regression, not a held-out observation.

Other parameters are unchanged. The posterior is relative lattice evidence, not a calibrated probability of correctness.

The reviewed development examples remain outside headline held-out positive scores:

- `pair_000016__A_line_014__B_line_003`: IoU stays **89.50%**.
- `pair_000028__A_line_005__B_line_002`: IoU increases from **76.01% to 78.57%**, retaining `ابتغا`, `الله`, and `من` as the three normalized hypotheses.

## Scope and test protocol

The stricter preset applies **only to the physical A–B book pair**. Applying it indiscriminately to C reduced validation recall, so A–C, B–C and unknown book pairs retain their previous base settings. Book identity comes from catalog metadata/image paths and supplies only the preset scope. It supplies no transcript, word label or alignment target. The notebook records and displays the per-pair policy, and matching-cache keys include the actual applied settings.

After the preset was frozen, it was evaluated on the **exact same 300 aligned and 300 presumed-negative A–B pairs** as the previous large run. Its mixed-positive membership is retained: 218 positive pairs include a training image. The separate saved-test score uses the 23 positive pairs remaining after development-image exclusion. These reused cohorts are a controlled before/after comparison, not new independent test positives.

An additional 300 saved-test A–B presumed-negative combinations exclude all 300 original negative pairs. They come from the remaining 1,604 eligible comparisons and reuse the same test-line inventory; disjoint combinations do not imply independent new source images. No test or additional-negative score is used to choose a preset or revise its thresholds.

## Final before/after evaluation

| Same 300 aligned + 300 presumed-negative A–B pairs | Previous | New |
| --- | ---: | ---: |
| Mask IoU | 73.21% | 71.93% |
| Dice | 81.07% | 78.89% |
| Foreground precision | 75.73% | 74.40% |
| Foreground recall | 89.68% | 85.84% |
| Interval-region F1 | 65.03% | 66.07% |
| Negative pair FPR | 23.67% | 10.33% |
| Mean negative masked area | 6.60% | 2.87% |
| Positive pairs with any foreground | 283/300 | 270/300 |
| Negative false detections | 71/300 | 31/300 |

The original-cohort false-positive count falls by **56.3%**. Its 10.33% FPR remains just above the 10% target. Positive localization recall and the number of detected positives fall; this is a measured trade-off, not an improvement in every metric. The new point maximizes validation positive IoU among the tested settings satisfying the declared validation/development constraints.

| Saved-test positive subset, same 23 pairs | Previous | New |
| --- | ---: | ---: |
| Mask IoU | 72.08% | 70.71% |
| Dice | 81.05% | 77.53% |
| Foreground precision | 75.87% | 74.28% |
| Foreground recall | 89.19% | 82.87% |
| Interval-region precision | 66.67% | 92.86% |
| Interval-region recall | 86.96% | 84.78% |
| Interval-region F1 | 75.47% | 88.64% |

The additional negative audit returned **20/300 (6.67%)** false detections. Across the original and additional disjoint pair sets, the selected preset produces **51/600 (8.50%)** false detections. The previous preset was evaluated on only the original 300 negatives, so 8.50% is an expanded-cohort point estimate rather than a same-cohort before/after comparison.

## Same fixed A–B illustrative examples

| Fixed example | Previous IoU | New IoU |
| --- | ---: | ---: |
| A_B_example_01 | 78.62% | 92.27% |
| A_B_example_02 | 87.44% | 87.44% |
| A_B_example_03 | 84.75% | 94.95% |
| A_B_example_04 | 84.83% | 84.83% |
| A_B_example_05 | 86.39% | 86.39% |

These five examples were retained without reselection; they include training, validation and outside-subset images and are illustrative rather than headline test evidence. Both notebooks embed their refreshed A–B figures and the two reviewed development cases. The ten C gallery cells are byte-for-byte unchanged.

The complete suite passed **388 tests and 7 subtests**. Frozen weights, character directions, vocabulary and active settings passed the unchanged-state checks. The small two-word exception is disabled by default and activated only by the scoped A–B preset.

[Selected settings](evaluation/ab_low_fpr_20261109/selected.json) · [Full comparison](evaluation/ab_low_fpr_20261109/comparison.json) · [All 600 scores](evaluation/ab_low_fpr_20261109/per_pair.csv) · [Additional negatives](evaluation/ab_low_fpr_20261109/additional_negatives.csv) · [Verification](evaluation/ab_low_fpr_20261109/verification_summary.json)

## Reproduce and inspect

Run `scripts/eval/calibrate_ab_confidence.py` for the coarse validation search, `scripts/eval/refine_ab_confidence.py` for the guarded strong-pair refinement, then `scripts/eval/evaluate_ab_confidence.py` to evaluate the frozen choice. The calibration commands expect the saved validation/test feature artifacts and fixed-example selection from the preceding word review; they reuse the verified features and candidate pools under `Results/Evaluation/word_alignment_20261109`; the preset itself is committed and the main notebook can evaluate it directly from the checkpoint and `DataSet/ArabicData`.

In [model_evaluation.ipynb](../notebooks/model_evaluation.ipynb), the base settings remain available for C and `AB_CONFIDENCE_PRESET` loads the new A–B overrides. Run the configuration, folder-mask helper, session and settings cells, then the quantitative section. CNN features and predictions are cached; no training job is involved.

Generated calibration artifacts live in `Results/Evaluation/ab_low_fpr_20261109`, and the 600-pair evaluation plus additional-negative audit live in `Results/Evaluation/ab_300_low_fpr_20261109`. Compact reports, selection protocol, settings, per-pair scores and verification hashes are copied under `docs/evaluation/ab_low_fpr_20261109` after the completed evaluation.
