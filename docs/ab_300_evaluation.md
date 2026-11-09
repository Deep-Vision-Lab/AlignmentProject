# A–B quantitative evaluation on 600 pairs

The quantitative section of [model_evaluation.ipynb](../notebooks/model_evaluation.ipynb) now evaluates **300 aligned A–B pairs and 300 presumed unaligned A–B pairs**. Its population is independent of the qualitative sample slots and their smaller comparison pool. The frozen `real_abc_balanced_optional_space_20261108` checkpoint, full-line preprocessing and existing validation-selected word-matching settings are retained. No training, vocabulary rebuilding or parameter search is performed.

## Completed results — 9 November 2026

| Cohort | Aligned | Presumed negatives | IoU | Dice | Precision | Recall | Negative FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Large A–B catalog (mixed membership) | 300 | 300 | 73.21% | 81.07% | 75.73% | 89.68% | 23.67% |
| Saved-test subset; development excluded | 23 | 300 | 72.08% | 81.05% | 75.87% | 89.19% | 23.67% |

The 300-positive cohort produced foreground in **283/300** pairs (94.33%). The held-out subset produced foreground in **22/23** pairs. Both rows use the same 300 presumed-negative comparisons: **71 false detections**, or **23.67% FPR**. This exceeds the requested 10% operating target. Mean negative masked area was 6.60%.

| Additional metric | Large catalog | Held-out subset |
| --- | ---: | ---: |
| Positive interval-region precision | 69.17% | 66.67% |
| Positive interval-region recall | 61.36% | 86.96% |
| Positive interval-region F1 | 65.03% | 75.47% |
| Mean pair pixel accuracy | 90.46% | 93.12% |

The configured candidate cap was reached in 430/600 pairs. This run evaluates that bounded production matcher; the cap and acceptance settings were not changed after seeing the results. The foreground/background pixel-accuracy figure includes negative background and should be interpreted alongside foreground precision/recall and FPR.

All frozen-state checks passed and training steps were zero. The focused selection, scoring and existing word-matcher suite passed **16 tests**. [Metrics JSON](evaluation/ab_300_20261109/metrics.json), [per-pair CSV](evaluation/ab_300_20261109/per_pair.csv), [cohort manifest](evaluation/ab_300_20261109/cohort.json), [verification](evaluation/ab_300_20261109/verification.json).

## Population and split membership

The saved checkpoint membership covers 675 of the 1,000 supplied A–B aligned catalog pairs. Only 27 are entirely in its saved test split. The requested 300-positive diagnostic keeps all 27 and draws the remaining 273 from the broader saved catalog with seed 42. It contains:

| Positive pair membership | Pairs |
| --- | ---: |
| Train / train | 199 |
| Validation / validation | 54 |
| Test / test | 27 |
| Train / test | 11 |
| Train / validation | 8 |
| Validation / test | 1 |

Thus **218 positive pairs include a training image**. The 300-positive result is a mixed-membership diagnostic, not held-out generalization accuracy. The notebook also reports the 23 saved-test positives remaining after excluding the previously reviewed development pairs and all comparisons sharing their images. Saved splits are never regenerated or relabeled.

All 300 negatives use saved-test A–B `Unalignedlines` images. The existing catalog builder identifies 1,904 eligible combinations, excluding related pages, identical images and shared normalized transcript words; seed 42 selects 300 distinct comparisons. These are **presumed negatives**, not human-reviewed semantic no-overlap labels. Transcript text is used for this catalog check, never for prediction. Negative evaluation masks are black; aligned positives use their supplied masks.

The 600 distinct image pairs contain **583 unique images**. The negative comparisons reuse 91 images, so 300 comparisons do not represent 300 independent pairs of unseen source lines. The image/page audit found no saved-test overlap with training or validation. Two existing train/validation duplicate images in the full three-book dataset are reported in the audit; they do not become held-out observations.

## Metrics and interpretation

Positive IoU, Dice, foreground precision and foreground recall average the two source-image mask scores, then all positive pairs. Missed alignments contribute zero. Negative pair FPR is the number of presumed-negative pairs with any predicted foreground divided by 300. Mean negative masked area is a separate metric. Region F1 is available only when all source/reference masks support full-height interval scoring.

The large diagnostic and the saved-test subset have separate positive denominators. Detection of any region does not mean all shared words were recovered. Pixel accuracy is retained in the detailed output, but its background true negatives do not replace foreground localization scores.

These numbers measure the current frozen matcher on a larger cohort. They are not a before/after model comparison: the model and settings have not changed, and the positive sample now includes training and validation images. Reused source lines/pages also limit independence; a binomial confidence interval over 300 presumed negatives would not account for that dependence.

## Notebook and artifacts

The model/data configuration exposes `QUANTITATIVE_ALIGNED_PAIRS=300`, `QUANTITATIVE_UNALIGNED_PAIRS=300`, the seed, worker count and output directory. The notebook uses one scoring worker on the local 8 GB machine. CNN features and image-only prediction masks are cached with checkpoint, input, preprocessing, matcher-source and settings identities. Reruns reuse valid predictions and score the supplied ground truth again.

Run the configuration, folder-mask helpers, session, matching-settings cells and section 7. The quantitative section uses the active matching settings and prints the 600-pair summary, separate held-out summary, membership counts and unchanged-state verification. It does not change the ordinary saved-test session or its qualitative catalog.

Detailed generated artifacts are written to `Results/Evaluation/ab_300_20261109/`: the selected `cohort.json`, `per_pair.csv`, `metrics.json`, `summary.csv`, `membership.csv`, `run_settings.json`, split audit, feature/prediction caches and `verification.json`. Compact portable copies are saved under `docs/evaluation/ab_300_20261109/` after the completed run. Model parameters, buffers, character directions, settings and vocabulary are checked for changes before returning results.
