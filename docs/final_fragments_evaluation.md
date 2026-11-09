# Final optional-space alignment evaluation — 9 November 2026

Checkpoint: `Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt`. Matcher: `local-path-quality-5-reordered-regions`. Representation: fused; crop: content.

Final results use **29 saved-test positives + 100 seeded presumed-negative comparisons**. Both supplied pairs and comparisons sharing any of their source images are excluded. This is a fresh execution of the final settings on the saved-test cohort, which has appeared in earlier aggregate reports; it is not a new independent benchmark.

| Metric | Final value |
| --- | ---: |
| Foreground mask IoU | 60.88% |
| Dice | 67.71% |
| Foreground precision | 63.96% |
| Foreground recall | 73.23% |
| Positive-pair detection recall | 79.31% |
| Presumed-negative pair false-positive rate | 17.00% |
| Presumed-negative masked area | 4.03% |

**Test false-positive rate exceeds the requested 10% budget.** Validation qualification does not establish a reliable deployment ceiling.

False detections: **17/100**. Positive pairs with any detected region: **23/29**. Foreground metrics are macro means over positive pairs (each pair averages its two line masks). Pair detection does not imply correct localization or correspondence.

Selection maximized validation foreground recall among 32 declared presets with presumed-negative pair FPR ≤ 10%. Validation: 147 positives, 98 negatives; recall 82.24%, FPR 8/98 (8.16%). The selected settings were frozen before scoring test. The test FPR is measured independently and is not guaranteed to satisfy the validation budget.

## What causes the missed passages?

Each passage must preserve its internal character order. Separate passages may change order, provided their window spans do not overlap on either line. Gap-free fragments are considered independently instead of being forced into one continuous path. Optional-space DTW follows the saved training alphabet and skip penalty; it cannot by itself remove image-to-image support and confidence rejections.

`pair_000016__A_line_014__B_line_003`: the short “الله” correspondence has a strong candidate with two distinct windows on A, three on B, four matched cells, mean cosine about 0.809 and three mutual anchors. A three-window minimum on each side rejects it even though the long “من المصلح ولو” passage is detected. Unequal window counts reflect writing scale and spacing; matched-cell count must not be mistaken for distinct support.

`pair_000028__A_line_005__B_line_002`: the supplied fragments “ابتغاء”, “الله”, and “من” occur in the same order. Its failure is not solely a reversed-order restriction. The full “ابتغاء” route is fragmented: one visually strong core supports only two A windows / three B windows and has only one mutual anchor (fraction 0.25). Some competing routes match the wrong word. “من” has a viable partial route near mean cosine 0.679. Relaxing confidence can recover foreground while also accepting wrong links. Masks measure location, not word identity; counting accepted regions cannot establish that every fragment was correctly paired.

## Final predictions for the two supplied cases

- `pair_000016__A_line_014__B_line_003`: 2 accepted regions; mask IoU **91.09%**. Final figure is embedded in both notebooks.
- `pair_000028__A_line_005__B_line_002`: 3 accepted regions; mask IoU **20.50%**. Final figure is embedded in both notebooks.

| Book pair | Positive pairs | Mask IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| A–B | 23 | 73.45% | 81.61% | 76.69% | 88.74% |
| A–C | 4 | 19.07% | 21.63% | 22.72% | 20.64% |
| B–C | 2 | 0.00% | 0.00% | 0.00% | 0.00% |

## Additional metrics and limits

Region interval F1: 62.50%; precision 59.21%; recall 66.18%. Counts over both lines of positive pairs: TP=45, FP=31, FN=23. Candidate-cap hits: 57.

Negatives are presumed unrelated comparisons, not manually verified negatives. Two train/validation duplicate images were quarantined before validation selection; the original saved split manifest was retained. Both positive and negative cohorts collapse to one connected dependency group, so valid grouped confidence intervals are unavailable. These are descriptive point estimates with limited independent evidence. Per-book-pair summaries use the same overall negative cohort.

The second supplied case remains a partial result. Recovering complete words without wrong links needs better candidate evidence and correspondence supervision, such as annotated fragment-to-fragment links and cross-book hard negatives, followed by validation on new pages. Merely increasing the allowed false-positive rate cannot guarantee those links.

[All metrics and provenance](evaluation/final_fragments_20261109/metrics.json) · [Final settings](evaluation/final_fragments_20261109/settings.json) · [Every scored pair](evaluation/final_fragments_20261109/per_pair.csv) · [Validation search](evaluation/final_fragments_20261109/validation_presets.csv)

## Reproduce

Run `python scripts/eval/final_fragments.py --device auto --workers 4`, then `python scripts/eval/render_final_fragments.py --update-notebooks`. Cached image features and complete image-only candidate pools are checked against the checkpoint and generation settings; metric values are recomputed. Reopen notebooks from disk to load edited sources. A kernel restart alone does not reload cells.
