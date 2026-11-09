# Optional-space weights and independent passage evaluation — 9 November 2026

Both notebooks use `real_abc_balanced_optional_space_20261108/checkpoint_best.pt`. The matcher now permits disjoint passages in different orders, proposes gap-free passage candidates, checks long-region visual confidence, and can require multiple short passages to corroborate a line match. Individual paths remain monotonic and window spans cannot be reused.

## Held-out localization benchmark

32 saved-test positive pairs and 100 seeded presumed-negative comparisons. The user-provided test example was excluded before these headline scores; saved split IDs were never moved. Two duplicated Book C validation images were quarantined from tuning (147 positives, 98 negatives remain).

| Method | IoU | Dice | Precision | Recall | Positive pairs detected | Presumed-negative false detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous checkpoint, same cohort/settings | 21.18% | 26.06% | 25.46% | 35.86% | 14/32 | 18/100 (18.0%) |
| Current settings | 51.03% | 56.40% | 62.65% | 55.21% | 22/32 | 3/100 (3.0%) |
| Reordering only | 51.03% | 56.40% | 62.65% | 55.21% | 22/32 | 3/100 (3.0%) |
| Validation selected | 62.63% | 67.76% | 65.95% | 70.81% | 24/32 | 8/100 (8.0%) |

Higher recall costs more false detections: the validation-selected matcher improves held-out IoU by 11.60 percentage points and recall by 15.61 points, while presumed-negative FPR rises from 3% to 8%. No test parameters were changed after inspecting those scores. Reordering alone leaves these mask scores unchanged; the structural fix is verified separately by reversed-passage regression tests.

The reference checkpoint has recorded training exposure in 60/132 evaluated pairs. It and the current checkpoint share zero saved-test positives. These same-cohort old/new weight scores are descriptive, not an independent causal claim about training improvements.

## More metrics

| Method | Region F1 (IoU ≥ 0.5) | Negative masked area | Candidate-limit hits |
| --- | ---: | ---: | ---: |
| Previous checkpoint, same cohort/settings | 23.08% | 8.32% | 121/132 |
| Current settings | 60.00% | 0.46% | 59/132 |
| Reordering only | 60.00% | 0.46% | 59/132 |
| Validation selected | 70.07% | 1.59% | 59/132 |

[All metrics, confidence intervals, validation settings and split/exposure details](evaluation/reordered_regions_20261109/metrics.json). [Summary CSV](evaluation/reordered_regions_20261109/summary.csv), [per-pair scores](evaluation/reordered_regions_20261109/per_pair.csv), [validation presets](evaluation/reordered_regions_20261109/validation_presets.csv).

## Supplied example

`alignedPairs:pair_000028__A_line_005__B_line_002` contains three supplied matching fragments, “ابتغاء”, “الله”, and “من”, in the same order. Original rejection reasons: 556 insufficient-support candidates, 170 below path density, 2 insufficient anchors. The prior notebook required 9 windows per side, 12 matched cells, 95% density, and disabled short regions.

The selected matcher returns 2 regions instead of zero, but mean mask IoU is only 21.89%. This remains a weak, partially recovered case; the smaller fragment and exact correspondence need further improvement. Lower short-region thresholds produce more regions but also more false positives on validation, so they were not silently adopted.

## Reproduction and notebook refresh

The quantitative section is embedded near the top of both notebooks, including real before/after figures and an explicitly labeled synthetic reversed-order example. Reload/reopen the notebook from disk before restarting the kernel; restarting a kernel alone does not reload edited cell source. Previously rendered outputs from the original matcher are archived locally under `Results/Evaluation/reordered_regions_20261109/notebooks_before/`.

Run `python scripts/eval/review_reordered.py --output Results/Evaluation/reordered_regions_20261109 --device auto --workers 4`. Prepared feature/candidate caches retain the frozen baseline. A fresh output directory rebuilds features while restoring the pre-change matcher from the versioned metric report when available. The `--development-pair` images are excluded from headline test scoring. Then run `python scripts/eval/render_reordered_report.py --output Results/Evaluation/reordered_regions_20261109 --update-notebooks` to embed the measured report and before/after figures. Full inference artifacts stay under the output directory.

Mask localization metrics do not verify the identity of every word-to-word correspondence. Confidence intervals account for available shared line/page dependencies; 100 presumed negatives do not establish a universal false-positive rate.

## Real reordered-passage illustration

`pair_000001__A_line_011__B_line_005` has reordered annotated fragments: A contains “الذين … قولا غير”, while B contains “قولا غير … الذين”. Enforcing global order accepts 1 region; allowing disjoint reordered passages accepts 2. Mask IoU changes from 52.38% to 87.10% with all other settings fixed.

Both source lines are saved training members. This is an illustration of matcher behavior identified after parameter selection, not additional held-out model performance. The separate supplied three-word example above remains only partially recovered.
