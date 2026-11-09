# Image-only word verification and A/B/C evaluation — 9 November 2026

The A–B confidence preset was subsequently refined; see [the current A–B results](ab_false_positive_reduction.md). The A–C and B–C base settings remain unchanged. A–B metrics below describe the earlier preset.

Weights: `Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt`; full-line preprocessing (`crop=none`); decoder `word_dtw`.

All three books are recorded in training: A=1,107 lines, B=1,107 lines, C=1,107 lines. A figure’s line split describes that individual image. A line outside the balanced saved subset does not imply that its book was absent from training. The refreshed figures show book exposure and line membership separately.

The decoder preserves strong long visual passages, including words absent from its fixed vocabulary. Short regions require a whole-word hypothesis on each image with matching normalized identity. A full-line lattice makes longer words compete with their internal letter fragments. Every proposed word also undergoes a full-span visual DTW check; masks are clipped to its inferred edges. Without a strong long passage, multiple distinct words and sufficient combined support must corroborate the line match; an isolated common word does not suffice. Separate passages may appear in different orders without reusing either span.

Inference reads image pixels, frozen model features and a fixed vocabulary built from saved training IDs only. It does not read the evaluated pair’s transcriptions, masks or annotated boxes. The vocabulary is an additional domain prior, not a change to the supplied neural weights. It is built from current local training-split files; their historical equivalence to the original HPC text snapshot is unverified.

## Final saved-test point estimates

| Book pair | Positive pairs | Mask IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| A–B | 23 | 72.08% | 75.87% | 89.19% |
| A–C | 4 | 54.24% | 64.70% | 62.76% |
| B–C | 2 | 54.91% | 93.07% | 56.90% |

Overall: IoU **68.44%**, Dice **77.79%**, precision **75.52%**, recall **83.32%**. Detected positive pairs: 27/29. Presumed-negative false detections: **15/100 (15.00%)**. Test FPR **exceeds** the requested 10% budget. Negative masked area: 3.54%.

Validation selection gives equal weight to A–B, A–C and B–C and maximizes mean foreground F0.5, which weights precision more than recall, under the 10% presumed-negative FPR budget. Both supplied development cases are regression requirements: the three-word case must recover its three word hypotheses without extra regions; the reordered case must retain its long passage and short word without an extra region. Settings were frozen before the final test run. Both supplied examples and all test comparisons sharing their source images are excluded. This saved-test cohort has been examined in previous aggregate reports and is not a new independent benchmark.

## Supplied cases

- `pair_000016__A_line_014__B_line_003`: IoU **89.50%**; normalized word hypotheses `الله`; 1 retained long visual passage(s).
- `pair_000028__A_line_005__B_line_002`: IoU **76.01%**; normalized word hypotheses `ابتغا`, `الله`, `من`; 0 retained long visual passage(s).

Canonical word identities merge common letter-form variants and remove standalone hamza; they are hypotheses, not verified OCR transcriptions. In particular, `ابتغا` is the normalized template used when checking “ابتغاء”. Repeated occurrences of the same word remain ambiguous and localization may be partial.

## Training and spacing findings

The checkpoint uses optional spaces and was trained on A, B and C. Optional-space image-to-text DTW can consume a space across zero or several windows; it does not, by itself, certify whole-word image-to-image matches. The checkpoint uses active hybrid typed negatives: strong-negative weight 0.5, wrong-letter weight 0.5 and order-negative weight 0.25. Their recorded losses are nonzero. `negative_dtw_weight=0` applies to the separate legacy term and does not disable those hybrid branches. These losses train image-to-transcript discrimination; they do not provide direct supervision for every complete-word image-pair boundary.

Current local C text files are concatenated. The user confirmed that the images are the same on Drive and the HPC; that does not establish the literal whitespace in the original training text files. Original training-text formatting is left unverified. The current word prior therefore receives no spaced-word tokens from local C files. This limits vocabulary coverage; it does not mean C was absent from neural training.

## Limits and reproducibility

The vocabulary can omit rare words and can contain imperfect tokens inherited from source transcripts. Long visual passages preserve some unknown-word coverage. Word boundaries and word identities remain predictions. Supplied foreground masks score localization, not correctness of every word correspondence. The region interval F1 in the metric JSON is not word-recognition F1.

Negatives remain presumed unrelated comparisons. Shared short words or incomplete source transcriptions can complicate their interpretation. Available line/page dependencies collapse the cohorts to one connected group, so grouped confidence intervals are unavailable. A–C and B–C test counts are particularly small.

[All metric values and provenance](evaluation/word_alignment_20261109/metrics.json) · [Final settings](evaluation/word_alignment_20261109/settings.json) · [Every scored pair](evaluation/word_alignment_20261109/per_pair.csv) · [Validation presets](evaluation/word_alignment_20261109/validation.csv)

Run `python scripts/eval/review_words.py`, then `python scripts/eval/render_word_review.py --update-notebooks`. To refresh the fixed gallery, run `scripts/eval/review_book_pairs.py` with `--settings-report Results/Evaluation/word_alignment_20261109/word_selected.json --crop none` and the saved fixed selection. Reopen the notebooks from disk to load edited cell sources.
