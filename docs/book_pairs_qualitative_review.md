# A–B, B–C and A–C qualitative evaluation

Both notebooks use `real_abc_balanced_optional_space_20261108` with the final
validation-selected parameters. They embed final predictions for both supplied
cases and fresh inference on the fixed 15-example gallery.

See the [final quantitative report](final_fragments_evaluation.md) for all metric
values and the two case diagnoses. Test FPR is 17%, above the requested 10%
budget. The second supplied case remains partially recovered. The final gallery
is [`final_fragments_20261109/fixed_examples`](../Results/Evaluation/final_fragments_20261109/fixed_examples/index.html).
The 15 deliberately selected examples include mixed checkpoint splits; their
scores are illustrations, not an independent population estimate.

The catalog and original selection procedure below are historical context.
Those original ranking constraints do not describe the final model's scores.

The Drive folders `alignedPairs/A_B`, `alignedPairs/A_C`, and
`alignedPairs/B_C` were added under `DataSet/ArabicData`, retaining their
original images and masks. Each folder has 1,000 pairs, 2,000 images,
2,000 masks, 2,000 line texts and 2,000 full bounding-box JSON files.
Each catalog now contains all its active assets; its manifest paths point into
that catalog. Original Drive manifests are retained under
`metadata/catalog_layout/source_files/`. The source is the connected
[alignedPairs folder](https://drive.google.com/drive/folders/1wF9qzqxfJ8Ueg46yBx659vqbyirVO_Sl).
The local Drive snapshot and verification report are in
`DataSet/ArabicData/metadata/drive_sync_2026-11-07/`.

The complete `BookC` inventory contains 2,205 Kufic Quran lines. The independent
`Unalignedlines` inventories exclude lines with any saved partner across all
three catalogs: A has 841 lines, B has 314, and C has 800. These inventories do
not certify negative pairs. Book C transcriptions retain their source spacing
and blank values; full boxes retain their original geometry and review flags.

Open [the main evaluation notebook](../notebooks/model_evaluation.ipynb),
[the compact review notebook](../notebooks/book_pairs_qualitative.ipynb), or
[the figure gallery](../Results/Evaluation/book_pairs_examples/index.html).

The gallery shows **five small-region A–B pairs, five A–C pairs, and five B–C
pairs**. All 1,000 pairs in each catalog have foreground IoU scores. Selection
prioritizes catching the supplied white foreground on **both** lines: rank by
the smaller of the two white F2 scores, then mean foreground IoU, then the
smaller black-background specificity. F2 is `5 * TP / (5 * TP + 4 * FN + FP)`;
it gives missing white four times the weight of extra white. Each line must
have at least 80% white precision and 90% black specificity. These are sample
selection rules; prediction thresholds and metric formulas are unchanged.
Repeated transcript content and previously displayed pairs remain eligible.
No annotation-topology or diversity filtering is used. Rigorous upper bounds
on achievable F2 prune candidates that cannot beat the selected fifth pair;
all remaining candidates are verified with actual model predictions.
For A–B, the predicted white foreground must occupy 10–45% of each source line,
and both supplied and predicted foreground must fit within a 45% horizontal
span. This excludes short predictions that merely miss a large supplied mask.
Only three A–B pairs met the precision/background requirements with the earlier
35% limit; 45% permits five stronger matches while keeping them below half a line.
A–C and B–C have no supplied pair confined to 45% of both lines, so their five
examples are explicitly labeled as larger regions. Their available white-mask
recovery is weaker; displaying five does not certify that each is a good match.

The white foreground of each predicted mask is compared with the actual supplied
mask in source-image coordinates. IoU is foreground intersection over union.
Per-line precision measures how much predicted white is correct; recall measures
how much supplied white was recovered. Both IoUs, their mean, precision, recall,
and the actual white masks are shown. Black specificity is `TN / (TN + FP)`:
the fraction of supplied black preserved as black. It is shown separately from
foreground IoU, so a large black background cannot inflate the white score.
Masks are not altered to improve the score.
Each group also has a direct supplied-versus-predicted white-mask comparison
(`white_masks_A_B.png`, `white_masks_A_C.png`, `white_masks_B_C.png`), embedded
before its first detailed sample. Supplied and predicted masks share source
coordinates; only their display size changes.

The checkpoint is
`Weights/book_c_phase4_sum_nogate_ft_20261108/checkpoint_best.pt`.
Matching parameters, crop, representation, and device come from
`notebooks/model_evaluation.ipynb`. The prediction algorithm, mask projection,
and metric formulas are unchanged. Masks are read after prediction for scoring.

These examples are deliberately selected best results, not a population
performance estimate. Current checkpoint split membership is shown on each
figure, and automatic annotations retain their source provenance.

CSV files retain the unrestricted catalog IoU order; `selected_for_gallery` marks
the displayed pairs after applying foreground, background and size requirements.
Additional CSV columns contain measured recall, precision, specificity and F2;
blank values mean exact foreground verification was pruned by the bound.

Scores, ranking, figures, and the report are saved in
`Results/Evaluation/book_pairs_examples/`. Checkpoint hashes and parameter
fingerprints protect cached scores. The notebooks embed the final figures.

Run the complete search and render:

```sh
python scripts/eval/select_book_pair_examples.py
```

Independent catalog scans can also be run with `--group A_B --scan-only`
(and `A_C` / `B_C`), followed by `--render-only`. A render-only run fails if any
catalog pair has no compatible cached score. The default candidate limit is zero,
meaning the complete catalog. The A–B size bounds can be set with
`--ab-min-coverage` and `--ab-max-coverage`; using 0 and 1 disables the size filter. An explicit `--candidate-limit` is for exploratory
subsets and cannot substantiate a complete-catalog best-results claim.
