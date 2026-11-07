# A–B, B–C and A–C qualitative evaluation

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

Open [the review notebook](../notebooks/book_pairs_qualitative.ipynb) or
[the figure gallery](../Results/Evaluation/book_pairs_qualitative/index.html).
The overview sheets show all ten examples for
[A–B](../Results/Evaluation/book_pairs_qualitative/contact_A_B.png),
[B–C](../Results/Evaluation/book_pairs_qualitative/contact_B_C.png), and
[A–C](../Results/Evaluation/book_pairs_qualitative/contact_A_C.png).
Green bands above each source image show supplied mask foreground; blue
bands below show the model prediction.

The fixed seed-42 selection contains five consecutive and five non-consecutive
pairs per book pair, with distinct lines within each stratum. Consecutive means
one connected foreground interval in both supplied masks. Non-consecutive means
at least two separated intervals in both masks. Asymmetric cases are excluded.
This describes annotation topology; the model can predict a different number
of regions. Selection never uses the model's output.

All A/B lines are in the checkpoint's saved test membership. Book C lines are
external to that membership. Membership was checked against both the saved
split manifest and the actual checkpoint. This does not establish that Book C
transcripts are disjoint from training transcripts. No training, split changes
or threshold search was performed.

Inference uses `Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt`,
the fused representation, CPU, and the existing notebook's content crop and
matching settings. The checkpoint SHA256 is
`d5fcb9f706a6456b38e1cef100a1b8e0ac491c79d653a968eb8ebdb209d023d6`.
The report saves the resolved settings, input hashes, predicted intervals,
candidate diagnostics and mask metrics. Supplied masks are read after prediction
for scoring; they do not enter the matcher. Automatic subword annotations retain
their original provenance and have not been certified by this review.

| Book pair | Supplied path type | Samples | Pairs with accepted routes | Mean A/B-side IoU |
| --- | --- | ---: | ---: | ---: |
| A–B | Consecutive | 5 | 5 | 0.861 |
| A–B | Non-consecutive | 5 | 5 | 0.424 |
| B–C | Consecutive | 5 | 3 | 0.150 |
| B–C | Non-consecutive | 5 | 2 | 0.071 |
| A–C | Consecutive | 5 | 2 | 0.134 |
| A–C | Non-consecutive | 5 | 3 | 0.158 |

The IoU column averages the two input-line IoUs for each sample, then averages
the five samples. Accepted routes do not necessarily overlap the supplied masks.
These small, stratified summaries describe this review only.

Visual inspection of all three overview sheets shows close boundary agreement
on most continuous A–B examples. `A_B_non_consecutive_01` and `_05` recover the
dominant region but miss a smaller supplied region. `_02` and `_04` place much
of the foreground away from the supplied regions; `_03` recovers only a subset.

Book C is delivered as black-and-white Kufic line imagery, unlike the manuscript
scans in A/B. C comparisons often produce no accepted route or cover only part
of a large supplied region. `B_C_consecutive_05` has zero B-side IoU despite an
accepted route; `B_C_non_consecutive_02` detects a region away from much of the
supplied B-side foreground. `A_C_consecutive_01` follows part of the shared span
but omits much of its extent. Several other C examples have completely empty
predictions. These observations suggest a domain/representation issue worth
investigating, but this review cannot separate representation, decoder and
automatic-annotation errors.

Reproduce from the repository root in the project Python environment:

```sh
python scripts/data/validate_book_pair_folders.py
python scripts/eval/review_book_pairs.py --device cpu
```

Use `--prepare-only` to regenerate just the fixed selection without loading
the model. The existing main evaluation notebook is preserved. The new notebook
contains saved real inference figures with explicit provenance; rerunning its
cells redisplays those artifacts. Run the script to recompute predictions.
