# Frozen synthetic–real Arabic evaluation

Open [the executed notebook](../notebooks/synthetic_real_evaluation.ipynb) and run all cells to render Arabic and evaluate it against real A, B and C lines. Rendering uses `arabic_reshaper`, `python-bidi`, Pillow and the local Arslan Wessam and Amiri Regular fonts. It does not invoke the legacy Bridge corpus builders.

The notebook uses `Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt`, with optional-space handling and the existing image-only word matcher. It performs **zero training steps**. Model parameters, buffers, character directions, vocabulary and thresholds remain frozen. A, B and C are all represented in the checkpoint's saved training split; the six real anchors here belong to its saved test split. C is not an unseen training book.

## Scope and labels

The default experiment contains 60 pairs: two test anchors per book × two fonts × five cases. The cases are a complete rendered reference, two separated shared spans with a distractor, those same spans in reversed order, a full sequence with wider gaps between annotated glyph groups, and unrelated rendered text. This gives 48 positives and 12 constructed negatives.

Anchor selection uses a seeded order before seeing predictions. Source annotations must match image hashes, dimensions and normalized reference characters. Their boxes must lie inside the line image, and fonts must support the exact reshaped glyphs. Selection does not filter on model scores. The split audit found no test-image or test-page overlap with train/validation. Two pre-existing train/validation image duplicates are reported separately in the notebook.

Source transcripts and annotations are used to generate stimuli and evaluation masks. The prediction interface receives only two image paths and their features. Synthetic images use a separate external-image extractor; they are never added to the checkpoint's saved split membership. The fixed training vocabulary is not augmented with any evaluation reference.

The source annotations contain subwords/glyph groups. Partial blocks can cut through words, so rejection by a whole-word decoder is not automatically a word-alignment error. These are controlled **span-localization diagnostics**, not certified whole-word accuracy. Full-line controls retain all source characters. Wide-gap controls alter glyph-group joins and do not reconstruct natural word boundaries.

Negatives reject shared normalized three-character sequences and known words/groups of at least two letters, including substring checks against unspaced C references. They are constructed negatives, not human-reviewed semantic labels.

## Results

Measured on 9 November 2026: positive-mask **IoU 13.79%, Dice 19.55%, precision 33.21%, recall 14.66%**. Any region was detected in 18/48 positive pairs. Constructed-negative pair FPR was 0/12 (0%).

| Font | Positive pairs with any region | Positive IoU | Positive recall | Negative false positives |
| --- | ---: | ---: | ---: | ---: |
| Amiri-Regular | 17/24 | 26.95% | 28.63% | 0/6 |
| Arslan_Wessam | 1/24 | 0.63% | 0.69% | 0/6 |

| Real book | Positive IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| A | 16.46% | 23.01% | 37.71% | 17.46% |
| B | 14.35% | 19.35% | 26.93% | 15.70% |
| C | 10.56% | 16.30% | 35.00% | 10.82% |

The quantitative snapshot, source manifest and frozen-state checks are saved in [the results directory](evaluation/synthetic_real_frozen_20261109/). The notebook embeds visual examples for both fonts, using the first eligible anchor in each book for every case.

Localization metrics average the two image-side scores per pair, then average positive pairs; missed predictions score zero. Negative pair FPR counts unrelated pairs with any accepted region. Masked area on negatives is reported separately. Detection of any region does not mean every shared word or span was recovered.

The marked difference between the two fonts shows that successful real–real alignment does not imply comparable transfer to arbitrary rendered Arabic. This experiment changes neither the model nor its operating point. It cannot distinguish every source of failure, such as font appearance, spacing, vocabulary coverage and whole-word rejection of partial blocks.

Only six distinct real anchors are used, repeatedly across fonts and cases. The negative sample is too small and too constrained to establish a reliable 10% population false-positive rate. These results do not replace the real–real benchmark or establish an improvement over it.

## Reproduce

Use the repository environment with `arabic-reshaper`, `python-bidi`, Pillow, fonttools, PyTorch, pandas, matplotlib and a Jupyter kernel. The rendering packages are already listed in `requirements.txt`. The local checkpoint, `DataSet/ArabicData` line images/texts/BoundingBoxes, and the two configured font files must be present.

Run all cells in `notebooks/synthetic_real_evaluation.ipynb` from the repository or its notebooks directory. CUDA, MPS and CPU are supported through automatic device selection. The notebook writes generated images, masks, figures, a source manifest, split audit, per-pair/grouped CSVs and verification hashes under `Results/Evaluation/synthetic_real_frozen_20261109`. It fails if the frozen model contract changes.
