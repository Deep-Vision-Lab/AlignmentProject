# AlignmentProject — Master Training and Evaluation Plan

This document defines one fixed research pipeline for the three canonical visual branches:

- `agent/training-speed-optimization` — CNN / CNN+BiLSTM branch.
- `agent/use-vit-encoder` — ViT branch.
- `agent/use-dinov3-convnext` — DINOv3 ConvNeXt branch.

The scientific order is intentionally identical across branches. Only the visual encoder changes. Data splits, text encoder policy, training objectives, evaluation manifests, thresholds, and stage order must stay fixed whenever architectures are compared.

## Research goal

Learn local image representations that align Arabic manuscript line images without requiring OCR at evaluation time. Training may use transcript supervision to shape the visual embedding space. Evaluation is image-to-image: local visual embeddings are compared with cosine similarity and aligned with Smith–Waterman local alignment.

## Fixed experimental rules

1. Never use validation/test handwriting to create training augmentations.
2. Keep the same train/validation/test split seed across branches.
3. Keep `NUM_NEGATIVES=10`; activate the hardest four negatives for Span-DTW unless an ablation explicitly changes it.
4. Use the same image geometry, preprocessing, window size, and evaluation manifests across architecture comparisons.
5. Do not tune a model on the final full-real evaluation. Final evaluation is run only after the augmented fine-tuning stage is frozen.
6. Save all logs under `out/`, checkpoints under `Weights/<job_id>/`, and evaluation artifacts under `Results/Evaluation/ResearchPipeline/<run_prefix>/`.
7. Every automatic stage uses an `afterok` SLURM dependency. If a stage fails, every downstream stage remains blocked instead of continuing on invalid outputs.

---

# Stage sequence

## S0 — Preflight and experiment freeze

Purpose: make sure the branch, datasets, caches, scripts, and hardware assumptions are valid before GPUs are requested.

Checks:
- active branch is one of the canonical branches;
- `DataSet/AugmentedArabicDataset63` exists for synthetic pretraining;
- `DataSet/ArabicDataset/dataset_manifest.jsonl` exists for real evaluation/fine-tuning;
- AraBERT files are available locally/offline;
- DINO branch additionally requires `DINOV3_REPO_DIR`;
- output directories can be created;
- the exact git commit and backend name are recorded.

Output: a frozen run prefix, branch name, commit, and backend used by every later stage.

Gate: all preflight checks pass.

## S1 — Synthetic pretraining

Input: `DataSet/AugmentedArabicDataset63`.

Purpose: first learn the generic visual-to-text alignment geometry on the larger synthetic corpus before exposing the model to scarce real handwriting.

Default research settings:
- max epochs: 20;
- learning rate: `1e-4`;
- effective global batch: 64;
- window size: 32 px;
- synthetic stride ratio: 0.5;
- 10 negatives per positive;
- 4 active hard negatives for Span-DTW;
- validation every 2 epochs.

What is learned:
- visual local/window representation;
- sequence/context representation supplied by the branch visual encoder;
- shared-space projection against Arabic text spans;
- local discrimination against negative text sequences.

Output: `Weights/<prefix>_synth/model_latest.pth` and `checkpoint_latest.pth`.

Gate: training completes and validation is finite. The later evaluation stages determine whether the representation is scientifically useful.

## S2 — Qualitative evaluation after synthetic pretraining

Purpose: inspect what the synthetic-trained model does on unseen real handwriting before any real fine-tuning.

Evaluate a small fixed set from every real relationship class:
- `high_match`;
- `medium_match`;
- `low_match`;
- `no_shared_content`.

Save:
- original line pair;
- cosine/DP heatmap;
- Smith–Waterman traceback/alignment path;
- score, path length, matched fraction, and mean path cosine.

Questions to answer visually:
- Are high/medium pairs producing coherent local paths?
- Are no-shared pairs broken/short rather than forming long false alignments?
- Is the model focusing on ink rather than blank margins?
- Are partial matches localized rather than stretched through unrelated regions?

Output: `Results/Evaluation/ResearchPipeline/<prefix>/s2_synth_qualitative/`.

## S3 — Quantitative evaluation after synthetic pretraining

Purpose: establish the zero-shot real baseline before fine-tuning.

Two evaluations are run:

1. Real bbox/localization metrics on the held-out real split for high/medium relationships.
2. Fixed positive-vs-no-shared discrimination sweep at thresholds `0.40, 0.50, 0.60, 0.65, 0.70` using the exact same deterministic diagnostic manifests for all branches.

Primary metrics:
- alignment score;
- path steps;
- matched fraction;
- mean path cosine;
- region IoU / start-end localization error when bbox ground truth exists;
- ROC-AUC and average precision for positive vs no-shared discrimination.

Output: `.../s3_synth_quantitative/`.

This stage is the zero-shot reference that every later stage must be compared against.

## S4 — Canonical real fine-tuning, no augmentation

Input checkpoint: synthetic model from S1.

Training data: canonical train-safe real `high_match` and `medium_match` pairs only. No bridge augmentation is used in this stage.

Purpose: adapt synthetic visual geometry to real manuscript texture, stroke statistics, scanning artifacts, and cross-writer appearance without mixing the later augmentation experiment into the first real adaptation.

Default research settings:
- 5 epochs;
- learning rate `2e-6`;
- 10 negatives per positive;
- 4 active Span-DTW negatives;
- real binarization with Otsu preprocessing;
- image-text loss on both real lines;
- image-image positive correspondence enabled;
- `AUGMENT=0`.

Output: `Weights/<prefix>_real/model_latest.pth`.

## S5 — Qualitative evaluation after canonical real fine-tuning

Repeat exactly the S2 qualitative set and rendering configuration using the S4 checkpoint.

Purpose: visually compare synthetic-only vs real-adapted behavior on the same kinds of examples.

Output: `.../s5_real_qualitative/`.

## S6 — Quantitative evaluation after canonical real fine-tuning

Repeat exactly the S3 quantitative protocol using the S4 checkpoint.

Purpose: measure whether real adaptation improved localization and discrimination without moving the goalposts.

Output: `.../s6_real_quantitative/`.

The important comparison is S3 -> S6.

## S7 — Build and audit RealSyntheticBridge V2 augmentation

Input: leakage-safe real training anchors from `DataSet/ArabicDataset`.

For every real anchor group, create:
- one synthetic positive line containing 1, 2, or 3 randomly selected aligned islands from the real transcript;
- unrelated synthetic distractor regions between/beside the aligned islands;
- a `128 x 1024` alignment mask where aligned regions are white and unaligned/distractor regions are black;
- K synthetic negative lines, default 4, with no complete normalized shared word and no shared normalized 3-character sequence with the real anchor.

The positive shared islands preserve their original transcript order. Validation/test pages are excluded before generation.

Dataset output: `DataSet/RealSyntheticBridge_v2/` containing `images/`, `texts/`, `masks/`, `dataset_manifest.jsonl`, and `metadata.json`.

Audit before any fine-tuning:
- smoke test paths and image dimensions;
- positive island count distribution (1/2/3);
- masks exactly match stored shared boxes;
- negatives satisfy the no-overlap guarantee;
- inspect generated positive image + mask pairs visually.

Gate: `SMOKE_TEST=PASS`.

## S8 — Evaluate the augmentation before training on it

Purpose: separate dataset quality from training effects. The S4 real-fine-tuned model is evaluated zero-shot on the newly generated bridge corpus before it is allowed to learn from it.

Qualitative:
- real anchor vs synthetic positive heatmaps;
- real anchor vs synthetic negative heatmaps;
- inspect whether the current model already concentrates on the white-mask shared regions.

Quantitative:
- positive vs negative score distributions;
- path steps and matched fractions;
- positive-vs-negative ROC-AUC / AP;
- optional mask-overlap metrics from the positive alignment mask.

Output: `.../s8_bridge_pretrain_eval/`.

## S9 — Fine-tune on RealSyntheticBridge V2

Input checkpoint: S4 canonical real checkpoint.

Training composition:
- 50% positive bridge rows;
- 50% guaranteed no-shared negative rows;
- image-text supervision remains valid for every synthetic segment and every real anchor transcript;
- bridge-specific real-image/synthetic-text ranking uses only the actual shared islands;
- whole-line positive sequence ranking is disabled for V2 because distractor content is intentionally unaligned;
- the mask is carried through the dataloader for diagnostics/future mask-loss ablations, but this baseline does not add a new mask loss.

Text policy:
- AraBERT backbone frozen;
- shared-space projection / normalization / learned special embeddings trainable.

Default research settings:
- max 8 epochs;
- learning rate `7.5e-7`;
- validation every epoch;
- preserve `checkpoint_best_val.pth`.

Output: `Weights/<prefix>_bridge_v2/checkpoint_best_val.pth` plus latest checkpoints.

## S10 — Post-augmentation qualitative and quantitative evaluation

Use the S9 best-validation checkpoint.

Repeat:
- the exact real qualitative protocol from S2/S5;
- the exact real quantitative protocol from S3/S6;
- bridge positive-vs-negative diagnostics.

Purpose: isolate the improvement caused specifically by augmentation fine-tuning.

Key comparisons:
- S3 vs S6: effect of canonical real fine-tuning;
- S6 vs S10: effect of bridge augmentation fine-tuning;
- S3 vs S10: total improvement from the complete curriculum.

Output: `.../s10_post_bridge_qualitative/` and `.../s10_post_bridge_quantitative/`.

## S11 — Final evaluation on the complete real dataset

This is the final frozen evaluation. No more training or threshold tuning follows it.

Evaluate every row in `DataSet/ArabicDataset/dataset_manifest.jsonl`, separated by:
- `high_match`;
- `medium_match`;
- `low_match`;
- `no_shared_content`.

Use one fixed operating threshold (pipeline default `0.50`, override only before the final run if a threshold was selected from development diagnostics).

Report:
- sample count per label;
- mean/STD alignment score;
- mean path steps;
- mean matched fraction;
- mean path cosine;
- bbox/localization metrics where annotations exist;
- positive (`high+medium`) vs negative (`no_shared`) ROC-AUC and AP for score, path steps, and matched fraction;
- `low_match` separately as an ambiguous/partial-overlap class rather than forcing it into positive or negative binary labels.

Output: `.../s11_final_all_real/` including per-sample CSVs and `final_summary.json`.

---

# Final experiment table

For the paper/report, every architecture should end with one row per checkpoint stage:

| Architecture | Checkpoint stage | Zero-shot/FT | Real qualitative | Real quantitative | Full-real final |
|---|---|---|---|---|---|
| CNN+BiLSTM | Synthetic S1 | zero-shot | S2 | S3 | — |
| CNN+BiLSTM | Real S4 | real FT | S5 | S6 | — |
| CNN+BiLSTM | Bridge S9 | augmentation FT | S10 | S10 | S11 |
| ViT | Synthetic S1 | zero-shot | S2 | S3 | — |
| ViT | Real S4 | real FT | S5 | S6 | — |
| ViT | Bridge S9 | augmentation FT | S10 | S10 | S11 |
| DINOv3 ConvNeXt | Synthetic S1 | zero-shot | S2 | S3 | — |
| DINOv3 ConvNeXt | Real S4 | real FT | S5 | S6 | — |
| DINOv3 ConvNeXt | Bridge S9 | augmentation FT | S10 | S10 | S11 |

The winning architecture must be selected from the same frozen protocol, not from architecture-specific test tuning.
