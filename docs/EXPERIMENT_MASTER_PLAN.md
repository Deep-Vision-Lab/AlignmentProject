# AlignmentProject — Revised Master Training and Evaluation Plan

This is the fixed research pipeline for the three canonical architecture branches:

- `agent/training-speed-optimization` — CNN / CNN+BiLSTM.
- `agent/use-vit-encoder` — ViT.
- `agent/use-dinov3-convnext` — DINOv3 ConvNeXt.

The scientific protocol is identical across branches. Only the visual encoder changes.

## Main decision

The RealSyntheticBridge V2 dataset is created, audited, and frozen **before any model training begins**. The former standalone canonical-real fine-tuning block (old S4-S6) is removed from the curriculum. After synthetic pretraining, the model adapts directly on Bridge V2, which already contains genuine real anchors plus controlled positive shared islands and guaranteed no-shared negatives.

This gives a cleaner experiment:

`frozen dataset -> synthetic pretraining -> zero-shot real evaluation -> Bridge-V2 evaluation -> Bridge-V2 fine-tuning -> post-Bridge evaluation -> final all-real evaluation`

## Fixed experimental rules

1. Build Bridge V2 once with seed 42 before training any architecture and reuse exactly the same frozen dataset for CNN, ViT, and DINOv3.
2. Never use validation/test handwriting to create training augmentation.
3. Keep `NUM_NEGATIVES=10` for image-text Span-DTW and activate the hardest four negatives unless running a dedicated negative-count ablation.
4. Keep preprocessing, window geometry, real diagnostic manifests, thresholds, and dataset splits fixed across architectures.
5. Bridge V2 positive rows contain intentional distractors, so generic whole-line positive sequence ranking must stay disabled. Only shared-island-aware bridge ranking is allowed for this baseline.
6. Bridge training is a longer adaptation run: default maximum 15 epochs, validation every epoch, and `checkpoint_best_val.pth` is the checkpoint used by later stages.
7. The final all-real evaluation is frozen. No training or threshold tuning follows it.
8. Model stages are connected with SLURM `afterok` dependencies. A failed stage blocks every downstream model stage.

---

# Phase A — Create the dataset before running models

## D0 — Build and audit RealSyntheticBridge V2

Run this once before CNN, ViT, or DINOv3 training.

Input: leakage-safe real training anchors from `DataSet/ArabicDataset`.

For every real anchor group, create:
- one synthetic positive containing 1, 2, or 3 ordered shared islands from the real transcript;
- unrelated synthetic distractor content between and/or beside those islands;
- a `128 x 1024` alignment mask: white = shared/aligned, black = distractor/unaligned;
- four guaranteed synthetic negatives by default;
- negatives may not share a complete normalized word or normalized 3-character sequence with the real anchor.

Output:

`DataSet/RealSyntheticBridge_v2/`

containing:
- `images/`
- `texts/`
- `masks/`
- `dataset_manifest.jsonl`
- `metadata.json`

Required audit before models are submitted:
- all files exist and load;
- image/mask dimensions are correct;
- positive island count distribution contains 1/2/3-island examples;
- masks agree with stored shared regions;
- negative overlap guarantees hold;
- a visual sample of positive images/masks is inspected;
- `smoke_test_real_synthetic_bridge.py` passes.

**Gate:** dataset smoke test passes. If D0 fails, do not submit any model training.

---

# Phase B — Model curriculum

## S1 — Synthetic pretraining

Input: `DataSet/AugmentedArabicDataset63`.

Purpose: learn generic Arabic visual/text alignment before adapting to scarce real handwriting.

Default settings:
- maximum 20 epochs;
- learning rate `1e-4`;
- effective global batch 64;
- window width 32;
- stride ratio 0.5;
- 10 text negatives per positive;
- 4 active hardest Span-DTW negatives;
- validation every 2 epochs.

Output: `Weights/<prefix>_synth/checkpoint_latest.pth`.

## S2 — Qualitative zero-shot real evaluation

Use the S1 checkpoint on unseen real handwriting before any Bridge training.

Inspect fixed examples from:
- `high_match`;
- `medium_match`;
- `low_match`;
- `no_shared_content`.

Save line pairs, similarity/DP heatmaps, Smith-Waterman paths, path length, matched fraction, score, and mean path cosine.

Questions:
- Are high/medium paths coherent?
- Are no-shared paths short/broken?
- Does the model focus on ink rather than blank margins?
- Are partial matches localized rather than stretched across unrelated regions?

## S3 — Quantitative zero-shot real evaluation

This is the synthetic-only real baseline.

Run:
1. held-out real bbox/localization metrics where annotations exist;
2. fixed positive-vs-no-shared discrimination sweeps at thresholds `0.40, 0.50, 0.60, 0.65, 0.70`.

Primary metrics:
- score;
- path steps;
- matched fraction;
- mean path cosine;
- region IoU and start/end error where available;
- ROC-AUC and average precision for positive vs no-shared discrimination.

## S4 — Bridge V2 evaluation before fine-tuning

Input checkpoint: S1 synthetic checkpoint.

Purpose: measure how the synthetic-only model behaves on the frozen augmentation dataset **before learning from it**.

Evaluate:
- real anchor vs Bridge-V2 positive;
- real anchor vs guaranteed negative;
- positive/negative score distributions;
- path steps and matched fractions;
- ROC-AUC / AP;
- mask-overlap/localization diagnostics on positive shared islands when available.

This stage separates dataset quality from training effects.

## S5 — Direct Bridge V2 fine-tuning

Input checkpoint: S1 synthetic checkpoint. There is no intermediate standalone real fine-tuning stage.

Bridge V2 provides the real-domain adaptation signal because every group contains a genuine real anchor.

Training composition:
- 50% Bridge positive rows;
- 50% guaranteed no-shared negative rows;
- real anchor image-text supervision;
- synthetic image-text supervision;
- image-image positive/negative discrimination;
- direct real-image/synthetic-text ranking restricted to actual shared islands;
- generic whole-positive-line sequence ranking disabled;
- mask carried for diagnostics/future mask-loss ablations, but no new mask loss in this baseline.

Text policy:
- AraBERT backbone frozen;
- shared projection/normalization/special embeddings remain trainable according to the existing bridge runtime.

Default settings:
- **maximum 15 epochs**;
- learning rate `1e-6`;
- validation every epoch;
- 10 image-text negatives per positive;
- 4 active hardest Span-DTW negatives;
- balanced positive/negative Bridge sampling;
- preserve and use `checkpoint_best_val.pth`.

The 15 epochs are a maximum, not an instruction to use the last epoch. All later evaluations use the best validation checkpoint. If the best validation checkpoint is still at epoch 15 and validation is still improving, a controlled continuation to 20 total epochs can be run as a follow-up rather than changing the first comparison protocol.

## S6 — Qualitative evaluation after Bridge fine-tuning

Use S5 `checkpoint_best_val.pth` and repeat the exact qualitative real protocol from S2.

Direct comparison: **S2 vs S6**.

## S7 — Quantitative evaluation after Bridge fine-tuning

Use S5 `checkpoint_best_val.pth` and repeat:
- the exact real quantitative protocol from S3;
- the Bridge-specific positive-vs-negative evaluation from S4.

Key comparisons:
- **S3 vs S7:** total improvement caused by direct Bridge-V2 adaptation;
- **S4 vs post-Bridge evaluation:** improvement on the augmentation task itself.

## S8 — Final complete-real evaluation

Freeze the S5 best-validation checkpoint and evaluate every relationship in `DataSet/ArabicDataset/dataset_manifest.jsonl`, separated into:
- `high_match`;
- `medium_match`;
- `low_match`;
- `no_shared_content`.

Report:
- exact sample count per class;
- mean/std score;
- mean path steps;
- mean matched fraction;
- mean path cosine;
- bbox/localization metrics where annotations exist;
- positive (`high+medium`) vs negative (`no_shared`) ROC-AUC and AP for score, path steps, and matched fraction;
- `low_match` separately as the ambiguous partial-overlap class.

No model or threshold changes are made after S8.

---

# Final comparison table

| Architecture | Checkpoint | Qualitative | Quantitative | Final all-real |
|---|---|---|---|---|
| CNN+BiLSTM | Synthetic S1 | S2 | S3 | — |
| CNN+BiLSTM | Bridge best S5 | S6 | S7 | S8 |
| ViT | Synthetic S1 | S2 | S3 | — |
| ViT | Bridge best S5 | S6 | S7 | S8 |
| DINOv3 ConvNeXt | Synthetic S1 | S2 | S3 | — |
| DINOv3 ConvNeXt | Bridge best S5 | S6 | S7 | S8 |

The same frozen Bridge V2 dataset and evaluation protocol must be used for every architecture.
