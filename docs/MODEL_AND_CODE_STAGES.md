# Model and Code Stages — ViT Branch

Branch: `agent/use-vit-encoder`

Active backend: `vit` selected by `model_backend.py`.

The training curriculum, datasets, text encoder, losses, evaluation manifests, and SLURM stage order are intentionally shared with the CNN and DINOv3 branches. The controlled architecture difference is the visual encoder.

## 1. Input and local sequence

Each manuscript line is normalized to the common `128 x 1024` canvas and represented as an ordered sequence of horizontal local observations. Standard window width is 32 px; stride is configured per training/evaluation stage.

Core shared code:
- `unified_line_geometry.py`;
- `DataLoader.py` / `RealDataSet.py`;
- `training_runtime/entrypoint.py`.

## 2. ViT visual encoder

`model_backend.py` builds the model through `vit_embedding_model.py`.

Canonical defaults recorded by the backend:
- input height: 128;
- transformer layers: 4;
- attention heads: 4;
- MLP dimension: 512;
- dropout: 0.10;
- maximum tokens: 256;
- positional base tokens: 63;
- output projected into the shared visual/text embedding dimension.

For this branch:
- `USE_BILSTM=0`;
- `USE_LOCAL_WINDOW_GROUPING=0`.

Sequence context therefore comes from transformer self-attention rather than an additional recurrent encoder. This keeps the ViT experiment interpretable: the visual transformer itself is responsible for contextualizing the ordered line representation.

Main code:
- `model_backend.py`;
- `vit_embedding_model.py`;
- `Evaluation/vit_evaluation.py` for checkpoint-aware evaluation reconstruction.

## 3. Arabic text encoder

The text side is shared across branches and uses `ArabicSpanTextEncoder`:
- AraBERT v02 backbone;
- frozen backbone features;
- trainable projection into the shared space;
- trainable special-space/blank representations;
- cached frozen surface features.

Main code:
- `arabic_span_text_encoder.py`;
- `bridge_frozen_text.py`.

## 4. Training objective

For each image line the ViT produces normalized local/contextual visual embeddings. Arabic span embeddings are normalized into the same space. Their similarity matrix is optimized by differentiable Span-DTW.

Standard negative policy in the research pipeline:
- 10 negative texts per positive;
- 4 active hardest Span-DTW negatives;
- local hard negatives available in addition to sequence negatives.

Canonical real positive pairs add cross-writer image-image supervision while keeping image-text supervision on both real lines.

Main code:
- `train.py`;
- `LossFunctionWithHelpers.py`;
- `Parameters.py`;
- `extra_real_training*.py`.

## 5. RealSyntheticBridge V2

The augmentation is architecture-independent. Each group contains:
- real anchor;
- synthetic positive containing 1-3 aligned islands plus unrelated distractors;
- white/black alignment mask for aligned/unaligned positive regions;
- guaranteed no-shared synthetic negatives.

The bridge-specific real-image/synthetic-text ranking is computed only on true shared islands. Full-line positive sequence ranking is not used for the V2 baseline because the synthetic positive deliberately contains unaligned distractors. The mask is propagated for diagnostics and later ablations, without adding a mask loss in the baseline.

Main code:
- `scripts/data/build_real_conditioned_synthetic_bridge.py`;
- `bridge_mask_runtime.py`;
- `bridge_multi_island_runtime.py`;
- `real_synthetic_bridge_training.py`.

## 6. Evaluation

Evaluation is image-to-image and does not require OCR:
- extract visual embeddings from both lines;
- compute image-window cosine similarity;
- run Smith-Waterman local alignment;
- save qualitative heatmaps/tracebacks;
- record score, path steps, matched fraction, path cosine, and bbox/localization metrics when available.

Main code:
- `Evaluation/eval_img_align_sw.py`;
- `Evaluation/eval_img_align_sw_no_png.py`;
- `Evaluation/sw_runner.py`.

---

# Code curriculum

| Stage | Purpose | Input | Main command/code | Output |
|---|---|---|---|---|
| S0 | preflight | branch/data | `submit_full_research_pipeline.sh` | frozen run metadata |
| S1 | synthetic pretraining | synthetic corpus | `run_branch_fixed63_synthetic.sh` | `<prefix>_synth` |
| S2 | qualitative real zero-shot | S1 | `run_stage_qualitative.sh` | heatmaps |
| S3 | quantitative real zero-shot | S1 | `run_stage_quantitative.sh` | baseline metrics |
| S4 | real FT without augmentation | S1 | `run_stage_real_finetune.sh` | `<prefix>_real` |
| S5 | qualitative post-real | S4 | `run_stage_qualitative.sh` | heatmaps |
| S6 | quantitative post-real | S4 | `run_stage_quantitative.sh` | metrics |
| S7 | build/audit Bridge V2 | real train-safe data | `prepare_real_synthetic_bridge_v2.sh` | augmentation corpus |
| S8 | bridge zero-shot evaluation | S4 | `run_stage_bridge_eval.sh` | pre-FT bridge metrics |
| S9 | bridge FT | S4 | `run_real_synthetic_bridge.sh` | best bridge checkpoint |
| S10 | post-bridge evaluations | S9 best | stage evaluation scripts | comparison metrics |
| S11 | all-real final | S9 best | `run_stage_final_all_real.sh` | complete-real summary |

For architecture comparison, do not change ViT depth/heads or data/evaluation protocol mid-pipeline unless the run is explicitly labeled as a separate architecture ablation.
