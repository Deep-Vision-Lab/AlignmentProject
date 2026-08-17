# Model and Code Stages — ViT Branch

Branch: `agent/use-vit-encoder`

Backend: `vit` selected by `model_backend.py`.

## Model

### Input and local sequence

Every manuscript line is normalized to the shared `128 x 1024` canvas and represented as an ordered sequence of overlapping horizontal observations. The standard local window width is 32 px. `unified_line_geometry.py`, `DataLoader.py`, `RealDataSet.py`, and `training_runtime/entrypoint.py` keep training/evaluation geometry consistent.

### ViT visual encoder

`model_backend.py` constructs the visual model through `vit_embedding_model.py`.

The ViT branch replaces the CNN/BiLSTM visual encoder with the branch Transformer representation while keeping the same training data, text supervision, loss family, and evaluation protocol. Window/patch observations are projected into the project's shared image/text embedding dimension and contextualized by Transformer layers. The controlled comparison therefore asks whether Transformer visual sequence modeling learns better local manuscript alignment geometry than the CNN+BiLSTM baseline.

### Arabic text side

The text encoder is the same as in the other branches: frozen AraBERT-v02 backbone features plus the existing shared-space projection/normalization and learned special embeddings. Text spans are enumerated over visible Arabic units and compared to visual observations.

Main code: `arabic_span_text_encoder.py`, `arabic_span_text_encoder_legacy.py`, and `bridge_frozen_text.py`.

### Image-text alignment

For each line:
1. extract ViT visual embeddings;
2. encode Arabic spans;
3. normalize both sides;
4. build the image-span similarity matrix;
5. apply differentiable Span-DTW;
6. use 10 negative text sequences by default;
7. use the hardest four negative candidates for the expensive Span-DTW negative computation;
8. optionally apply local hard-negative and variance regularization shared with the other branches.

Main code: `train.py`, `LossFunctionWithHelpers.py`, `Parameters.py`, and `training_runtime/`.

## RealSyntheticBridge V2

Bridge V2 is the **only real-domain adaptation stage** after synthetic pretraining in the active protocol. There is no separate canonical-real fine-tuning block.

Each Bridge group contains:
- a genuine real manuscript anchor;
- a synthetic positive with 1-3 ordered shared islands;
- unrelated positive distractor content;
- a white/black shared-region mask;
- guaranteed no-shared synthetic negatives.

Training signals:
- real anchor ↔ its own transcript;
- synthetic line ↔ its own transcript;
- image-image positive/negative discrimination;
- real-image ↔ synthetic-text ranking restricted to the actual shared islands.

Generic whole-positive-line sequence ranking is disabled because Bridge V2 positives intentionally contain distractors. Bridge training defaults to a maximum of 15 epochs at `1e-6`, validates every epoch, and all later evaluations use `checkpoint_best_val.pth`.

Main Bridge code:
- `scripts/data/build_real_conditioned_synthetic_bridge.py`;
- `scripts/data/prepare_real_synthetic_bridge_v2.sh`;
- `bridge_mask_runtime.py`;
- `bridge_multi_island_runtime.py`;
- `real_synthetic_bridge_training.py`;
- `scripts/train/run_real_synthetic_bridge.sh`.

## Image-only evaluation

Evaluation uses only the visual branch:
1. preprocess both manuscript lines;
2. extract ViT local/contextual features;
3. compute image-window × image-window cosine similarity;
4. run Smith-Waterman local alignment;
5. save qualitative heatmaps/path overlays;
6. compute score, path steps, matched fraction, path cosine, discrimination AUC/AP, and bbox/localization metrics where available.

Main code: `Evaluation/eval_img_align_sw.py`, `Evaluation/eval_img_align_sw_no_png.py`, `Evaluation/sw_runner.py`, and `scripts/eval/`.

---

# Active code curriculum

| Stage | Purpose | Input | Main code | Output |
|---|---|---|---|---|
| D0 | build/freeze Bridge V2 before models | real train anchors | `submit_bridge_v2_dataset.sh`, `prepare_real_synthetic_bridge_v2.sh` | frozen `RealSyntheticBridge_v2` |
| S1 | synthetic pretraining | synthetic fixed-63 corpus | `run_branch_fixed63_synthetic.sh` | synthetic checkpoint |
| S2 | qualitative zero-shot real | S1 | `run_stage_qualitative.sh` | heatmaps/paths |
| S3 | quantitative zero-shot real | S1 | `run_stage_quantitative.sh` | baseline real metrics |
| S4 | Bridge pre-finetune evaluation | S1 + frozen Bridge V2 | `run_stage_bridge_eval.sh` | pre-training Bridge metrics |
| S5 | direct Bridge V2 adaptation | S1 | `run_real_synthetic_bridge.sh` | `checkpoint_best_val.pth` |
| S6 | post-Bridge qualitative real | S5 best | `run_stage_qualitative.sh` | heatmaps/paths |
| S7 | post-Bridge quantitative + Bridge eval | S5 best | quantitative + Bridge eval scripts | comparison metrics |
| S8 | final complete-real evaluation | S5 best | `run_stage_final_all_real.sh` | full-real CSVs + `final_summary.json` |

The same frozen Bridge V2 dataset must be reused for CNN, ViT, and DINOv3 architecture comparisons.
