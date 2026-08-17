# Model and Code Stages — DINOv3 ConvNeXt Branch

Branch: `agent/use-dinov3-convnext`

Backend: `dinov3_convnext` selected by `model_backend.py`.

## Model

### Input and local sequence

Every manuscript line is normalized to the shared `128 x 1024` canvas and represented as an ordered sequence of overlapping horizontal observations. The standard local window width is 32 px. `unified_line_geometry.py`, `DataLoader.py`, `RealDataSet.py`, and `training_runtime/entrypoint.py` keep geometry identical to the CNN and ViT branches.

### DINOv3 ConvNeXt visual encoder

`model_backend.py` constructs the visual encoder through `dinov3_convnext_embedding_model.py`.

Canonical design:
- official Meta DINOv3 ConvNeXt-Tiny local visual backbone;
- projection from foundation-model features into the project's shared image/text embedding dimension;
- `DINOV3_FREEZE_BACKBONE=1` for the first controlled comparison;
- optional `USE_BILSTM=0/1` sequence context on top of DINOv3 local features;
- optional local neighboring-window grouping;
- window chunking to control GPU memory.

For initial S1 training set both:

```bash
export DINOV3_REPO_DIR=/path/to/local/dinov3
export DINOV3_WEIGHTS=/path/to/authorized/dinov3_convnext_tiny_weights
```

Later AlignmentProject checkpoints contain the learned DINO state, but the local official DINO repository is still needed to construct the architecture.

### Arabic text side

The text encoder is shared with the other branches: frozen AraBERT-v02 backbone features plus the existing shared-space projection/normalization and learned special embeddings. Arabic spans are compared with DINOv3 visual observations through the same shared-space objective.

### Image-text alignment

For each line:
1. extract DINOv3 local/contextual embeddings;
2. encode Arabic spans;
3. normalize image/text vectors;
4. build the similarity matrix;
5. apply differentiable Span-DTW;
6. use 10 negative text sequences by default;
7. keep the hardest four candidates active for expensive negative Span-DTW;
8. use the same optional local hard-negative and variance regularization as the other branches.

Main code: `train.py`, `LossFunctionWithHelpers.py`, `Parameters.py`, and `training_runtime/`.

## RealSyntheticBridge V2

Bridge V2 is the **only real-domain adaptation stage** after synthetic pretraining in the active protocol. There is no separate canonical-real fine-tuning block.

Each Bridge group contains:
- a genuine real manuscript anchor;
- a synthetic positive with 1-3 ordered shared islands;
- unrelated positive distractor regions;
- a white/black shared-region mask;
- guaranteed no-shared synthetic negatives.

Training signals:
- real anchor ↔ its own transcript;
- synthetic line ↔ its own transcript;
- image-image positive/negative discrimination;
- real-image ↔ synthetic-text ranking restricted to actual shared islands.

Generic whole-positive-line sequence ranking is disabled because Bridge V2 positives intentionally contain distractors. Bridge training defaults to a maximum of 15 epochs at `1e-6`, validates every epoch, and all later evaluations use `checkpoint_best_val.pth`.

Main Bridge code:
- `scripts/data/build_real_conditioned_synthetic_bridge.py`;
- `scripts/data/prepare_real_synthetic_bridge_v2.sh`;
- `bridge_mask_runtime.py`;
- `bridge_multi_island_runtime.py`;
- `real_synthetic_bridge_training.py`;
- `scripts/train/run_real_synthetic_bridge.sh`.

## Image-only evaluation

Evaluation does not use OCR/text inference:
1. preprocess both real manuscript lines;
2. extract DINOv3 local/contextual visual features;
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
