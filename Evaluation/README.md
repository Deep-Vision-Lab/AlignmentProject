# Evaluation

The ViT experiment branches expose two active evaluation entry points.

## Image-image Needleman-Wunsch diagnostic

```bash
python Evaluation/eval_img_align_nw_diagnostic.py \
  --dataset <dataset-root-or-manifest> \
  --weights <Weights/job_name/model_best.pth>
```

This supports synthetic, real ArabicDataset, explicit real split manifests, generic pair manifests, and RealSyntheticBridge V2/V3 layouts. It uses `Parameters.py` for defaults and saves masked line overlays, value-annotated cosine and NW score/DP heatmaps, traceback evidence, numeric matrices, predicted masks, and mask metrics when ground truth is available.

## Window-to-token embedding / PCA diagnostic

```bash
python Evaluation/eval_window_token_pca.py \
  --dataset "$PWD/DataSet/Synthetic63" \
  --weights "$PWD/Weights/<job_name>/model_best.pth" \
  --n-samples 10
```

This diagnostic tests whether image-window embeddings occupy the same shared representation space as Arabic one-character text targets. For every window it computes direct cosine similarity against the token prototypes and records Top-1/Top-k nearest letters and the Top-1 versus Top-2 margin.

It saves two complementary views:

- **core / pure-letter prototypes** — a one-character text embedding such as `ب`, useful for asking which literal Arabic letter a window is closest to;
- **contextual letter prototypes** — averages of the checkpoint text encoder's one-character contextual targets in the selected transcripts, closer to the representation used by current Span-DTW training.

Outputs include per-line core/context cosine heatmaps and CSV matrices, `windows_nearest_tokens.csv`, token prototype vectors, raw window embeddings, `pca_core_letters.png`, `pca_context_letters.png`, `pca_explained_variance.png`, and `summary.json`.

PCA is used only for visualization. Window colors are assigned by nearest token in the original shared embedding space before PCA projection.

Synthetic63 does **not** store independent per-character pixel bounding boxes. The optional hard Span-DTW token assignment is therefore reported only as **path consistency** and `summary.json` explicitly records `path_reference_is_ground_truth: false`. It must not be interpreted as ground-truth token classification accuracy.

Useful options:

```bash
--feature contextual|local|grouped
--side 1|2|both
--vocab-mode dataset|arabic|union
--top-k 5
--max-pca-windows 1500
--no-path-reference
```

## Internal support modules

These are implementation modules shared by the active evaluators and should not normally be invoked directly:

- `__init__.py` — package marker.
- `_eval_utils.py` — checkpoint reconstruction, image features, cosine similarity, and global NW DP.
- `vit_evaluation.py` — reconstructs canonical ViT checkpoints through the shared evaluation loader.
- `sw_core.py` — shared score-mode and match-score utilities reused by NW.
- `sw_dataset.py` — real/synthetic manifest discovery and display preprocessing helpers.
- `window_alignment.py` — shared window score normalization used by `sw_core.py`.
- `zero_shot_sw.py` — shared real preprocessing, dataset patches, and ink-aware score masking reused by NW.
- `trace_components.py` — supported-component extraction, pixel mask geometry, traceback rendering, and numeric evidence.

Older real-only NW, Smith-Waterman entry points, fixed-63 evaluators, transcript/quantitative benchmark stacks, and subword-box evaluators were removed from the active ViT branches. They remain recoverable from Git history if a historical experiment needs them.
