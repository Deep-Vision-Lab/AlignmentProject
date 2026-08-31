# Evaluation

The canonical evaluation entry point for the ViT experiment branches is:

```bash
python Evaluation/eval_img_align_nw_diagnostic.py \
  --dataset <dataset-root-or-manifest> \
  --weights <Weights/job_id/model_best.pth>
```

It supports synthetic, real ArabicDataset, explicit real split manifests, generic pair manifests, and RealSyntheticBridge V2/V3 layouts. It uses Parameters.py for defaults and saves masked line overlays, value-annotated cosine and NW score/DP heatmaps, traceback evidence, numeric matrices, predicted masks, and mask metrics when ground truth is available.

## Internal support modules

These are implementation modules used by the canonical NW evaluator and should not normally be invoked directly:

- `__init__.py` — package marker.
- `_eval_utils.py` — checkpoint reconstruction, image features, cosine similarity, and global NW DP.
- `vit_evaluation.py` — reconstructs ViT checkpoints through the shared evaluation loader.
- `sw_core.py` — shared score-mode and match-score utilities reused by NW.
- `sw_dataset.py` — real/synthetic manifest discovery and display preprocessing helpers.
- `window_alignment.py` — shared window score normalization used by `sw_core.py`.
- `zero_shot_sw.py` — shared real preprocessing, dataset patches, and ink-aware score masking reused by NW.
- `trace_components.py` — supported-component extraction, pixel mask geometry, traceback rendering, and numeric evidence.

Older real-only NW, Smith-Waterman entry points, fixed-63 evaluators, transcript/quantitative benchmark stacks, and subword-box evaluators were removed from the active ViT branches. They remain recoverable from Git history if a historical experiment needs them.
