# Evaluation

This directory is intentionally limited to the evaluation paths still used by the ViT experiments and the shared real-data benchmarks.

## Public commands

### Unified Needleman-Wunsch diagnostic

`eval_img_align_nw_diagnostic.py` is the single qualitative/debugging entry point when you want to inspect the actual image-image NW alignment. It reads evaluation defaults from `Parameters.py` and requires only the dataset and trained weights.

Synthetic example:

```bash
python Evaluation/eval_img_align_nw_diagnostic.py \
  --dataset "$PWD/DataSet/Synthetic63" \
  --weights "$PWD/Weights/<job_id>/model_best.pth" \
  --n-samples 1
```

Real ArabicDataset example:

```bash
python Evaluation/eval_img_align_nw_diagnostic.py \
  --dataset "$PWD/DataSet/ArabicDataset" \
  --weights "$PWD/Weights/<job_id>/model_best.pth" \
  --n-samples 1
```

Native real-synthetic Bridge V3 example:

```bash
python Evaluation/eval_img_align_nw_diagnostic.py \
  --dataset "$PWD/DataSet/RealSyntheticBridge_v3" \
  --weights "$PWD/Weights/<job_id>/model_best.pth" \
  --n-samples 1
```

The evaluator auto-detects synthetic, ordinary real, explicit real/injection split manifests, the native `RealSyntheticBridge_v3` layout, mixed roots, and generic pair manifests. For each pair it saves both masked lines, a value-annotated cosine matrix, a value-annotated NW match-score matrix, a value-annotated accumulated DP matrix, the terminal-to-origin NW traceback, predicted binary masks, numeric matrices, and per-trace window scores. Synthetic/Bridge masks are scored with IoU/Dice when ground truth is available.

### Fixed-63 synthetic Needleman-Wunsch

Use the tuned component-aware NW wrapper for the held-out `AugmentedArabicDataset63` split:

```bash
WEIGHTS="$PWD/Weights/<run>/model_best.pth" \
bash Evaluation/evaluate_synthetic_fixed63_nw_balanced.sh
```

This delegates to `run_fixed63_vit_eval.sh` with `EVAL_MODE=nw` and uses `vit_fixed63_nw_eval.py` for the component-aware NW evaluation.

### Fixed-63 synthetic mask metrics

```bash
WEIGHTS="$PWD/Weights/<run>/model_best.pth" \
bash Evaluation/evaluate_synthetic_fixed63_mask_metrics.sh
```

Reports precision, recall, IoU, Dice, and F1 against the synthetic alignment masks through the same fixed-63 NW engine.

### Real qualitative / bbox evaluation

```bash
WEIGHTS="$PWD/Weights/<run>/model_best.pth" \
bash Evaluation/evaluate.sh
```

### Real quantitative benchmark

```bash
WEIGHTS="$PWD/Weights/<run>/model_best.pth" \
bash Evaluation/evaluate_quantitative.sh
```

This includes real crop localization, image-pair retrieval/discrimination, and optional sparse-interval evaluation.

### Transcript-supervised quantitative benchmark

```bash
WEIGHTS="$PWD/Weights/<run>/model_best.pth" \
bash Evaluation/evaluate_transcript_quantitative.sh
```

## Internal modules

Do not invoke these directly unless debugging:

- `_eval_utils.py`: checkpoint reconstruction, feature extraction, and shared DP utilities.
- `run_fixed63_vit_eval.sh`, `fixed63_test_manifest.py`, `vit_fixed63_nw_eval.py`: fixed-63 ViT NW and mask evaluation stack.
- `eval_img_align_sw.py`, `sw_runner.py`, `sw_core.py`, `sw_dataset.py`: Smith-Waterman / real-data evaluation stack.
- `zero_shot_sw.py`, `window_alignment.py`: preprocessing, ink-aware scoring, and score normalization shared by evaluation paths.
- `vit_evaluation.py`: ViT checkpoint reconstruction while retaining checkpoint-format compatibility.
- `quantitative_real.py`: real quantitative benchmark engine.
- `transcript_quantitative.py`: transcript-supervised benchmark engine.
- `real_subword_box_geometry.py`, `real_subword_box_json.py`, `real_subword_box_metrics.py`, `real_subword_box_patch.py`: real bbox annotation and metric support.
- `__init__.py`: package marker.

Removed legacy/duplicate entrypoints should be recovered from Git history rather than reintroduced into this directory.
