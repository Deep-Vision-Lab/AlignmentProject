# Evaluation

There are two public real-data commands.

## Qualitative / per-pair Smith-Waterman evaluation

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate.sh
```

This produces pair visualizations and per-sample Smith-Waterman diagnostics under:

```text
Results/Evaluation/<model>/Real_Experiments/<run>/<label>/
```

## Quantitative real-data evaluation

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate_quantitative.sh
```

This benchmark does not require dense masks. It runs:

1. **Real crop localization** with known crop coordinates, interval IoU, boundary MAE, center error, and Success@IoU thresholds.
2. **Real pair retrieval** with Recall@1/5/10, MRR, and mAP.
3. **Pair discrimination** with AUROC, average precision, and F1 using a separate calibration subset for threshold selection.
4. **Optional sparse interval evaluation** when `INTERVAL_MANIFEST` is supplied.

Default output:

```text
Results/Evaluation/<model>/Real_Quantitative/<run>/
├── crop_localization.csv
├── retrieval.csv
├── pair_scores.csv
├── sparse_intervals.csv      # only when annotations are supplied
├── summary.json
└── report.md
```

Useful overrides:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
CROP_LINES=100 \
CROPS_PER_LINE=3 \
RETRIEVAL_QUERIES=100 \
RETRIEVAL_POOL_SIZE=50 \
EVAL_SEED=42 \
bash Evaluation/evaluate_quantitative.sh
```

For a fast smoke test:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
CROP_LINES=5 \
CROPS_PER_LINE=1 \
RETRIEVAL_QUERIES=5 \
RETRIEVAL_POOL_SIZE=5 \
bash Evaluation/evaluate_quantitative.sh
```

An optional sparse interval manifest can be CSV, JSON, or JSONL. Coordinates must be measured on the preprocessed `1024x128` evaluation canvas. Required fields are:

```text
image1,image2,line1_start_px,line1_end_px,line2_start_px,line2_end_px
```

Optional field:

```text
pair_id
```

Run with:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
INTERVAL_MANIFEST="$PWD/DataSet/ArabicDataset/sparse_intervals.csv" \
bash Evaluation/evaluate_quantitative.sh
```

Both shell launchers submit their own one-GPU Slurm jobs. Run them from the repository root on the login node; do not wrap them with `sbatch`.

## Internal modules

Do not run the Python files directly:

- `eval_img_align_sw.py`: qualitative/per-pair Python entrypoint.
- `quantitative_real.py`: quantitative benchmark engine.
- `_eval_utils.py`: checkpoint loading and feature extraction.
- `sw_runner.py`: qualitative CLI execution and report writing.
- `sw_core.py`: Smith-Waterman scoring and visualization.
- `sw_dataset.py`: real-pair loading and split handling.
- `zero_shot_sw.py`: real-image preprocessing and ink-aware scoring.
- `vit_evaluation.py`: ViT checkpoint reconstruction while retaining CNN compatibility.
- `window_alignment.py`: score-matrix normalization used by Smith-Waterman.
- `__init__.py`: makes `Evaluation` importable as a package.
