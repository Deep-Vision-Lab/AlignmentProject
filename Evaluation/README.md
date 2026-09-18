# Evaluation

## Public command

For the current restoration / positive-DTW model, use **one public launcher only**:

```bash
WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
EVAL_MODE=all \
bash Evaluation/evaluate.sh
```

Supported modes:

```text
EVAL_MODE=qualitative
EVAL_MODE=quantitative
EVAL_MODE=all
```

Run the launcher from the repository root on the login node. It submits its own
one-GPU Slurm job. Do not wrap it in another `sbatch`.

The current default evaluation image policy is the one used for the recent
Point-2/Point-3 work:

- original RGB pixels;
- deterministic full-image resize to `1024x128`;
- no binarization;
- no foreground crop;
- no aspect-ratio padding;
- checkpoint window geometry (current model: physical `128x32` windows, stride `16`);
- `FEATURE=contextual`, which for the restoration checkpoint is the trained
  normalized local+context fused representation.

Override `REAL_DATA_DIR` and `ARABIC_MANIFEST` if the real dataset is stored
outside `DataSet/ArabicDataset`.

## Quantitative evaluation

`EVAL_MODE=quantitative` implements complementary tests rather than treating
one internal alignment score as ground-truth accuracy.

### 1. Controlled real-image crop localization

A horizontal crop is taken from a real manuscript line at known coordinates,
degraded, and localized back in the complete line.

Default crop fractions:

```text
10%, 20%, 30%, 40%, 50%
```

The default run creates five crops per selected line and evaluates every crop
under the configured `CROP_DEGRADATIONS`.

Reported metrics include:

- mean and median interval IoU;
- Success@IoU 0.30 / 0.50 / 0.70;
- boundary MAE in pixels and stride-16 windows;
- center error in pixels and windows.

**Limitation:** this is controlled same-line re-localization. It does not prove
localization between independently written manuscript lines.

### 2. Real pair retrieval and discrimination

The true partner is ranked against candidates from different pair identities.
Ranking uses a length-normalized score by default.

Reported retrieval metrics:

- Recall@1 / Recall@5 / Recall@10;
- MRR;
- mAP;
- actual mean candidate-pool size.

Pair discrimination reports:

- AUROC;
- AUPRC / average precision;
- equal error rate;
- precision / recall / F1 / accuracy.

The classification threshold is selected **only from the validation split** and
then applied unchanged to the test comparisons. Pair-ID groups remain intact in
the real-data split logic.

### 3. Sparse real cross-line interval benchmark

Provide `INTERVAL_MANIFEST` to evaluate true cross-line localization against
human spatial annotations.

Required fields:

```text
image1,image2,line1_start_px,line1_end_px,line2_start_px,line2_end_px
```

Optional:

```text
pair_id
```

Coordinates are measured on the `1024x128` evaluation canvas.

Reported metrics:

- line-1 and line-2 interval IoU;
- arithmetic mean two-line IoU;
- geometric JointIoU = sqrt(IoU1 * IoU2);
- boundary MAE in pixels and windows;
- center error;
- success when both lines reach IoU 0.30 / 0.50 / 0.75.

When available, this is the strongest direct real cross-line localization
benchmark in this evaluation stack.

### 4. Cycle consistency — diagnostic only

For a valid window `i` on line A, the evaluator finds its nearest window
`j` on line B and maps that window back to `i_hat` on line A.

Reported:

- mean / median cycle error in windows;
- normalized cycle error;
- percentage returning within 1 / 2 / 4 windows.

A consistently wrong correspondence can still have good cycle consistency.
Do not report this as localization accuracy.

### 5. Perturbation stability — diagnostic only

The evaluator perturbs one member of a real pair with:

- Gaussian blur;
- contrast;
- brightness;
- noise;
- small horizontal scale change;
- small vertical displacement;
- erosion;
- dilation.

Reported:

- normalized endpoint drift;
- predicted interval IoU before/after perturbation;
- path-cell IoU;
- SW-score coefficient of variation;
- path-length coefficient of variation.

This measures robustness, not correctness.

## Output contract

The quantitative directory contains:

```text
summary.json
report.md
per_sample.csv
crop_localization.csv
retrieval_results.csv
pair_classification.csv
validation_pair_classification.csv
sparse_intervals.csv          # only when INTERVAL_MANIFEST is supplied
cycle_consistency.csv
robustness.csv
```

Internal quantities such as `normalized_sw_score`, `mean_path_cosine`, or
matched/path fractions are retained for diagnosis and ranking. They must **not**
be presented as localization accuracy by themselves.

## Useful commands

Full quantitative evaluation:

```bash
WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
EVAL_MODE=quantitative \
bash Evaluation/evaluate.sh
```

Small smoke test:

```bash
WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
EVAL_MODE=quantitative \
CROP_LINES=3 \
CROPS_PER_LINE=1 \
CROP_FRACTIONS=0.30 \
CROP_DEGRADATIONS=blur \
RETRIEVAL_QUERIES=3 \
RETRIEVAL_POOL_SIZE=3 \
CALIBRATION_QUERIES=3 \
CYCLE_PAIRS=3 \
ROBUSTNESS_PAIRS=3 \
ROBUSTNESS_MODES=blur,noise \
bash Evaluation/evaluate.sh
```

Sparse interval run:

```bash
WEIGHTS="$PWD/Weights/res18_tinyvit_point2/model_latest.pth" \
EVAL_MODE=quantitative \
INTERVAL_MANIFEST="$PWD/DataSet/ArabicDataset/sparse_intervals.csv" \
bash Evaluation/evaluate.sh
```

## Internal evaluators

These support the public launcher or preserve earlier Point-2/Point-3
diagnostics. They are not the preferred public interface:

- `eval_img_align_sw.py` / `sw_runner.py` — per-pair Smith-Waterman evaluation.
- `quantitative_real.py` — external-target real-data benchmark engine.
- `quantitative_diagnostics.py` — cycle and perturbation diagnostics.
- `_eval_utils.py` — checkpoint reconstruction and feature extraction.
- `sw_core.py`, `sw_dataset.py`, `zero_shot_sw.py`, `window_alignment.py` — shared SW/runtime support.
- `eval_point2.py`, `point2_runtime.py`, comparison scripts — Point-2 architecture ablations.
- `eval_point3_hard_paths.py`, `point3_spatial_metrics.py` — earlier synthetic/GT spatial diagnostics.
- `eval_yelda.py` and Yelda support modules — previous NW evaluation artifacts.

## Interpretation hierarchy

Use the metrics according to what their targets actually prove:

1. **Sparse human intervals:** direct real cross-line localization correctness.
2. **Controlled crops:** exact-coordinate real-image localization, but same-line.
3. **Retrieval/discrimination:** real pair recognition.
4. **Cycle/perturbation:** self-consistency and robustness diagnostics.
5. **SW/NW/path cosine:** internal confidence/ranking diagnostics only.
