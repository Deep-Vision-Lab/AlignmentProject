## Restormer prerequisite

Point 03 is a true pretrained-model test and requires the official upstream
Restormer source plus an official checkpoint.

Run once from the repository root:

The former Restormer setup launcher has been retired.

This creates:

```text
third_party/Restormer/
Weights/Pretrained/Restormer/real_denoising.pth
```

The setup verifies that `real_denoising.pth` loads strictly into the official
3-channel Restormer architecture with `LayerNorm_type="BiasFree"`.

After that run:

The former Restormer probe launcher has been retired.

---

# Full synthetic-data restoration diagnostic

Branch: `agent/restoration-positive-dtw-window-encoder`

Run one command:

The former full-suite launcher has been retired.

No flags are required.

The suite automatically:

- prefers `DataSet/Synthetic63`, then `DataSet/Synthetic_Arabic`, then another available `Synthetic*/images` dataset;
- prefers complete pair index `132`, otherwise selects the first complete `img1/img2/text1/text2` pair;
- uses CUDA when available, otherwise CPU;
- keeps running after an individual diagnostic fails;
- records every result in `statuses.tsv`;
- produces a final `DIAGNOSIS.md`.

## What is tested on actual synthetic data

### Input/preprocessing

The suite first saves the actual selected synthetic line before and after:

```text
original RGB
  -> outer-foreground crop
  -> aspect-preserving resize
  -> artificial outer padding
  -> 128x32 windows, stride 16
```

Inspect:

```text
preprocessing/
  01_original_synthetic_line.png
  02_detected_crop_on_real_line.png
  03_cropped_resized_padded_line.png
  04_window_boundaries.png
  05_windows_physical_left_to_right.png
  06_windows_arabic_logical_right_to_left.png
```

### Local capacity on real synthetic windows

`real_tiny_overfit/` selects the most informative eight-window region from the
actual synthetic line and trains only the local CNN+decoder on those windows.

Inspect:

```text
real_tiny_overfit/
  02_selected_real_8window_region.png
  03_real_targets_restored_swapped.png
  04_loss_curve.png
  metrics.json
  summary.txt
```

If this fails, the problem is already before Transformer/context/DTW.

### Stage A checkpoint

If this checkpoint exists:

```text
Weights/restore_rgb_pretrain_s16/model_best.pth
```

the suite runs the full per-window diagnostic on both sides of the exact same
synthetic pair and a pair-level representation diagnostic.

Folders:

```text
stage_a_side1/
stage_a_side2/
stage_a_pair/
```

This tells us whether restoration pretraining itself learned distinct local
features.

### Stage B checkpoint

If this checkpoint exists:

```text
Weights/restore_fused_rgb_dtw_s16/model_best.pth
```

the exact same diagnostics are repeated:

```text
stage_b_side1/
stage_b_side2/
stage_b_pair/
```

The direct Stage-A versus Stage-B comparison is important. It can show whether
alignment training destroys a previously healthy restoration representation.

### Legacy restoration checkpoint

If present, the suite also analyzes:

```text
Weights/vit_restore_dtw_s16/model_best.pth
```

under:

```text
legacy_side1/
legacy_side2/
```

This is useful for comparing the older collapsed restoration behavior with the
new staged design.

## Pair-level trained-model diagnostics

For modern Stage-A/Stage-B checkpoints, the pair diagnostic uses the actual
synthetic `img1_INDEX` and `img2_INDEX` and saves:

```text
01_real_feature_swap.png
02_real_window_four_spatial_vectors.png
10_local_cross_line_similarity.png
10_context_cross_line_similarity.png
10_fused_cross_line_similarity.png
11_local_within_side1_similarity.png
11_context_within_side1_similarity.png
11_fused_within_side1_similarity.png
12_context_change_per_window.png
13_fusion_ablation.png
20_positive_transcript_dtw.png
21_negative_transcript_dtw.png
metrics.json
README.txt
```

These answer separate questions:

1. Does the decoder actually depend on the local vector?
2. Do local features collapse within one line?
3. Does Transformer context improve or damage the representation?
4. Does fusion use both local and contextual vectors?
5. Is image-image alignment better with local, context, or fused features?
6. Does the positive transcript obtain lower DTW cost than a real negative transcript?
7. Is the final evaluation representation image-only?

## Structural recommendation tests

The original 13 recommendation gates are still run. They verify graph and
algorithm properties independently of learned checkpoint quality.

The important difference is that the suite now adds actual synthetic-data
diagnostics around them. For example:

- Point 3 uses real synthetic windows when Restormer assets are available.
- Point 12 has an additional actual-synthetic eight-window overfit.
- restoration, context, fusion and DTW are evaluated again through the saved
  Stage-A/Stage-B checkpoints on the selected real synthetic pair.

## Final report

The main file to read after the run is:

```text
Results/Diagnostics/restoration_points/synthetic_suite/index_INDEX/DIAGNOSIS.md
```

It follows this failure hierarchy:

1. real tiny-overfit fails -> local encoder/decoder capacity or optimization;
2. tiny-overfit passes but Stage A collapses -> Stage-A training/data/loss;
3. Stage A works but Stage B worsens -> alignment/DTW training damages local representation;
4. local is healthy but context becomes worse -> Transformer context;
5. context is healthy but fusion becomes worse -> fusion head;
6. image vectors are healthy but positive/negative DTW ordering fails -> text grounding / contrastive DTW;
7. text DTW is healthy but paired image-image similarity is poor -> insufficient cross-image invariance.

The automatic thresholds are diagnostics only. Always open the PNGs around the
first stage flagged by the report.
