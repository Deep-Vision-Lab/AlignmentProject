# Full synthetic-data diagnosis

To run the complete diagnostic workflow on an actual synthetic pair and compare
local restoration, Stage A, Stage B, context, fusion, DTW, and image-image
similarity, run:

```bash
The historical full-suite launcher has been retired.
```

No flags are required. The main result is written to
`Results/Diagnostics/restoration_points/synthetic_suite/index_<N>/DIAGNOSIS.md`.

See `docs/restoration_points/SYNTHETIC_DIAGNOSTIC_SUITE.md` for the complete
output map.

---

# Real synthetic line first

Before the controlled point checks, you can inspect the actual preprocessing and
window slicing on a real synthetic line:

```bash
The historical synthetic-line launcher has been retired.
```

The script automatically prefers `DataSet/Synthetic63/images/img1_132.png`,
then other available synthetic images. It saves the original line, detected crop,
processed RGB line, window boundaries, and both physical and Arabic logical
window contact sheets under:

```text
Results/Diagnostics/restoration_points/synthetic_line/
```

For a full **trained-model** window-by-window diagnostic on the restoration
branch's synthetic line, use:

```bash
The historical trained-line launcher has been retired.
```

That second command requires a compatible trained restoration checkpoint.

---

# Viewing the Restoration Point Results

Branch: `agent/restoration-positive-dtw-window-encoder`

Every point is still run with **no user flags**. Each command writes its visible
results to:

```text
Results/Diagnostics/restoration_points/point_XX/
```

Start by opening `summary.txt` in the point folder. Then inspect the PNG files.

## Point 01 — Crop geometry

Run:

```bash
The historical Point 01 launcher has been retired.
```

Inspect:

- `01_original_line.png` — line before crop.
- `02_detected_crop_box.png` — red rectangle shows the detected content crop.
- `03_cropped_resized_rgb.png` — actual RGB image that continues to the model.
- `geometry.json` — crop offsets, resize scale, padding offsets and dimensions.
- `test_result.txt` — automated assertion result.

What you should see: outer blank margin removed, internal blank space kept, the
small upper dot retained, and RGB/color information preserved.

## Point 02 — Complete original reconstruction target

Run:

```bash
The historical Point 02 launcher has been retired.
```

Inspect:

- `01_model_input_line.png`
- `02_target_vs_current_reconstruction.png`
- `summary.txt`

Each row shows the exact RGB target beside the current decoder output. The target
must be the complete physical window. The decoder picture is only a structural
diagnostic unless trained weights are being used.

## Point 03 — Pretrained Restormer

Run:

```bash
The historical Point 03 launcher has been retired.
```

Inspect:

- `summary.txt`
- `01_input_vs_pretrained_output.png` when Restormer assets are installed.

If the official Restormer checkout/checkpoint is missing, `summary.txt` says
`STATUS: SKIPPED` and explains exactly what is missing. If installed, compare
the input and pretrained output yourself before deciding whether Restormer is a
good initialization.

## Point 04 — Preserve fine spatial detail

Run:

```bash
The historical Point 04 launcher has been retired.
```

Inspect:

- `01_spatial_resolution.png` — height and width through the CNN blocks.
- `02_final_8x8_feature_map.png` — final local spatial feature map.
- `spatial_shapes.json`.

What you should see: width remains 32 through the first two downsampling blocks,
then decreases more gradually to 8 rather than collapsing early.

## Point 05 — Reconstruction depends on encoded features

Run:

```bash
The historical Point 05 launcher has been retired.
```

Inspect:

- `01_feature_swap_reconstruction.png`
- `summary.txt`

The first row is decoder output for feature A and feature B. The second row is
after exchanging those features. The outputs should exchange/change with the
features, proving the decoder cannot bypass the encoded vector.

## Point 06 — Sequence context

Run:

```bash
The historical Point 06 launcher has been retired.
```

Inspect:

- `01_neighbor_influence.png`
- `context_change_per_token.csv`

Window 0 is deliberately changed. The plot shows how much every contextual
window vector changes. Non-zero bars away from window 0 demonstrate actual
neighbor/context influence.

## Point 07 — Local + contextual fusion

Run:

```bash
The historical Point 07 launcher has been retired.
```

Inspect:

- `01_fusion_dependency.png`
- `summary.txt`

The plot separately shows how much the final fused vector changes when only the
local input changes and when only the contextual input changes. Both must have an
effect. The summary also prints the fused-vector norms, which should be near 1.

## Point 08 — Reconstruction + contrastive losses

Run:

```bash
The historical Point 08 launcher has been retired.
```

Inspect:

- `01_loss_signals.png`
- `summary.txt`

The collapsed reconstruction should have a larger restoration loss than the
faithful reconstruction. The summary also shows the contrastive loss and its
gradient on the positive DTW cost.

## Point 09 — DTW transition semantics

Run:

```bash
The historical Point 09 launcher has been retired.
```

Inspect:

- `01_many_windows_to_one_letter.png`
- `02_one_window_to_many_letters.png`
- `summary.txt`

The route overlays let you visually verify that repeated windows can map to one
letter and, when required, one window can span several text positions.

## Point 10 — DTW route recomputation

Run:

```bash
The historical Point 10 launcher has been retired.
```

Inspect:

- `01_cost_matrix_A_route.png`
- `02_cost_matrix_B_route.png`
- `summary.txt`

The two cost matrices are deliberately different. Their routes should also be
different. This is the visual proof that the route follows the current cost
matrix rather than being forced to stay fixed or forced to change arbitrarily.

## Point 11 — Multiple vectors per physical window

Run:

```bash
The historical Point 11 launcher has been retired.
```

Inspect:

- `01_four_spatial_vectors_cosine.png`
- `summary.txt`

This shows the cosine similarity among K=4 optional spatial vectors extracted
from one physical window. It is an ablation only; the main model still uses one
fused vector per physical window.

## Point 12 — Eight-window tiny overfit

Run:

```bash
The historical Point 12 launcher has been retired.
```

Inspect:

- `01_training_line.png`
- `02_target_restored_swapped.png`
- `03_loss_curve.png`
- `losses.csv`
- `summary.txt`

This is one of the most important visual gates. The restored row should contain
different images instead of one repeated template. The swapped row should react
when the first and last encoded features are exchanged. The loss curve should
fall strongly.

Do not launch full restoration training if this point fails visually, even if a
single numerical metric looks acceptable.

## Point 13 — Image-only representation

Run:

```bash
The historical Point 13 launcher has been retired.
```

Inspect:

- `01_line_A.png`
- `02_line_B.png`
- `03_image_image_similarity.png`
- `summary.txt`

The similarity matrix is built directly from the two lines' fused image vectors;
no text embedding is supplied.

This point proves the inference path is image-only. The no-flag point diagnostic
uses fresh diagnostic weights, so it is not a final accuracy claim. For actual
trained-model alignment quality, run:

```bash
The historical image-only evaluation launcher has been retired.
```

and inspect the normal evaluation outputs.

## Run all checks

After inspecting the points individually:

```bash
The historical all-points launcher has been retired.
```

For model development, it is better to run points individually first, especially
01, 03, 05, 09, 10 and 12.
