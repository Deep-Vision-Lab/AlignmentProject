# Restoration Recommendation Checklist

Branch: `agent/restoration-positive-dtw-window-encoder`

Every numbered check below is runnable with **no command-line flags**. Run from
the project root. The scripts stop on a failed assertion.

| # | Recommendation implemented / checked | Script | What the script verifies |
|---|---|---|---|
| 1 | Crop only outer blank margins, preserve internal spaces/RGB, save geometry | `scripts/restoration_points/01_crop_geometry.sh` | Builds an RGB line with two separated content regions and a tiny dot; verifies crop safety margin, internal gap retention, color preservation, scale and offsets. |
| 2 | Reconstruct the complete original window | `scripts/restoration_points/02_original_window_target.sh` | Verifies the reconstruction target is exactly the de-normalized 3-channel window cut from the model input, not a stroke/binary map. |
| 3 | Evaluate a known pretrained restoration model | `scripts/restoration_points/03_pretrained_restormer.sh` | If official Restormer code + checkpoint are present, loads them, captures the latent encoder feature, verifies two manuscript-like windows are distinct, and performs one identity-restoration fine-tuning step. Missing third-party assets produce an explicit SKIP. |
| 4 | Preserve fine spatial detail | `scripts/restoration_points/04_preserve_fine_detail.sh` | Checks the local CNN keeps width at 32 through the first two downsampling blocks and reaches an 8x8 feature map rather than the old 8x2 map. |
| 5 | Reconstruction must depend on the encoded vector | `scripts/restoration_points/05_feature_dependency.sh` | Confirms the decoder accepts only local vectors and that swapping two vectors swaps/changes their reconstructions; there is no image skip path. |
| 6 | Build sequence context | `scripts/restoration_points/06_sequence_context.sh` | Changes one neighboring token and confirms the Transformer changes another window's contextual representation. |
| 7 | Fuse local and contextual vectors before DTW | `scripts/restoration_points/07_local_context_fusion.sh` | Confirms concat+projection uses both inputs and returns a normalized final alignment vector. |
| 8 | Train with reconstruction + positive/negative alignment supervision | `scripts/restoration_points/08_reconstruction_contrastive.sh` | Confirms negative-margin loss backpropagates and faithful RGB reconstruction has lower loss than a collapsed same-template prediction. |
| 9 | Verify DTW transition semantics | `scripts/restoration_points/09_dtw_transitions.sh` | Tests multiple windows→one letter (vertical transitions) and one window→multiple letters when required (horizontal transitions). |
| 10 | Recompute DTW from current features; do not force route changes | `scripts/restoration_points/10_dtw_recompute.sh` | Supplies two different cost matrices and confirms the recomputed hard route changes naturally. |
| 11 | Optional multiple vectors per window | `scripts/restoration_points/11_multi_vector_ablation.sh` | Checks the optional spatial-vector API returns K=4 normalized vectors for each window without making it the default architecture. |
| 12 | Small overfit gate | `scripts/restoration_points/12_small_overfit.sh` | Actually trains the local encoder+decoder on eight distinct windows; requires loss reduction, non-collapsed outputs and sensitivity to swapped features. |
| 13 | Final evaluation remains image-only | `scripts/restoration_points/13_image_only_evaluation.sh` | Encodes two image lines with the shared local/context/fusion path and builds an image-image similarity matrix without a text input. |

Run everything in order:

```bash
bash scripts/restoration_points/run_all.sh
```

## Training after the checks

Stage A:

```bash
bash scripts/train_restoration_stage_a.sh
```

Stage B (after Stage A produces `model_best.pth`):

```bash
bash scripts/train_restoration_stage_b.sh
```

Image-only evaluation after Stage B:

```bash
bash scripts/eval_yelda_restoration_positive_dtw.sh
```

## Important interpretation rules

- Point 3 is an evaluation gate. Restormer is **not** promoted to the default
  encoder merely because it is pretrained; it should replace the local baseline
  only if manuscript-window reconstruction and feature diagnostics improve.
- Point 10 does not demand a different DTW route every epoch. The route is
  recomputed every forward pass, while epoch diagnostics track matrix changes,
  parameter updates and route overlap.
- Point 11 is intentionally an ablation. The main baseline stays one fused vector
  per physical window until diagnostics show that one vector is insufficient.
- Point 13 uses the same fused representation used by DTW training, but does not
  load or require the frozen text codebook during image-image alignment.
