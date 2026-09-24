# Restoration Recommendation Checklist

Branch: `agent/restoration-positive-dtw-window-encoder`

The following checklist describes historical diagnostics. Their shell launchers
have been retired; the descriptions are retained as experiment documentation.

| # | Recommendation implemented / checked | Script | What the script verifies |
|---|---|---|---|
| 1 | Crop only outer blank margins, preserve internal spaces/RGB, save geometry | Retired | Builds an RGB line with two separated content regions and a tiny dot; verifies crop safety margin, internal gap retention, color preservation, scale and offsets. |
| 2 | Reconstruct the complete original window | Retired | Verifies the reconstruction target is exactly the de-normalized 3-channel window cut from the model input, not a stroke/binary map. |
| 3 | Evaluate a known pretrained restoration model | Retired | If official Restormer code + checkpoint are present, loads them, captures the latent encoder feature, verifies two manuscript-like windows are distinct, and performs one identity-restoration fine-tuning step. Missing third-party assets produce an explicit SKIP. |
| 4 | Preserve fine spatial detail | Retired | Checks the local CNN keeps width at 32 through the first two downsampling blocks and reaches an 8x8 feature map rather than the old 8x2 map. |
| 5 | Reconstruction must depend on the encoded vector | Retired | Confirms the decoder accepts only local vectors and that swapping two vectors swaps/changes their reconstructions; there is no image skip path. |
| 6 | Build sequence context | Retired | Changes one neighboring token and confirms the Transformer changes another window's contextual representation. |
| 7 | Fuse local and contextual vectors before DTW | Retired | Confirms concat+projection uses both inputs and returns a normalized final alignment vector. |
| 8 | Train with reconstruction + positive/negative alignment supervision | Retired | Confirms negative-margin loss backpropagates and faithful RGB reconstruction has lower loss than a collapsed same-template prediction. |
| 9 | Verify DTW transition semantics | Retired | Tests multiple windows→one letter (vertical transitions) and one window→multiple letters when required (horizontal transitions). |
| 10 | Recompute DTW from current features; do not force route changes | Retired | Supplies two different cost matrices and confirms the recomputed hard route changes naturally. |
| 11 | Optional multiple vectors per window | Retired | Checks the optional spatial-vector API returns K=4 normalized vectors for each window without making it the default architecture. |
| 12 | Small overfit gate | Retired | Actually trains the local encoder+decoder on eight distinct windows; requires loss reduction, non-collapsed outputs and sensitivity to swapped features. |
| 13 | Final evaluation remains image-only | Retired | Encodes two image lines with the shared local/context/fusion path and builds an image-image similarity matrix without a text input. |

The former all-points launcher has been retired.

## Training after the checks

Stage A:

The former Stage A launcher has been retired.

Stage B (after Stage A produces `model_best.pth`):

The former Stage B launcher has been retired.

Image-only evaluation after Stage B:

The former image-only evaluation launcher has been retired.

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
