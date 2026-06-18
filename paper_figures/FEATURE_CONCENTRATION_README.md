# Feature Concentration Analysis — Interpretation Guide

## What is this?

`fig12_window_feature_concentration.py` runs three explainability methods on
individual sliding-window patches extracted from a trained Arabic line-image.
The goal is to understand *what* the model uses inside each patch when it
computes the cosine similarity with a text character embedding.

---

## Explanation Methods

| Method | What it measures |
|--------|-----------------|
| **Grad-CAM** | Which spatial regions in the patch had the strongest influence on the target score, weighted by gradient magnitude back through the full model (BiLSTM → CNN backbone). |
| **Saliency** | Pixel-level gradient of the target score w.r.t. the full input image, cropped to the window.  Faster than Grad-CAM; shows fine-grained pixel importance. |
| **Occlusion** | How much the target score drops when small rectangular patches are blanked out.  Slowest but most interpretable: "if I cover *this* region, the score falls by X". |

All maps are resized to match the display patch and blended as overlays
(hotter = more important).

---

## Target Scoring Modes

| Mode | Score used for backprop / masking |
|------|----------------------------------|
| `aligned_char` | cosine\_sim(window\_emb, **DTW-aligned char** embedding). Best for understanding letter-level attention. |
| `max_similarity` | max cosine\_sim over all text characters. Useful when alignment is unreliable. |
| `embedding_norm` | L2 norm of the window embedding. Image-only, no text branch needed. |

---

## Interpreting the Heatmaps

### High activation on dots / diacritics
**Good.** The model captures the discriminative detail that distinguishes, e.g.,
ب (one dot below) from ت (two dots above).  Dots are the most discriminative
features in many connected-script Arabic words.

### High activation on the main letter body / strokes
**Good.** The model learns the glyph structure (vertical strokes, curves, loops).
This is the baseline expected behaviour for a well-trained OCR-style feature extractor.

### High activation on inter-letter connection strokes
**Good.** The BiLSTM and 50% overlap windows help the model model the
contextual information carried by the *baseline* and connecting ligatures.
Overlapping windows are particularly important here because a single
connection stroke spans two adjacent patches.

### High activation on background, image borders, or whitespace
**Bad — shortcut learning.**
The model has latched onto image-level artefacts rather than glyph content.

Possible fixes:
- Stronger augmentation: random cropping, brightness jitter, salt-and-pepper noise.
- Normalise the line image vertically before training (crop to tight bounding box).
- Harder negatives: pairs with different content but similar background textures.

### High activation on letter body only, dots ignored
**Risky.**
The model may confuse character families that share a body but differ only
in dot count or position (ب / ت / ث / ن / ي).

Possible fixes:
- Use **multi-scale windows** (`multi_scale_enabled = True` in Parameters.py).
  Large windows capture body; small windows capture dots.
- Increase window size (`window_size = 32`) for single-scale training.
- Add **dot-confusion hard negatives** — pairs where the transcript differs only
  in dot count.

### High activation on the wrong region (neighbouring patch)
**Bad alignment.**
The DTW path is associating the window with a character that belongs to an
adjacent spatial region.

Possible fixes:
- Increase overlap (`stride_ratio = 0.25`, i.e. 75% overlap).
- Reduce the diagonal prior strength (`diagonal_prior_weight`).
- Add a blank / boundary token in the transcript (already done via `pad_text`).

---

## Example Commands

```bash
# Quick Grad-CAM on 6 auto-selected windows
python paper_figures/fig12_window_feature_concentration.py \
    --checkpoint Weights/JOB_ID/model_latest.pth \
    --data_dir   DataSet/Synthetic_Arabic_100000 \
    --sample_idx 1 \
    --method gradcam \
    --target_mode aligned_char

# Saliency on specific windows
python paper_figures/fig12_window_feature_concentration.py \
    --checkpoint Weights/JOB_ID/model_latest.pth \
    --data_dir   DataSet/Synthetic_Arabic_100000 \
    --sample_idx 1 \
    --window_indices 5 6 7 8 \
    --method saliency

# Occlusion (slower) with fine-grained patch
python paper_figures/fig12_window_feature_concentration.py \
    --checkpoint Weights/JOB_ID/model_latest.pth \
    --data_dir   DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0 \
    --method occlusion \
    --occlusion_patch_size 4 \
    --occlusion_stride 2

# All methods combined
python paper_figures/fig12_window_feature_concentration.py \
    --checkpoint Weights/JOB_ID/model_latest.pth \
    --data_dir   DataSet/Synthetic_Arabic_100000 \
    --sample_idx 1 \
    --method all \
    --num_windows 4 \
    --target_mode aligned_char
```

---

## Known Limitations

1. **Grad-CAM is at backbone[-1] (layer4).**  For window patches of size 16px,
   the feature map at layer4 is only 4×2 pixels, so spatial resolution is
   limited.  Upsample bilinearly to display at full patch resolution.

2. **Occlusion is slow.**  With `patch_size=8, stride=4` on a 16px-wide window,
   there are only ~6 positions, so it is fast.  For larger windows increase
   `--occlusion_stride`.

3. **sub-feature approximation.**  If `total_seq_len % total_windows != 0`
   (very rare with standard configs), sub-feature boundaries are approximated.
   A warning is printed in that case.
