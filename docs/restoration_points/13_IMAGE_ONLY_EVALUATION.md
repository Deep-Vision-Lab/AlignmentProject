# Point 13 — Image-only final evaluation

## Goal
Keep the final manuscript-to-manuscript evaluation image-only. Both lines must use the same local encoder, context model, and fusion layer learned during training, without requiring text embeddings at inference time.

## Run
```bash
bash scripts/restoration_points/13_image_only_evaluation.sh
```

No flags are required.

## What the script checks
It encodes two RGB image lines through the shared model path, obtains local and fused window vectors, and builds an image-image similarity matrix directly from the fused vectors. No text input is supplied to this evaluation path.

## Main code involved
- `vlm_restoration_positive_dtw.py`
- `embeddingModel.py`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; both lines yield valid fused vectors and their image-image similarity matrix contains finite values.

## Full dataset evaluation
After training, use:
```bash
bash scripts/eval_yelda_restoration_positive_dtw.sh
```
