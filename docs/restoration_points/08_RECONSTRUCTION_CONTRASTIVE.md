# Point 08 — Reconstruction + contrastive alignment losses

## Goal
Train the image representation to preserve visual detail while also learning positive image-text correspondence and separation from negatives.

## Run
```bash
bash scripts/restoration_points/08_reconstruction_contrastive.sh
```

No flags are required.

## What the script checks
It verifies two independent training signals:
- the negative-margin contrastive term produces a positive loss and a gradient on the positive alignment cost;
- faithful RGB reconstruction is scored lower than a collapsed same-template prediction.

## Main code involved
- `restoration_recommended_components.py` — contrastive margin helper
- `vlm_restoration_positive_dtw.py` — restoration loss
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; both the contrastive gradient and reconstruction-quality assertions succeed.
