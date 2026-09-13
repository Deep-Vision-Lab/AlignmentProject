# Point 02 — Original window reconstruction target

## Goal
Use the complete original RGB window as both the restoration input and reconstruction target. Do not mask, corrupt, binarize, or replace it with a stroke-only target for the baseline experiment.

## Run
```bash
bash scripts/restoration_points/02_original_window_target.sh
```

No flags are required.

## What the script checks
It verifies that `return_training_bundle=True` exposes a `restoration_target` that exactly matches the de-normalized RGB windows produced by the sliding-window operation. It also checks that the decoder output has the same window shape.

## Main code involved
- `vlm_restoration_positive_dtw.py`
- `restoration_recommended_components.py`
- `embeddingModel.py`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed` and the target equality assertion succeeds.
