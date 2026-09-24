# Point 09 — DTW transition semantics

## Goal
Make sure the DTW recurrence allows the correspondence patterns required by the manuscript problem: several windows may belong to one letter, and when needed one window may cover more than one letter.

## Run
```bash
The historical Point 09 launcher has been retired.
```

No flags are required.

## What the script checks
It evaluates the differentiable DTW cost on two edge cases:
- a 4-window sequence aligned to a single letter, requiring repeated movement along the window axis;
- a single window aligned to four letters, requiring repeated movement along the text axis when necessary.

It also backpropagates through both cases to confirm they remain differentiable.

## Main code involved
- `vlm_restoration_positive_dtw.py` — `_soft_dtw_cost_matrix`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; both costs are finite and gradients reach the input matrices.
