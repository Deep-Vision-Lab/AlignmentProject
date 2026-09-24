# Point 10 — Recompute DTW from current features

## Goal
Do not artificially force a different DTW route every epoch. Instead, recompute the route from the current feature-derived cost matrix and monitor whether learning changes similarities, gradients, and alignment quality.

## Run
```bash
The historical Point 10 launcher has been retired.
```

No flags are required.

## What the script checks
It feeds two different cost matrices into the hard-DTW diagnostic and verifies that the resulting optimal route changes naturally while preserving valid start/end points. This proves the route is recomputed from current costs rather than cached or fixed.

## Main code involved
- `restoration_epoch_probe.py` — `_hard_dtw`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; the two cost matrices produce different valid routes.
