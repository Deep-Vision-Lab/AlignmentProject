# Point 07 — Fuse local and contextual vectors before DTW

## Goal
Combine each window's local visual evidence with its sequence-aware contextual representation before alignment.

## Run
```bash
bash scripts/restoration_points/07_local_context_fusion.sh
```

No flags are required.

## What the script checks
It verifies that `LocalContextFusion` concatenates/uses both local and contextual inputs, projects them to the alignment dimension, and L2-normalizes the final fused representation. Changing either input must change the fused output.

## Main code involved
- `restoration_recommended_components.py` — `LocalContextFusion`
- `vlm_restoration_positive_dtw.py`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; the output has unit norm and is sensitive to both local and contextual inputs.
