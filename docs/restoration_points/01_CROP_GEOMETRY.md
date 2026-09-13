# Point 01 — Crop geometry

## Goal
Crop only the blank outer margins of each full manuscript line before window slicing, while preserving internal spaces, Arabic dots/diacritics, RGB/intensity information, and geometry needed to map predictions back to the original image.

## Run
```bash
bash scripts/restoration_points/01_crop_geometry.sh
```

No flags are required.

## What the script checks
It runs `test_point_01_crop_outer_margin_preserve_internal_space_rgb_and_geometry`. The test creates an RGB line with separated foreground regions plus a tiny high dot, preprocesses it, and verifies:
- outer whitespace is cropped;
- the internal gap is preserved;
- a safety margin retains the tiny dot;
- the processed model image remains RGB rather than becoming a binary mask;
- crop offsets and resize scale are saved.

## Main code involved
- `zero_shot_preprocessing.py` — `ManuscriptLinePreprocessor`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`.
