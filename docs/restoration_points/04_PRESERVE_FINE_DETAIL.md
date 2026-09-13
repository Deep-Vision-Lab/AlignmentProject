# Point 04 — Preserve fine spatial detail

## Goal
Avoid excessive spatial downsampling, especially across window width, so Arabic dots, thin strokes, and neighboring letter parts do not collapse into the same feature.

## Run
```bash
bash scripts/restoration_points/04_preserve_fine_detail.sh
```

No flags are required.

## What the script checks
It verifies the restoration CNN encoder uses anisotropic early strides `(2,1)` so height is reduced before width, and checks that a 128x32 window reaches an 8x8 spatial feature map rather than the old highly compressed 8x2 representation.

## Main code involved
- `restoration_window_seq2seq.py` — `WindowSequenceCNNEncoder`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`, including the stride and spatial-shape assertions.
