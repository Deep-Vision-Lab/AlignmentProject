# Point 11 — Multiple vectors per window ablation

## Goal
Keep one fused vector per physical window as the baseline, but make it possible to test whether several spatial vectors per window better represent windows containing multiple letters or letter parts.

## Run
```bash
The historical Point 11 launcher has been retired.
```

No flags are required.

## What the script checks
It calls the optional spatial-vector API with `vectors_per_window=4` and verifies that every physical window returns four normalized vectors with the expected tensor shape.

## Main code involved
- `restoration_window_seq2seq.py` — `WindowSequenceCNNEncoder.spatial_vectors`
- `tests/test_restoration_recommendation_points.py`

## Interpretation
This is an ablation capability, not the default model path. Only promote it if the single-vector representation proves insufficient.

## Pass condition
Pytest ends with `1 passed`; K=4 vectors are returned per window and all are normalized.
