# Point 05 — Reconstruction must depend on encoded features

## Goal
Prevent a decoder or skip path from reconstructing the image while the vector used for alignment contains little useful information.

## Run
```bash
The historical Point 05 launcher has been retired.
```

No flags are required.

## What the script checks
It creates two different encoded window vectors, decodes them, then swaps the vectors. The reconstructed outputs must swap accordingly. This confirms the decoder depends on the encoded vectors and has no image skip path that can bypass the representation.

## Main code involved
- `restoration_window_seq2seq.py` — `WindowSequenceStrokeDecoder`
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; distinct features produce distinct reconstructions and swapping features swaps the outputs.
