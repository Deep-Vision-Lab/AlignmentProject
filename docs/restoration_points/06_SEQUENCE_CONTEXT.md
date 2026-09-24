# Point 06 — Sequence context

## Goal
Let every physical window use information from neighboring windows instead of making its alignment decision from local appearance alone.

## Run
```bash
The historical Point 06 launcher has been retired.
```

No flags are required.

## What the script checks
It feeds local vectors plus positional information through the Transformer, changes one neighboring token strongly, and verifies that another window's contextual representation changes. This demonstrates actual cross-window contextual interaction.

## Main code involved
- `embeddingModel.py` — Transformer/context encoder path
- `tests/test_restoration_recommendation_points.py`

## Pass condition
Pytest ends with `1 passed`; altering a neighbor changes the contextual vector of another window.
