# Point 03 — Pretrained Restormer evaluation

## Goal
Evaluate whether a known pretrained restoration architecture provides better manuscript-window features than the current baseline before promoting it into the main model.

## Run
```bash
bash scripts/restoration_points/03_pretrained_restormer.sh
```

No flags are required.

## What the script does
It runs `restormer_pretrained_probe.py`. When the official Restormer source and checkpoint are available, the probe loads the pretrained model, captures its encoder/latent feature, checks that two manuscript-like windows produce distinct features, and performs one identity-restoration fine-tuning step.

If the third-party Restormer assets are absent, the script reports an explicit SKIP instead of silently pretending the model was tested.

## Main code involved
- `restormer_pretrained_probe.py`
- `tests/test_restoration_recommendation_points.py`

## Interpretation
Passing this probe means Restormer is usable for the experiment. It does not automatically mean Restormer should replace the baseline encoder; compare reconstruction quality and alignment-feature diagnostics first.
