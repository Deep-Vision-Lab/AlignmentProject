# Point 12 — Small overfitting gate

## Goal
Before full training, prove that the restoration encoder/decoder can memorize several distinct windows and that the reconstruction really changes when encoded features are swapped.

## Run
```bash
The historical Point 12 launcher has been retired.
```

No flags are required.

## What the script does
The retired tool trained the local restoration encoder+decoder on eight distinct windows and checked:
- reconstruction loss decreases;
- the eight outputs do not collapse to the same image;
- swapping encoded features measurably changes the reconstructions.

## Main code involved
- `restoration_window_seq2seq.py`

## Pass condition
The tool finishes successfully after satisfying its loss-reduction, diversity, and feature-swap checks. A failure here should stop full-scale training.
