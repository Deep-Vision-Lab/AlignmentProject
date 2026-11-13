# Variable-width Soft Drop-DTW experiment

The four presets in `scripts/train/run_drop_dtw_ablation.sh` share the same
ArabicData A/B/C population, page-safe 80/10/10 split seed 42, ResNet18,
Transformer, sum fusion without a gate, transcript spaces, and typed negatives.
Each run saves its full `Config` and split IDs in its checkpoint. The fixed-size
baseline deliberately retains the prior 128×1024 direct resize. The new image
preprocessor fixes height to 128 and computes width from the cropped source
aspect ratio; it does not resize images to a batch-global width.

| Preset | Image geometry | Alignment loss |
|---|---|---|
| `baseline` | fixed width | existing aligned-mean Soft-DTW |
| `a` | aspect ratio / ragged windows | existing aligned-mean Soft-DTW |
| `b` | fixed width | Soft Drop-DTW |
| `c` | aspect ratio / ragged windows | Soft Drop-DTW |

For `c`, `prepare_image` records source/crop/model sizes, the *real* resized
width, artificial pixel columns (only when narrower than one window), exact
physical window starts, and foreground-derived background confidence. The final
window is shifted inside the image. The dataset supplies real windows and their
validity; collation keeps images/windows ragged. The model encodes **only real
windows** with the CNN and pads local embeddings after each line has been put
in RTL reading order. The Transformer masks invalid keys. `token_valid`,
`physical_window_indices`, and `window_starts` keep PAD distinct from genuine
white space. Evaluation uses the saved geometry to project masks back to the
full-height source image. The image-image local matcher already permits gaps
and multiple regions; its route is separate from the training image-text DP.

For each *valid* `[T,L]` NLL matrix, Soft Drop-DTW has matching transitions
(diagonal, repeated window, repeated letter), a real-window drop transition,
and the existing optional transcript-space skip. Every non-space transcript
letter must be matched at least once; an all-drop path is impossible. A real
window's drop cost interpolates from `space_drop_cost` to `text_drop_cost` using
continuous background confidence. PAD rows are mandatory zero-cost drops:
they are removed before letter logits and DP, mathematically identical to a
zero-cost PAD transition but without allocating padded cost grids. The
differentiable alignment occupancy is the gradient of the soft path free
energy with respect to the cell NLL. Reported energy is

`(expected matched NLL + expected real-window drop penalties) / expected matched-cell count`

plus a positive ink-coverage shortfall penalty when enabled. This is an
alignment-evidence score, **not** a calibrated probability or the raw soft-path
free energy. The same normalized energy is used for the existing strong/order
negative sequence objectives; local substitution uses the positive occupancy.
Negative transcript targets must be checked on validation because their scale
changes with the drop penalty.

The earlier suggestion `text_drop_cost=1.0` is below `log(37)=3.61` for the
available A/B/C alphabet and below sampled full-matrix NLL (~4.1–4.3), so it
would make dropping ink cheap. The experiment starts at 4.0. In a one-batch
random-weight smoke check, the old uniform-ratio strong-negative target range
had 100% success before training. The presets therefore start at ratios
1.0–1.3; that same smoke check had 0% initial success. These are diagnostic
starting settings, not tuned or validated hyperparameters.

Run the four jobs with distinct names:

```bash
RUN_NAME=drop_baseline GPUS=2 bash scripts/train/run_drop_dtw_ablation.sh baseline
RUN_NAME=drop_aspect_softdtw GPUS=2 bash scripts/train/run_drop_dtw_ablation.sh a
RUN_NAME=drop_fixed_dropdtw GPUS=2 bash scripts/train/run_drop_dtw_ablation.sh b
RUN_NAME=drop_aspect_dropdtw GPUS=2 bash scripts/train/run_drop_dtw_ablation.sh c
```

The wrapper calls `sbatch` on Linux. Use `OUTPUT_ROOT`, `BATCH_SIZE`, `EPOCHS`,
`BOOKS`, and `GPUS` to vary resources/paths; extra `train.py` flags follow the
preset name. `--checkpoint-every N` controls latest-checkpoint cadence and
defaults to one epoch for preemption safety. Checkpoints land at
`Weights/<RUN_NAME>/checkpoint_best.pt` and `checkpoint_latest.pt`; logs land
at `out/<RUN_NAME>_<SLURM_JOBID>.out`.

For each frozen checkpoint, change `CHECKPOINT` in notebook setup cell 4 and
rerun the added “Variable-width / Drop-DTW experiment diagnostics” cells. Use
the same held-out split, pair IDs, matcher settings, and GT mask policy for all
four runs. Report positive foreground IoU/Dice/precision/recall and negative
false-positive rate. The current local A/B/C folder copy has no `alignedPairs`
catalog or Book C `Unalignedlines` inventory; it supports feature/geometry
checks but cannot produce the requested paired GT comparison until the reviewed
pair/mask folders are available. Never substitute randomly mismatched lines as
certified negative ground truth.
