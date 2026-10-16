# Tune evaluation matching with Optuna

Run this from the repository root in your PyTorch environment. Optuna is already
listed in `requirements.txt`; install `optuna>=4,<5` if it is missing.

```bash
python scripts/eval/tune_matching.py \
  --checkpoint Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt \
  --dataset DataSet/ArabicData \
  --n-trials 200 \
  --device auto
```

This loads a frozen checkpoint once and defaults to **its saved validation split**.
It caches CNN/Transformer features per unique line and cosine matrices per pair.
Trials rerun the existing image matcher and mask builder; no training takes place.
`auto` chooses CUDA, then MPS, then CPU for feature extraction. Matching runs on CPU.
The default content crop matches the evaluation notebook; use `--crop checkpoint`
to preserve the checkpoint's preprocessing instead.

## Objective and masks

The maximized objective is

```text
mean_mask_iou = mean over positive validation pairs of (IoU_A + IoU_B) / 2
```

A positive pair has foreground in **both** ground-truth masks. Each pair and each
side has equal weight regardless of image size or foreground area. Missing
predictions on positive pairs score zero. Precision, pixel accuracy, path score,
and false-positive penalties are not part of this objective.

For ArabicData folder pairs, GT is the source-size PNG in `masksA`/`masksB` beside
`imagesA`/`imagesB`, exactly as in the notebook. For manifest datasets, GT is the
explicit `mask` annotation. `--mask-source folders|manifest` overrides automatic
selection. A missing file becomes a black mask, matching the notebook's policy.
No box, transcript, or LCS-derived GT is used. Source/mask size mismatches fail.

Both-black pairs are excluded from the selection objective: empty-vs-empty IoU
would otherwise favor suppressing predictions when black examples dominate.
Their false-positive rate is reported separately (any white predicted pixel on
either side counts). Pairs with foreground on only one side are reported as
`partial` and excluded from the objective. Their individual IoUs are still saved.
The script fails if no positive pair exists in the selected split, with no split
fallback. If validation masks are unavailable, you can explicitly use
`--split train` for **in-sample calibration**. Its results measure fit on images
seen during training, rather than held-out validation quality. Test-split tuning
is not offered. Keep a separate output directory for this study:

```bash
python scripts/eval/tune_matching.py --split train --n-trials 200 \
  --output Results/Evaluation/matching_optuna_train
```

## Search choices

The first three queued configurations cover the cosine baseline, cosine with
active letter verification, and letter-evidence rewards. Remaining trials use
Optuna's seeded TPE sampler, with 25 startup trials. All trials finish evaluating
the same selected population; no intermediate-score pruning is used.

| Parameter | Search |
| --- | --- |
| Similarity | Cosine or letter evidence; candidate/reward modes synchronized |
| Cosine threshold | 0.35–0.80 in steps of 0.025; cosine mode only |
| Background contrast margin | 0–0.12 in steps of 0.01; cosine mode only |
| Normal distinct support in A and B | Shared minimum of 4–7 windows |
| Normal matched pairs | 4–8 |
| Short regions | Disabled, or relaxed/balanced/strict confidence profile |
| Maximum internal gap | 0–2 windows |
| Gap opening penalty | 0.05, 0.10, 0.20, 0.30 |
| Gap extension penalty | Opening penalty × 0.25, 0.5, or 1.0 |
| Repeat penalty | 0.02, 0.05, 0.10, 0.15 |
| Maximum consecutive repeats | 1–3 |
| Mutual anchor top-k | 1–4 |
| Normal mutual anchor fraction | 0.10, 0.15, 0.25, 0.40 |
| Global minimum path density | Disabled, 0.65, 0.80, 0.90 |
| Path extension | On/off; maximum weak run 0–2 |
| Region merging | On/off; shared A/B merge gap 0–2 |
| Dominance quality ratio | 0.50, 0.75, 0.90, 1.0 |
| Cosine letter verification | On/off; minimum mean evidence 0, 0.10, 0.25; positive fraction 0.50, 0.60, 0.75 |

Short profiles couple related confidence checks to keep the search manageable:

| Profile | Mean/median cosine | Mean reward | Path density | Mutual anchor fraction |
| --- | --- | --- | --- | --- |
| Relaxed | 0.68 | 0.05 | 0.70 | 0.25 |
| Balanced | 0.72 | 0.15 | 0.80 | 0.40 |
| Strict | 0.78 | 0.25 | 0.90 | 0.60 |

Short regions always need at least 3 distinct windows on each side, 4 matched
pairs, 2 anchors, and no internal gap. Normal minima stay above the short floor,
so short regions can actually reach their confidence checks. A/B normal minima
are tied because both lines use the same window grid; repeats accommodate unequal
spans. Architecture, representation, crop, checkpoint alphabet/prior, and candidate
capacity stay fixed during a study. The default capacity is 512; trials report how
many pairs reach this limit. Display-only and inactive legacy options are not tuned.

These are bounded starting ranges, not a guarantee of a global optimum. Run more
trials if the best score is still improving. A small validation population can
overfit; the validation score is a tuning result, not a held-out test result.

## Outputs and resuming

The default output is `Results/Evaluation/matching_optuna/`:

- `study.sqlite3`: persistent Optuna study.
- `run_manifest.json`: checkpoint, exact pair IDs, image/mask hashes, preprocessing,
  search definition, code hashes, seed, and validation membership provenance.
- `trials.csv`: scores, A/B IoU, negative false-positive rate, duration, sampled parameters.
- `trial_00000_pairs.csv`, etc.: per-pair IoU and prediction diagnostics for each trial.
- `best_settings.json`: the complete resolved matcher settings, ready for evaluation.
- `best_result.json`: winning trial, metrics, parameters, settings, and provenance.

Best settings and CSV summaries update after each completed trial. To add another
200 trials to the same study:

```bash
python scripts/eval/tune_matching.py --n-trials 200 --resume
```

Use the same checkpoint, dataset, representation, crop, seed, pair limits, code,
and dependency versions when resuming. Changed provenance requires a new output
directory or study name. Resuming preserves completed trials and the winning
score; the sampler is re-seeded, so it does not reproduce an uninterrupted random
sequence exactly. Optuna's SQLite persistence is described in its
[study API documentation](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.study.create_study.html).

For a quick smoke run:

```bash
python scripts/eval/tune_matching.py --n-trials 3 --max-pairs 6 \
  --output Results/Evaluation/matching_optuna_smoke
```

`--max-pairs 0` evaluates all eligible pairs (default); a positive value selects a
fixed seeded subset, preserving positive and black-mask examples where possible.
A subset study selects its winner on that subset only. `--negative-pairs 100`
bounds the ArabicData unmatched-line comparison pool; those examples remain
diagnostic. `--timeout 3600` limits optimization to roughly one hour, excluding
initial loading/caching; the current trial is allowed to finish.

Apply the saved settings in the notebook **after** its normal settings-application
cell, then rerun population evaluation or sample output cells:

```python
import json
from pathlib import Path
from evaluate import resolved_match_settings

SETTINGS = resolved_match_settings(json.loads(
    Path(ROOT / 'Results/Evaluation/matching_optuna/best_settings.json').read_text()
))
session.settings = dict(SETTINGS)
# The population cell checks this function against session.settings.
def current_match_settings(threshold=None):
    settings = dict(SETTINGS)
    if threshold is not None:
        settings['threshold'] = float(threshold)
    return resolved_match_settings(settings)
```

Re-running the original settings-definition/application cells replaces the loaded
winner with their manual values. Freeze the winning settings before evaluating
the saved test split; do not tune again based on test results.
