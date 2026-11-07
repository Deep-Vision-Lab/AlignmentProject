# Experimental image-only bidirectional window matching

`bidirectional_window` is an optional inference reward for the existing
`local_repeat_dtw` evaluation decoder. It compares image window embeddings
straight across the two images. It uses no character vectors, letter
probabilities, alphabet prior, transcripts, or text position prior. Training,
losses, model weights and checkpoint formats are unchanged.

## Selection and compatibility

In `notebooks/model_evaluation.ipynb`, set these variables in the existing
**EVALUATION-ONLY MATCHING SETTINGS** cell, then rerun the following settings
application cell and the desired existing sample output cells:

```python
ALIGNMENT_MODE = "local_repeat_dtw"
CANDIDATE_SIMILARITY_MODE = "bidirectional_window"
SIMILARITY_MODE = "bidirectional_window"
WINDOW_MATCH_TEMPERATURE = 0.10
WINDOW_MATCH_MIN_COSINE = 0.50
WINDOW_MATCH_REWARD_OFFSET = 0.0
VERIFY_WITH_LETTER_EVIDENCE = False
```

These are starting experimental values, not claimed optimal values. The ordinary
Python settings use the same names in lowercase. The CLI exposes them too:

```bash
python evaluate.py --mode shared_regions --checkpoint CHECKPOINT.pt \
  --image-a A.png --image-b B.png --output NEW_OUTPUT_DIRECTORY \
  --similarity-mode bidirectional_window \
  --candidate-similarity-mode bidirectional_window \
  --alignment-mode local_repeat_dtw \
  --window-match-temperature 0.10 --window-match-min-cosine 0.50 \
  --window-match-reward-offset 0.0
```

The existing `resolved_match_settings` function has always resolved
`candidate_similarity_mode or similarity_mode` into **both** settings. There
are no independent candidate and reward modes. The old cosine/letter override
behavior is retained: the notebook's existing `letter_evidence` candidate
setting overrides its `cosine` similarity setting. Defaults and current notebook
experiment settings are preserved.

For the new mode, conflicting explicit settings are rejected, including setting
only the candidate mode to `bidirectional_window` while leaving similarity at
`cosine`. Setting only `similarity_mode='bidirectional_window'` in the Python API
or CLI works when the candidate setting is unset. Use both settings in the
notebook. Letter verification and the legacy affine/stretch decoders are
explicitly unsupported with this mode. There is no fallback reward.

## Scores and validity

For compact valid sequences of lengths N and M:

```text
C = L2_normalize(Z_A.float()) @ L2_normalize(Z_B.float()).T
logits = C / window_match_temperature
log_p_B_given_A = log_softmax(logits, dim=1)
log_p_A_given_B = log_softmax(logits, dim=0)
evidence = 0.5 * (log_p_B_given_A + log(M) + log_p_A_given_B + log(N))
reward = evidence - window_match_reward_offset
eligible = C >= window_match_min_cosine
```

`dtw.bidirectional_window_scores` computes stable directional log-softmaxes in
float32. Normalization/matmul also use float32, disabling ambient autocast where
supported (PyTorch 2.4 does not have an MPS autocast context). The existing decoder
retains its float64 accumulators. `match_features` removes token-invalid rows
and physical identities before validation, normalization, cosine or reductions;
N and M are the per-pair valid lengths. Invalid padding may contain NaNs when it
is explicitly masked out. Valid vectors must have finite nonzero float32 norms;
nonfinite/zero/unrepresentable vectors fail clearly. Empty inputs return empty
matrices, no candidates, and zero maximum local score without log(0).

Temperature must be finite and positive, the cosine gate must be in [-1,1], and
the offset must be finite. Values too extreme for finite float32 scoring are
rejected. Cached cosine input must already describe this compact valid grid and
correspond to the supplied embeddings; shape/range checks cannot certify that a
caller supplied the right checkpoint's features.

The directional scores are relative normalized window scores, **not calibrated
match probabilities**. Evidence is measured against the uniform directional
baseline: uniform similarities give evidence zero before the offset. Even an
unrelated pair can have a strong relative best cell. Repeated valid windows
compete for directional mass. Neither relative evidence nor the absolute gate
alone solves every negative-pair false positive.

## Flow through decoding, acceptance and masks

1. CLI `evaluate_pair` and notebook/session `predict_cached_pair` call the same
   `match_features` dispatch after obtaining cached valid image embeddings.
2. The reward above replaces the cosine background/raw reward. Cosine remains
   available for diagnostics, existing short-region filters and mutual top-k
   anchor ranking. Anchors keep their existing stable-tie ranking over all valid
   cosine cells, intersected with positive reward and absolute eligibility;
   removing a cell does not promote lower-ranked cells into anchors.
3. `local_repeat_regions` sends reward and a separate boolean eligibility matrix
   into `_local_repeat_path`. Diagonal and both repeat match states require
   eligibility and positive reward. Gated cells retain their diagnostic rewards;
   they cannot become matches by replacing their rewards with zero. Affine gap
   states can traverse them with the same gap costs as before. Gap states never
   supply matched support.
4. Segmentation and trim variants only retain existing matches. Both extension
   passes require eligibility for every new matched pair. Merge bridge promotion
   also requires eligibility. A final candidate audit rejects an implementation
   error if any candidate contains a forbidden match. Existing repeat limits,
   physical gap bounds, support/anchor checks, dominance, acceptance and greedy
   noncrossing selection remain in place.
5. `region_mask` keeps the existing supported-window and allowed-small-fill
   projection through source crop/resize geometry. A gap window may still be
   foreground through existing `filled_physical` rendering or overlap with a
   neighboring matched window's footprint. Full-height source spans can appear
   continuous despite skipped cells. A foreground pixel does not certify an
   eligible matched pair; inspect `pairs`, `path_steps` and `filled_physical`.

Existing `threshold` and short-region cosine filters remain active where they
already applied. In particular extension also requires cosine >= `threshold`,
which may be stricter than the new absolute gate. `score_mode` and
`contrast_margin` do not affect bidirectional rewards.

## Assets, caches and artifacts

`load_checkpoint(..., load_text=False)` reconstructs only the image model;
training/loss callers retain the original default `load_text=True`. Evaluation
sessions load character vectors lazily only when an explicit letter mode or
recognition operation needs them. The bundled checkpoint archive is still read
as a whole; its character state/prior are not instantiated or used by the window
matcher. No separate letter inference assets are required.

In this mode the notebook skips character recognition, transcript ranking and
the window-letter inspector. The validation comparison also requires explicit
`ALLOW_LETTER_MODE_COMPARISON=True` before loading the letter baseline in an
otherwise image-only run.

All three parameters and both mode names are in resolved settings, result cache
keys and saved metadata. `MATCH_VERSION` is bumped to invalidate stale matcher
results. Image-feature caches are separate from reward/result caches; changing
temperature, gate or offset recomputes rewards/paths on compatible cached features
without running the CNN/Transformer. Saved pair artifacts include `cosine.npy`,
the usual rewards, `bidirectional_evidence.npy`, `absolute_cosine_eligible.npy`,
`log_p_B_given_A.npy`, `log_p_A_given_B.npy`, paths, settings, and checkpoint
identity. The quantitative evaluation cache continues to store frozen features
independently of the three new reward parameters.

## Diagnostics and comparison

New-mode sample outputs add four panels: raw cosine, bidirectional evidence,
reward and absolute eligibility. Each uses the existing accepted/rejected route
conventions. The usual predicted source masks, overlays and evaluation metrics
remain in the sample output. Directional log scores are also exported numerically.

Enable `RUN_WINDOW_DIAGNOSTIC_EXAMPLES` in the new notebook cell for two tiny
encoder-free examples, also available as
`evaluate.window_match_diagnostic_examples()`:

* `repeated_valid`: two identical valid windows compete for mass; the repeated
  match is still eligible and a region is accepted.
* `all_below_gate`: diagonal cosine 0.20 against off-diagonal 0.00 yields positive
  relative evidence at the diagonal, but no cells, anchors or paths pass the
  initial 0.50 cosine gate.

The existing **Same-checkpoint validation** cell now compares cosine,
letter_evidence and bidirectional_window on the same saved-validation pair IDs,
checkpoint and representation. `SIMILARITY_COMPARISON_SETTINGS` accepts separately
chosen per-mode validation settings. It performs evaluation, with no automatic
search. Model features remain cached across all modes. Character inference is
used only by the explicit letter baseline.

Each mode needs separate validation tuning, including gap/repeat penalties and
region acceptance thresholds, because reward scales differ. Do not select
settings on held-out test data. No training or large Optuna job is needed.

The existing metrics retain their definitions:

* Positive mask IoU = foreground intersection / foreground union.
* Positive Dice = 2 * foreground intersection / (predicted foreground size + GT
  foreground size). Existing positive pair means average A and B; positive
  population summaries average available annotated positive pairs. Missing GT is
  unavailable, except the notebook's explicit folder-masks-missing-black policy.
* `aggregate_summary.true_negative_false_positive_rate` = number of
  `true_no_overlap` pairs with any accepted region / number of `true_no_overlap`
  pairs. Unknown/partial-overlap cases are reported separately.
* The quantitative mask evaluator's `negative_pair_fpr` = negative pairs with
  any predicted foreground on either side / evaluated negative pairs;
  `negative_masked_area_fraction` averages the two side coverage fractions per
  negative pair, then averages across negative pairs. Presumed negatives retain
  their existing provenance labels.

The default Optuna search domains are unchanged; they do not tune the new mode.
Use explicit validation configurations for this inference experiment.

## Verification in this workspace

Using `.venv-mask-eval/bin/python -m pytest`, the eight relevant modules
(`test_bidirectional_window`, `test_dtw`, `test_local_repeat_dtw`,
`test_local_repeat_evaluation`, `test_path_quality_evaluation`,
`test_evaluation_notebook`, `test_matching_parameter_search`, and
`test_quantitative_masks`) finished with **124 passed, 4 deselected**. This
includes 32 new tests and actual CPU/MPS scoring, CLI parsing/export, cached
parameter changes, legacy rewards, and the notebook three-mode comparison.
Notebook code cells parse and both example plots were rendered and inspected.

The four exclusions were reproduced against original repository source plus
the pre-edit notebook in a temporary directory: tests assume a combined settings
cell (`current_match_settings` is now in a separate cell), five samples (the
existing notebook uses ten), a non-Jupyter setup (`get_ipython` is unavailable),
and an implicit `SHOWN_REPRESENTATIONS` global. These pre-existing test/notebook
contract mismatches are outside this experiment. A broader check also exposed
the existing training smoke failure on PyTorch 2.4/MPS: training constructs an
unsupported MPS autocast context even when AMP is disabled. Training code was
left unchanged apart from the optional inference checkpoint-loader flag. The
new window scoring uses a supported context or no context and passes on MPS.
