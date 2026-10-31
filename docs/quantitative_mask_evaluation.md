# Frozen-checkpoint quantitative mask evaluation

The default dataset is **DataSet/ArabicData**. With `--data-policy auto` and no
review manifest, quantitative evaluation reads **alignedPairs** and
**Unalignedlines** directly, following the notebook folder-mask policy. No pair
manifest is required for this mode.

Run from **AlignmentProject**, branch `refactor/simple-alignment-core`. This is
separate from the annotation website and from training Optuna. No training,
model, loss, matcher, notebook, existing split, or existing tuner is changed.

## Source-verified evaluator and notebook discrepancies

`scripts/eval/mask_quantitative.py` calls the exact notebook prediction helper
`evaluation_utils.predict_cached_pair`, which calls `evaluate.match_features`
and `evaluate.region_mask`. Candidate segmentation, path trimming, extension,
dominance selection, merging, supported-window filling and physical-window to
source-pixel conversion are therefore production operations, not reimplementations.
Baseline settings are read from the current notebook's literal assignments and
its actual `current_match_settings` function, without executing its data/plot cells.

Source inspection establishes:

* `candidate_similarity_mode or similarity_mode` determines BOTH candidates and
  rewards: `resolved_match_settings` overwrites `similarity_mode`. They are not
  independent controls despite the notebook descriptions.
* Letter evidence replaces the background/raw cosine reward matrix. `score_mode`
  and `contrast_margin` do not affect these rewards and are not searched. Cosine
  still defines mutual top-k anchors and short-region cosine filters. The fixed
  `.5` cosine threshold also gates extension. It is not the letter reward threshold.
* Short anchor counts/fractions use the **maximum of both alias values**. The
  current notebook's `.60` and `.95` fractions resolve to **.95**, not .60.
  Sampled Stage 2 aliases are synchronized explicitly.
* Density is `min(distinct_supported_A / valid_windows_in_A_span,
  distinct_supported_B / valid_windows_in_B_span)`. It is not "matched cells /
  path length" as the short-density notebook comment says. Repeats and fills
  do not add distinct support. Gaps measured for validation count skipped valid
  windows; segmentation additionally guards numeric physical discontinuities.
* Extension uses `min(max_gap, extension_max_internal_gap)`. Merging uses
  `min(max_gap, merge_max_gap_A/B)`, requires ordered paths and bridge evidence,
  and revalidates the combined path. Merging uses `extension_min_reward` for
  bridge evidence even if extension itself is disabled; inherited inactive
  extension settings therefore remain recorded and are not reset arbitrarily.
* Filling uses physical gaps up to `max_gap`, and adds only identities present
  in the valid physical-window list. It never creates padded/invalid identities.
* `acceptance_offset` and `prior_floor` affect letter-evidence mathematics;
  their "legacy" notebook descriptions are inaccurate. They retain baseline
  values. Repeat penalties grow with the consecutive repeat run, rather than
  being a constant charge on each repeated cell.
* Explicit normal minimums override the legacy fallback minima. Retired
  first-cell anchor settings are diagnostic only. Letter verification filters
  are inactive when `verify_with_letter_evidence=False`. The notebook labels
  its configuration "Optuna best trial 135" and "initial, uncalibrated" in
  different cells; neither proves held-out performance.
* The notebook enables split fallback, samples presumed unmatched negatives,
  and treats missing mask files as black. Folder evaluation adopts the seeded
  comparisons and missing-mask policy, but never split fallback. Reviewed mode
  instead requires explicit negative labels and mask files.

Every trial saves the complete requested settings, production-resolved settings,
derived alias/gap limits, active/inactive diagnostics, actual metrics and errors.
Unlisted settings retain the exact notebook baseline. `COMPARISON_MAX_PAIRS=0`;
display settings are not searched. A JSON baseline configuration is also saved
when a data-contract failure prevents scoring it.

## Required fixed data and GT contract

The checkpoint's saved train/val/test membership is mandatory. No fallback or
resplitting exists. Source paths, decoded pixel hashes (including copies under
different paths), and available page/group IDs are audited across ALL saved
splits. Image/page leakage aborts. Manuscripts may legitimately span page-level
splits under the existing policy; that limitation is reported, not hidden.

With `--data-policy auto`, an ArabicData root containing both export folders
selects `folder_masks`. `--data-policy folder_masks` selects it explicitly:

* `alignedPairs/imagesA` and `imagesB` are joined by their full filename stem.
  Their existing `masksA`/`masksB` files supply full-source GT for those partners.
* `Unalignedlines/imagesA` and `imagesB` are independent inventories. The existing
  production sampler includes **all eligible** cross-book comparisons per split
  by default (`--negative-count all`), excluding related page/pair groups, identical
  images and shared transcript words. Use `--negative-count N` to request a
  reproducible capped subset (seed 42). Matching filenames are not treated as pairs.
* The default `--pairs-per-split 500` keeps every aligned positive, preserves the
  original unmatched comparisons, then samples enough existing
  `metadata/dataset_manifest_full_pairs.jsonl` records labeled `no_shared_content`
  to reach the fixed target. Both source lines must belong to the current saved
  split and have copies in alignedPairs/Unalignedlines. Original book indexes
  only resolve identities; inference uses the exported image/text copies.
  Physical A/B identities are deduplicated, including duplicate metadata rows,
  copied paths and known positives. Seed 42 fixes supplemental selection.
* Supplemental negatives receive **pair-specific black GT from their recorded
  coarse label**. An aligned line's mask describes its original partner, so it
  is never reused for a supplemental pairing. These records remain presumed
  negatives rather than manually reviewed pairs. Each record saves its original
  label source, both actual source folders and `label_as_black` provenance.
  Insufficient distinct candidates cause an explicit error rather than repeated
  samples, split changes or invented labels.
* Missing masks become source-sized black masks, matching the notebook policy.
  Original-pool masks must agree on foreground presence; one-sided foreground and size
  mismatches remain explicit errors. Masks are never derived from transcripts.
* BookA/BookB indexes verify copied files and checkpoint membership only. Actual
  inference images come from the two export folders. Other splits are excluded;
  no source line is silently moved into validation or test.
* Every pair records `source_folder`, mask paths, missing-mask fallback, original
  target and `presumed_negative`. Reports count those cases and label the FPR as
  including **presumed comparisons**, rather than verified-negative accuracy.

`--pairs-per-split all` uses only the earlier original folder pool, without
supplemental records. A numeric value applies separately to validation and final
test, retaining every positive in each split. The sample target is part of run
provenance and must remain fixed on resume.

`--data-policy reviewed` retains the original strict workflow: no constructed
negatives, missing-mask fallback or inferred negative certification. Supplying
`--split-manifest` with `auto` also selects reviewed mode. Supplied known negative
labels require explicit empty masks. The same source/split leakage checks apply
to both modes.

For separately reviewed pairs, supply an explicit JSON list with
`--data-policy reviewed --split-manifest /actual/path/to/pairs.json`:


```json
{
  "schema_version": 1,
  "pairs": [
    {"sample_id": "reviewed-validation-positive", "split": "val", "target": 1,
     "verified": true,
     "sides": [{"image": "BookA/images/ACTUAL_A.png"},
               {"image": "BookB/images/ACTUAL_B.png"}],
     "masks": ["reviewed_gt/positive_A.png", "reviewed_gt/positive_B.png"]},
    {"sample_id": "reviewed-validation-negative", "split": "val", "target": 0,
     "verified": true,
     "sides": [{"image": "BookA/images/OTHER_A.png"},
               {"image": "BookB/images/OTHER_B.png"}],
     "masks": ["reviewed_gt/negative_A.png", "reviewed_gt/negative_B.png"]}
  ]
}
```

Replace example paths with actual manually reviewed lines in the corresponding
saved split; add separately reviewed `split: "test"` records for final test.
Relative paths are relative to `--dataset`. When a sample ID already exists in
the production catalog, `sides` may be omitted. Custom pairs must refer to exact
canonical saved-split image paths. Both negative GT files must be explicitly
empty. Do not set `verified: true` based on random pairing or missing masks.
The full saved line membership and selected pair list are written to
`split_audit.json` and `val_split_manifest.json` / `test_split_manifest.json`.

Masks are scored in original full source-image coordinates, separately for A/B
at their own heights and widths. Default: mismatched GT dimensions are an error.
`--resize-gt` explicitly permits nearest-neighbor GT resizing and records it.
Preprocessing adds no padding; the model's `token_valid` filtering and original
physical identities are preserved, verified on every cache load, and passed to
the exact source-space mask constructor. Blank physical margins are not
misidentified as artificial padding. Pixel metrics use 2D masks; 1D interval
extraction happens only after both masks are verified to be full-height spans.
For non-span GT, mask metrics remain valid but region F1 is explicitly unavailable
for the entire aggregate, with its missing-eligibility denominator reported.

## Metrics and uncertainty

For each positive pair, independently compute A and B foreground IoU, Dice,
precision and recall; average the two, then average positive pairs. Empty
positive predictions score zero on all four. Background IoU and empty-negative
overlap scores never enter these means. Negative FPR counts a pair if **either**
line predicts foreground. Negative area averages A/B foreground fractions, then
all negative pairs, including correct empty predictions. Raw counts/denominators
are saved. Every pair and line is exported, including missed/empty predictions.

Region F1 uses contiguous half-open intervals and Hungarian assignment with a
cardinality-dominating bonus plus IoU, restricted to edges with IoU >= .5. This
maximizes the number of matches before sum-IoU tie-breaking. Region TP/FP/FN
are pooled over BOTH lines of ALL positive pairs before computing precision,
recall and F1; this is distinct from the macro mask metrics.

Baseline and final paper metrics use 2,000 seeded bootstrap resamples by default.
Pairs sharing source page/group IDs or line identities form connected dependency
components and are resampled together. Disjoint pairs reduce to a pair bootstrap.
Fewer than two independent groups produce unavailable intervals, not misleading
zero-width certainty. An exact Clopper-Pearson negative FPR interval is also
reported for independent/disjoint negative pairs, including a positive upper
bound with zero false positives. Independence is conditional on available group
metadata; missing manuscript/page metadata limits that interpretation.

## Search, seeds and recovery

Defaults: Stage 1 300 completed trials; Stage 2 200; constrained TPE; 40 startup
trials; seed 42; no pruning; one worker. All completed evaluations use the same
complete validation set. Default FPR requirement .05 is declared in provenance
and never optimized/relaxed. Selection explicitly filters COMPLETED feasible
trials, maximizes IoU, then minimizes negative area on exact ties. No feasible
trial means `NO_FEASIBLE_TRIAL`, trade-offs only, and no final-test command allowed.

Stage 1 searches exactly the requested core ranges. Stage 2 jointly searches
those ranges and conditional short/extension/merging dimensions. Numeric bounds
may depend on earlier suggestions; categorical choices never change between
trials. Current baseline is saved/scored exactly and enqueued into Stage 1 when
representable. Stage 2 imports up to five strongest completed feasible Stage 1
trials, with their identical full-validation results and exact baseline additions,
as completed seeds. These count toward its total. This preserves asymmetric
baseline merge gaps without falsely claiming a shared gap suggestion represented
them. Subsequent Stage 2 suggestions use the required coupled search ranges.

SQLite is the local CLI default; `--storage` also accepts another Optuna RDB URL
or `journal:///absolute/project/path`. The BGU launcher uses an NFS-compatible
Optuna journal and symlink locks, never SQLite on shared storage. See
[the batch-job instructions](mask_batch_slurm.md) for environment, resources,
preflight, monitoring and recovery. Failed/interrupted trials do not consume the
completed-trial budgets; imported completed seeds count toward Stage 2's budget.
Resume checks the dataset/GT/checkpoint hashes, effective preprocessing/model
config, fixed objective/FPR requirement, search version, source fingerprints,
notebook, package versions and seeds. Changed contracts require a new output.
Local RDB heartbeat recovery marks abandoned RUNNING trials failed after 90
seconds. The batch journal worker instead verifies the old job is absent from
SLURM before recovering its RUNNING trials; unknown/live writers are not replaced.
Actual exceptions/tracebacks are saved as failed trials, never score zero.
On resume the seeded sampler uses completed history; a restarted sampler is not
claimed to reproduce the bitwise uninterrupted suggestion sequence.

NPZ caches contain only frozen image outputs, physical indices, validity and
geometry. Keys include checkpoint, image/pair IDs, preprocessing/fusion config,
representation, evidence settings, source/package versions, device and seed.
GT remains outside inference caches. Cosine is reused; parameter-dependent
routes, evidence validation, extension, merging, filling and final masks are
recomputed through the production functions every trial.

## Commands

Use the existing `manucripts_align` cluster environment. Local verification uses
the isolated `.venv-mask-eval` environment (PyTorch 2.4.0, Optuna 4.9.0); it does
not replace the user's base environment. From the repository root:

```bash
source .venv-mask-eval/bin/activate
# Use the exported ArabicData folders; no reviewed manifest is needed.
OUT=Results/Evaluation/quantitative_masks_500_pairs

# 1. Build/reuse frozen validation outputs and score exact current baseline.
python scripts/eval/tune_masks.py prepare --dataset DataSet/ArabicData --data-policy folder_masks --pairs-per-split 500 --output "$OUT" --device cpu

# 2. Full validation-only search, resumable to 300/200 total trials.
python scripts/eval/tune_masks.py tune --dataset DataSet/ArabicData --data-policy folder_masks --pairs-per-split 500 --output "$OUT" --device cpu --resume

# 3. Freeze validation selection; score test baseline, selected and two ablations.
python scripts/eval/tune_masks.py test --dataset DataSet/ArabicData --data-policy folder_masks --pairs-per-split 500 --output "$OUT" --device cpu

# 4. Regenerate paper exports without inference or parameter selection.
python scripts/eval/tune_masks.py export --output "$OUT"
```

Pass `--checkpoint`, `--dataset`, `--data-policy`, `--negative-count`, `--pairs-per-split`, `--cache-dir`, `--seed`, `--bootstrap`,
`--stage1-trials`, `--stage2-trials`, `--startup-trials`, `--max-negative-pair-fpr`,
`--study-name`, `--storage`, `--timeout` as needed. Use the SAME preprocessing,
device and provenance options for prepare/tune/test. `--resume` safely creates
a fresh study when none exists; it never increases totals beyond requested targets.

HPC submission, matching the repository's GPU partition and Conda environment:

```bash
bash scripts/eval/tune_masks.sh --dataset "$PWD/DataSet/ArabicData" \
  --data-policy folder_masks --pairs-per-split 500 \
  --job-name mask_optuna_500_pairs \
  --output-dir "$PWD/Results/Evaluation/quantitative_masks_500_pairs_cuda" --resume
```

Do not use a CPU baseline directory for a CUDA study: device/outputs are part of
the reproducibility contract. No full search is submitted automatically.

Final test freezes `selected.json` before reading test GT; further tuning in that
output is prohibited. Baseline, selected, extension-off and merging-off are
reported separately. Ablations are not retuned; already-disabled components
are marked identical. Files: `trials.csv`, `study.sqlite3` (local) or
`optuna.journal` (BGU), per-trial
`per_line.csv`, `per_pair.csv`, `diagnostics.json`, `aggregate.json/csv`,
`best_effective_settings.json`, `notebook_settings.py`, provenance/split manifests,
`test/methods.json`, `aggregate_results.json/csv`, `paper_table.csv`,
`paper_table.tex`, and `supplementary_region_f1.csv`.

## Verification and data availability

```bash
PYTHONPATH=.:tests python -m pytest tests/test_quantitative_masks.py -q
```

Tests cover metrics, empty/missed predictions, unequal widths and physical
identity preservation, full-height guards, one-to-one interval assignment,
conditional bounds/aliases, feasible selection, real SQLite resume, inference
cache reuse, current-notebook predictor parity, leakage rejection and final
test/LaTeX export on explicitly synthetic frozen-checkpoint fixtures. They do
not certify synthetic fixture labels as real manuscript negatives.

The supplied default checkpoint's saved splits contain 924 training, 112
validation and 166 test lines, with no decoded-image or known page-group leakage.
The current ArabicData folders provide **53 aligned validation pairs** and
**88 aligned test pairs**. There are 156 eligible validation and 351 eligible test
Unalignedlines combinations. The original folder-only pool has **209 validation
pairs** (53 positive + 156 presumed negative) and **439 test pairs** (88 positive +
351 presumed negative). With the new default target of 500, validation retains
its 53 positives and 156 original comparisons and adds **291 recorded negative
pairs**, giving **53 positive + 447 negative = 500**. Every trial uses this same
fixed validation sample. No training/test line moves into validation. More
positive pairs still require additional labeled validation data.
These comparisons have no mask files and are explicitly presumed negatives under
the notebook's black-mask policy. The full export contains 770 aligned pairs with
both masks; only saved-split members enter each quantitative run.

The earlier capped validation baseline and exact manifests are saved under
`Results/Evaluation/arabic_data_folder_masks_20261031/`: positive IoU **0.714124**,
Dice **0.789096**, presumed-negative pair FPR **0.73** (73/100). Confidence
intervals are unavailable because available dependency links form one component
per class. The candidate limit was reached on 146/153 pairs; no limit or matcher
setting was changed. See `baseline/report.md` for the complete interpretation. Historical strict-mode
preflight artifacts under `quantitative_masks_real_audit/` used a different
label contract and do not describe folder-mode availability. No real full
500-trial search or real final test is claimed.

The earlier original-pool all-comparisons run is saved separately under
`Results/Evaluation/arabic_data_all_pairs_20261031/`. Its manifest records all
209 validation pairs. The expanded baseline gives IoU **0.714124**, Dice
**0.789096**, and presumed-negative FPR **0.743590** (116/156). Full details
are in its `baseline/report.md`. Use a fresh output/study for the expanded population;
resuming an earlier capped study is rejected by the provenance check.

The 500-pair baseline and exact sample manifest are saved separately under
`Results/Evaluation/arabic_data_500_pairs_20261031/`. The manifest distinguishes
156 unmatched comparisons from 291 supplemental recorded negatives. Use a fresh
output/study when changing from 100/156 negatives to this 447-negative sample.
The 500-pair run does not change the frozen checkpoint or matcher settings.
Its completed baseline has positive IoU **0.714124**, Dice **0.789096**, and
presumed-negative FPR **0.697987** (312/447). By negative source this is 116/156
original unmatched comparisons and 196/291 supplemental recorded pairs. The lower
aggregate FPR than the earlier original-pool run reflects the added comparison
population, not a matcher improvement. All 500 physical pairings are distinct;
they reuse 106 source images. The report includes all 1,000 line scores and
source/label provenance.

Saved earlier standalone engineering smoke (source fingerprint predates the BGU
batch additions; do not resume it with changed source):
`Results/Evaluation/quantitative_masks_smoke_20261031/current_run/`. It completed prepare,
6 Stage 1 and 8 Stage 2 trial records, no-op resume, frozen synthetic test with
all four methods, and CSV/LaTeX export. Nine completed records were feasible;
selection respected the .05 FPR requirement. The fixture checkpoint hash was
unchanged. These are synthetic engineering checks, not manuscript accuracy.

Current folder-mode verification: **31 quantitative/batch tests pass**, including
both folder sources, saved membership, missing-mask provenance, one-sided-mask
rejection, deterministic comparisons and manifest-free batch submission.

The earlier broader evaluation regression run recorded **72 passes and 4
failures** in unchanged notebook tests:

* `test_notebook_session_population_predictions_identical_and_gt_independent`
  assumes `current_match_settings` is in the assignment cell; it now lives in
  the next cell. The new parity test reads the actual current function.
* `test_notebook_structure_and_python_cells` hardcodes `NUM_SAMPLES = 5`; the
  current user notebook sets 10.
* `test_notebook_setup_imports_project_dependencies` executes IPython setup
  magic in a plain Python namespace without `get_ipython`.
* `test_cached_samples_threshold_sweeps_and_independent_slots` does not supply
  the notebook's current `SHOWN_REPRESENTATIONS` global.

The notebook and these tests were left unchanged to preserve unrelated work.
The verification log and machine-readable handoff are saved in
`Results/Evaluation/quantitative_masks_verification/`.
