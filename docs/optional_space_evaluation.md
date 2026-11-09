# Optional-space evaluation

For the current image-to-image decoder and A/B/C metrics, see
[image-only word verification](word_alignment_evaluation.md). The checkpoint
records training on all three books. The recurrence details below remain the
checkpoint-faithful transcript diagnostics.

Both notebooks use `Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt`
(epoch 20). Its saved configuration contains `space_mode=optional`,
`space_skip_penalty_ratio=1.0`, and `dtw_normalization=aligned_mean`.
Its training-only prior contains 37 identities, with space at index zero in the
alphabet. Checkpoint reconstruction retains both new configuration fields;
older checkpoints default to ignoring spaces.

Transcript diagnostics normalize Arabic letters as before, collapse whitespace
between nonempty Arabic words to one space, and omit leading/trailing spaces.
Recognition uses the checkpoint alphabet instead of all Arabic Unicode letters.
Space is an ordinary codebook identity, separate from padding and CTC blank.

The training implementation comes from commit `de0bac3` (pulled from
`origin/refactor/simple-alignment-core`). Evaluation uses its
`clean_alignment_tokens`, `alignment_tokens`, and `effective_alphabet` functions,
and its Soft-DTW and negative-objective code directly.

The hard diagnostic implements the same recurrence with an argmin backtrace.
For a space column, it can advance from the preceding transcript column on the
same visual row without paying the current cell cost or consuming another
window. That transition emits no alignment cell. A matched space can repeat
vertically across several windows. Mandatory letters retain the original DTW
moves. Horizontal suppression uses the full transcript length, including spaces,
exactly as in training. Existing character-box positions count Arabic letters;
they map to the corresponding space-aware transcript columns. Explicit
token-indexed boxes can set `character_box_positions='tokens'`.

The saved skip penalty is `space_skip_penalty_ratio * log(alphabet_size)`.
Optional-space checkpoints require full-alphabet NLL and `aligned_mean`.
The penalty affects path selection and free energy; `aligned_mean` reports the
occupancy-weighted cell NLL, excluding transition penalties. Loss evaluation
reports the training implementation's expected number and fraction of skipped
spaces. Hard recognition reports skipped/matched transcript space positions,
and labels spaces explicitly in tables and heatmaps.

Hard/soft parity is tested against the pulled training recurrence at small
soft-min temperature, covering variable grids, skip masks, position priors, and
horizontal-suppression settings. Old checkpoints retain `space_mode=off`.

The 15 previous book-pair examples were re-evaluated with the new checkpoint,
the existing matcher settings, and refreshed split membership. They remain the
same fixed examples; they were not ranked again using the new model. Their
new figures and metrics are embedded in both notebooks. Local inference artifacts
are in `Results/Evaluation/real_abc_optional_space_fixed_examples/`, including
`report.json`, `recognition_smoke.json`, and `loss_smoke.json`. Previous artifacts remain in their
original directory. Other cached notebook outputs were cleared when switching
checkpoints. Restart the kernel and run the setup/session cells to refresh an
already open notebook.

Focused tests cover invisible spaces, one-window spaces, multi-window spaces,
mandatory letters, legacy behavior, finite soft occupancy and gradients, and
typed-negative position mapping. A real saved-test pair was also checked with
the checkpoint's 37-token alphabet and all 63 valid windows.
The real hard/soft normalized path scores agree within `6.5e-8`. A separate
saved-test line containing a space completed the full positive/typed-negative
loss on MPS; its smoke report is explicitly a one-line diagnostic. The full
regression suite passed: 330 tests and 7 subtests.
