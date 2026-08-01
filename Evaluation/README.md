# Evaluation

Use only the canonical real-data evaluator:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate.sh
```

Run it from the repository root on the login node. The script submits its own one-GPU Slurm job, reconstructs the CNN+BiLSTM or ViT backend and window geometry from the checkpoint, and evaluates `high_match` and `medium_match` by default.

Optional overrides:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
LABELS=high_match \
N_SAMPLES=200 \
START_INDEX=1 \
bash Evaluation/evaluate.sh
```

Results are written under:

```text
Results/Evaluation/<model>/Real_Experiments/<run>/<label>/
```

## Minimal folder layout

Only `evaluate.sh` is a public command. The Python files are internal modules used by that command:

- `eval_img_align_sw.py`: canonical Python entrypoint and evaluation patches.
- `_eval_utils.py`: checkpoint loading and feature extraction.
- `sw_runner.py`: CLI execution and report writing.
- `sw_core.py`: Smith-Waterman scoring and visualization.
- `sw_dataset.py`: real-pair loading and split handling.
- `zero_shot_sw.py`: real-image preprocessing and ink-aware scoring.
- `vit_evaluation.py`: ViT checkpoint reconstruction while retaining CNN compatibility.
- `window_alignment.py`: score-matrix normalization used by Smith-Waterman.
- `__init__.py`: makes `Evaluation` importable as a package.

`balanced_sampling.py` was removed after its only required logic was folded into `eval_img_align_sw.py`.

Do not run the internal Python modules directly. Do not restore the old synthetic, Needleman-Wunsch, clustering, retrieval, alignment-MAE, benchmark, or visualization entrypoints.
