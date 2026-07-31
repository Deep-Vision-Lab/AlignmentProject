# Evaluation

Use only the canonical real-data evaluator:

```bash
WEIGHTS="$PWD/Weights/<job_id>/model_best.pth" \
bash Evaluation/evaluate.sh
```

Run it from the repository root on the login node. The script submits its own one-GPU Slurm job, reconstructs the model backend and window geometry from the checkpoint, and evaluates both `high_match` and `medium_match` by default.

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
Results/Evaluation/<model>/<run>/<label>/
```

## Internal modules

Do not run the Python files in this directory directly. `Evaluation/evaluate.sh` calls `eval_img_align_sw.py`, which uses the `sw_*`, checkpoint-loading, preprocessing, and balanced-sampling helper modules.

The old Needleman–Wunsch, clustering, retrieval, alignment-MAE, standalone heatmap, `.sbatch`, and secondary shell entrypoints were removed.
