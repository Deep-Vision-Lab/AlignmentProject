# Training

Use only this command from the project root:

```bash
JOB_ID=<output_name> \
PRETRAINED_WEIGHTS="$PWD/Weights/cnn_bilstm/model_latest.pth" \
bash scripts/train/run_real_finetune.sh
```

The launcher submits its own Slurm job. Do not run it with `sbatch`.

## Files in this folder

- `run_real_finetune.sh` — the only public training command.
- `train_optimized.py` — the internal optimized trainer used by the launcher.

Branch selection and per-rank GPU isolation live under `training_runtime/` so the public training folder stays small and unambiguous.
