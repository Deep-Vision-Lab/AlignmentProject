# Training entrypoint

Run only:

```bash
JOB_ID=<output_name> \
PRETRAINED_WEIGHTS="$PWD/Weights/ViT/model_latest.pth" \
bash scripts/train/run_real_finetune.sh
```

Do not run the other files in this directory directly. They are internal runtime components used by `run_real_finetune.sh`:

- `train_model.py` selects the active branch model.
- `train_optimized.py` initializes DDP, data loading, losses, and checkpointing.
- `run_rank_isolated.sh` isolates one GPU per rank.

The public launcher submits its own Slurm job. Do not invoke it with `sbatch`.
