"""Minimal Weights & Biases logging: epoch-level losses only."""

import wandb


def init_wandb(job_id):
    """Start a run without uploading configuration or artifacts."""
    wandb.init(project="AlignmentProject", name=job_id)


def update_wandb(epoch, train_loss, val_loss):
    """Commit exactly one W&B update per epoch containing only losses."""
    wandb.log(
        {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        step=int(epoch),
    )
