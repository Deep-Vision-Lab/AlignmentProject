"""Weights & Biases logging helpers."""

import os

import wandb


def init_wandb(job_id):
    """Start a normal W&B run; do not disable W&B logging features."""
    run = wandb.init(
        project="AlignmentProject",
        name=job_id,
    )
    return run


def update_wandb(epoch, train_loss, val_loss):
    """Commit exactly one W&B update per epoch containing only losses."""
    wandb.log(
        {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        step=int(epoch),
        commit=True,
    )
    print(
        f"[WANDB] logged epoch={int(epoch)} "
        f"train_loss={float(train_loss):.6f} val_loss={float(val_loss):.6f}",
        flush=True,
    )


def log_wandb_weights(job_id, epoch, weights_path):
    """Upload a saved model weights file as a W&B artifact."""
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[WANDB] Skipping weights artifact; file not found: {weights_path}", flush=True)
        return

    safe_job_id = str(job_id).replace("/", "-").replace(" ", "_")
    artifact = wandb.Artifact(
        name=f"{safe_job_id}-weights-epoch-{int(epoch):04d}",
        type="model",
        metadata={"job_id": str(job_id), "epoch": int(epoch)},
    )
    artifact.add_file(weights_path, name=f"model_epoch_{int(epoch):04d}.pth")
    wandb.log_artifact(
        artifact,
        aliases=[f"epoch-{int(epoch)}", "latest-10epoch"],
    )
    print(f"[WANDB] uploaded weights artifact for epoch={int(epoch)}: {weights_path}", flush=True)
