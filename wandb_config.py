import wandb
import os
from Parameters import *

# Store artifact references to update them instead of creating new ones
_weights_artifact = None
_results_artifact = None

def init_wandb():
    wandb.init(
            # set the wandb project where this run will be logged
            project="AlignmentProject",
            name=f"Train model {window_size} - {model_arch} - {loss_type} - {normalize_type}",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "vector size": vector_size,
                "loss": loss_type,
                "architecture": model_arch,
                "epochs": epochs,
                "slicing_window_width": window_size,
                "normalizing method ": normalize_type
            })
    
def update_wandb(train_loss, val_loss, train_accuracy, val_accuracy):
    wandb.log({
    "train_loss": train_loss, 
    "val_loss": val_loss,
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy
    })

def upload_artifacts_to_wandb(job_id, epoch):
    """
    Upload weights and TrainResults to wandb as artifacts.
    Uses the same artifact name to update instead of creating new ones.
    """
    global _weights_artifact, _results_artifact
    
    weights_dir = os.path.join(os.path.dirname(__file__), "Weights", loss_type, job_id)
    results_dir = os.path.join(os.path.dirname(__file__), "TrainResults", loss_type, job_id)
    
    # Upload weights artifact (update existing)
    if os.path.exists(weights_dir):
        weights_artifact = wandb.Artifact(
            name=f"weights-{job_id}",
            type="model",
            description=f"Model weights at epoch {epoch}"
        )
        weights_artifact.add_dir(weights_dir)
        wandb.log_artifact(weights_artifact)
        print(f"Uploaded weights to wandb (epoch {epoch})")
    
    # Upload TrainResults artifact (update existing)
    if os.path.exists(results_dir):
        results_artifact = wandb.Artifact(
            name=f"train-results-{job_id}",
            type="results",
            description=f"Training visualizations at epoch {epoch}"
        )
        results_artifact.add_dir(results_dir)
        wandb.log_artifact(results_artifact)
        print(f"Uploaded TrainResults to wandb (epoch {epoch})")