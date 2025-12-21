import wandb
from Parameters import *

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