import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from saveDATA import *
from Evaluation import *
from DiffSWAlgo import *
from newDataLoader import *
from pathExtractor import *
from embeddingModel import *
from embeddingModel import *
from LossFunctionWithHelpers import *

import os
import gc
import time
import wandb
import warnings

warnings.filterwarnings("ignore")    

def interpolate_SW_matrix(SW_matrix, target_shape):
    """
    Interpolates the smith matrix to match the target shape (usually alignment output shape).
    Args:
        current_smith_matrix (torch.Tensor): Smith matrix, shape [B, H, W] or [H, W]
        target_shape (tuple): Target (H, W) shape
    Returns:
        torch.Tensor: Interpolated smith matrix, shape [B, H_new, W_new]
    """
    if SW_matrix.dim() == 2:
        squeezed_SW_matrix = SW_matrix.unsqueeze(0).unsqueeze(0)
    elif SW_matrix.dim() == 3:
        squeezed_SW_matrix = SW_matrix.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected smith matrix shape: {SW_matrix.shape}")
    
    interpolated_SW_matrix = F.interpolate(squeezed_SW_matrix, size=target_shape, mode='bilinear')
    squeezed_interpolated_SW_matrix = interpolated_SW_matrix.squeeze(1)
    
    del SW_matrix, squeezed_SW_matrix, interpolated_SW_matrix
    
    return squeezed_interpolated_SW_matrix



def smooth_and_normalize_matrix(matrix, normalize_type):
    """
    Optionally smooth and normalize the alignment output tensor.
    Args:
        matrix (torch.Tensor): The alignment output tensor.
        normalize_type (str): Normalization type: 'min_max', 'mean_std', or ''.
    Returns:
        torch.Tensor: Smoothed and normalized alignment output.
    """
    matrix = smooth_path(matrix)
    if normalize_type == 'min_max':
        alignment_min = matrix.min()
        alignment_max = matrix.max()
        matrix = 2 * (matrix - alignment_min) / (alignment_max - alignment_min + 1e-8) - 1
    elif normalize_type == 'mean_std':
        alignment_mean = matrix.mean(dim=(1, 2), keepdim=True)
        alignment_std = matrix.std(dim=(1, 2), keepdim=True)
        matrix = (matrix - alignment_mean) / (alignment_std + 1e-8)
        del alignment_mean, alignment_std
    return matrix




def compute_accuracy(pred_path, target_path, threshold=0.5):
    """
    Compute accuracy by measuring path agreement between predicted and target alignment paths.
    
    Args:
        pred_path (torch.Tensor): Predicted alignment path, shape [B, H, W]
        target_path (torch.Tensor): Target alignment path, shape [B, H, W]
        threshold (float): Threshold for binarizing paths (default: 0.5)
    
    Returns:
        accuracy (float): Percentage of matching alignment positions
    """
    # Binarize the paths
    pred_binary = (pred_path > threshold).float()
    target_binary = (target_path > threshold).float()
    
    # Calculate accuracy as the percentage of matching positions
    correct = (pred_binary == target_binary).float()
    accuracy = correct.sum().item() / correct.numel()
    
    return accuracy


# second line of parameters is for saving visualizations during debugging 
def compute_batch_loss(model, image1, image2, diffSWText, textSimilar, DiffSW, criterion, device, 
                       loss_type='MSE', debug=False, epoch=0, batch_idx=0, dataloader_length=0):
    
    tokens_a, tokens_b = model(image1, image2, show_dims=False)
    
    flip_tokens_a = torch.flip(tokens_a, dims=[1])
    flip_tokens_b = torch.flip(tokens_b, dims=[1])

    # Running the DiffSW Algorithm
    DiffSW.reset_cosine_similarity()
    diffSWimage = DiffSW(x1=flip_tokens_a, x2=flip_tokens_b, show_dims=False).to(device)
    

    # Use smaller interpolation size
    new_size = min(16, diffSWText.shape[-1])
    new_size = (new_size, new_size)

    diffSWimage = interpolate_SW_matrix(diffSWimage, new_size).to(device)
    cosine_sim = interpolate_SW_matrix(DiffSW.cosine_similarity, new_size).to(device)
    diffSWText = interpolate_SW_matrix(diffSWText, new_size).to(device)
    textSimilar = interpolate_SW_matrix(textSimilar, new_size).to(device)

    # Path extraction
    textSWpath, text_startPoints = diff_SW_Path(diffSWText, textSimilar,
                                                match_score=2, miss_score=-3, gap_penalty=-1)
    diffSWText_final = diffSWText * textSWpath

    imageSWpath, _ = diff_SW_Path(diffSWimage, cosine_sim,
                                match_score=2, miss_score=-3, gap_penalty=-1, 
                                position=text_startPoints)
    diffSWimage_final = diffSWimage * imageSWpath

    # Loss computation
    path_loss = criterion(diffSWText_final, diffSWimage_final)
    loss_value = path_loss.item()
    
    ######################################################################################################################################
    # Debugging: Save visualizations for the last batch every 10 epochs

    if debug and batch_idx == dataloader_length - 1 and (epoch + 1) % 10 == 0: # Save visualizations for last batch every 10 epochs
        # Prepare directories for saving visualizations
        vectors_epoch_dir = f'TrainResults/{loss_type}/VectorsPerEpoch/{model.model_arch}/Epoch_{epoch+1}'
        matrices_epoch_dir = f'TrainResults/{loss_type}/ScoreMatricesPerEpoch/{model.model_arch}/Epoch_{epoch+1}'
        os.makedirs(vectors_epoch_dir, exist_ok=True)
        os.makedirs(matrices_epoch_dir, exist_ok=True)

        # Clone tensors for visualization to avoid affecting gradients
        debug_image1 = image1.detach().cpu()
        debug_image2 = image2.detach().cpu()
        debug_tokens_a = tokens_a.detach().cpu()
        debug_tokens_b = tokens_b.detach().cpu()
        debug_diffSWText = diffSWText.detach().cpu()
        debug_diffSWimage = diffSWimage.detach().cpu()

        # debug_diffSWText is only available during training if calc_cosine=False in Alignment
        if hasattr(DiffSW, 'similarity_matrix'):
            debug_SWTextSimilar = DiffSW.similarity_matrix.clone().detach()
        else:
            debug_SWTextSimilar = None

        # original_text1_batch and original_text2_batch are only available during training if calc_cosine=False in Alignment
        if hasattr(DiffSW, 'original_text1_batch') and hasattr(DiffSW, 'original_text2_batch'):
            original_text1_batch = DiffSW.original_text1_batch
            original_text2_batch = DiffSW.original_text2_batch
        else:
            original_text1_batch = None
            original_text2_batch = None

        # Save heatmap visualizations
        saveHeatmapPlots(model, debug_image1, debug_image2, 
                        debug_tokens_a, debug_tokens_b, 
                        vectors_epoch_dir, epoch, batch_idx,
                        debug_diffSWimage, 
                        debug_diffSWText, 
                        debug_SWTextSimilar, 
                        matrices_epoch_dir, original_text1_batch,
                        original_text2_batch)

        del debug_image1, debug_image2
        del debug_tokens_a, debug_tokens_b
        del debug_diffSWimage, debug_diffSWText
        del vectors_epoch_dir, matrices_epoch_dir
        del original_text1_batch, original_text2_batch
        if debug_SWTextSimilar is not None:
            del debug_SWTextSimilar

    ######################################################################################################################################

    # Delete tensors to free memory
    del tokens_a, tokens_b
    del flip_tokens_a, flip_tokens_b
    del diffSWText, textSWpath, textSimilar
    del diffSWimage, imageSWpath, cosine_sim, text_startPoints
    torch.cuda.empty_cache()
    
    return path_loss, loss_value, diffSWText_final, diffSWimage_final


def Train(model, trainLoader, validLoader, DiffSW, criterion, loss_type, 
        device, normalize_type, epochs=100,
        learning_rate=1e-4, debug=False, 
        debug_wandb=True, show_gradients=False):

    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (image1, image2, diffSWText, textSimilar, image1_name, image2_name) in enumerate(trainLoader):
            # Ensure all data is on correct device
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            diffSWText = diffSWText.to(device, non_blocking=True)
            textSimilar = textSimilar.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            begin_time = time.time()
            
            # Compute loss using shared function
            path_loss, loss_value, diffSWText_final, diffSWimage_final = compute_batch_loss(
                model, image1, image2, diffSWText, textSimilar, DiffSW, criterion, device,
                loss_type=loss_type, debug=debug, epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader)
            )
            
            # Compute accuracy
            batch_accuracy = compute_accuracy(diffSWimage_final, diffSWText_final)
            train_accuracy += batch_accuracy
            

            train_loss += loss_value

            # Backpropagation and optimizer step
            path_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)

            # Final cleanup
            del path_loss
            del image1, image2
            del diffSWText_final, diffSWimage_final
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1} completed. Average Loss: {train_loss / len(trainLoader):.4f}", flush=True)
        train_loss = train_loss / len(trainLoader)
        train_accuracy = train_accuracy / len(trainLoader)
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}', flush=True)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch_idx, (image1, image2, diffSWText, textSimilar, image1_name, image2_name) in enumerate(validLoader):
                # Ensure all data is on correct device
                image1 = image1.to(device)
                image2 = image2.to(device)
                diffSWText = diffSWText.to(device)
                textSimilar = textSimilar.to(device)
                
                # Compute loss using shared function
                _, loss_value, diffSWText_final, diffSWimage_final = compute_batch_loss(
                    model, image1, image2, diffSWText, textSimilar, DiffSW, criterion, device
                )
                
                # Compute accuracy
                batch_accuracy = compute_accuracy(diffSWimage_final, diffSWText_final)
                val_accuracy += batch_accuracy
                
                del diffSWText_final, diffSWimage_final
                val_loss += loss_value

                # Final cleanup
                del image1, image2
                torch.cuda.empty_cache()
        
        val_loss = val_loss / len(validLoader)
        val_accuracy = val_accuracy / len(validLoader)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}', flush=True)
        
        # Log train and validation losses and accuracies to wandb
        if debug_wandb:
            wandb.log({
                "train_loss": train_loss, 
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            })
        loss_lst.append(train_loss)
        
        # Set model back to training mode
        model.train()

    return loss_lst



if __name__ == '__main__':
    loss_type = 'MSE' # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 'Dice', 'Wasserstein']
    model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'Transformer']
    window_size = 64
    vector_size = 128
    normalize_type = '' # ['min_max', 'mean_std']
    epochs = 300
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = True # Set to True to save patches and heatmaps for debugging
    debug_wandb = True # Set to True to log training to Weights & Biases
    show_gradients = True # Set to True to print gradients for debugging
    
    if debug_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="AlignmentProject",
            name=f"Train model {window_size} - {model_arch} - {loss_type} - {normalize_type} - AMP",
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

    cnn_transformer_model = EmbeddingModel(
        window_size=window_size,
        stride=window_size//2,
        vector_size=vector_size,
        model_arch=model_arch,
        use_checkpointing=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )  # In Train.py

    DiffSW = DiffSWAlgo(match_score=2, miss_score=-3, gap=-1)
    
    if loss_type == 'HeightDiff':
        criterion = HeightDiff_loss
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'GuidedAttention':
        criterion = guided_attention_loss
    elif loss_type == 'KL-Divergence':
        criterion = kl_divergence_loss
    elif loss_type == 'Dice':
        criterion = dice_loss
    elif loss_type == 'Wasserstein':
        criterion = wasserstein_distance
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    # try:
    loss_lst = Train(
        cnn_transformer_model,
        train_dataloader,
        valid_dataloader,
        DiffSW,
        criterion,
        loss_type,
        device,
        normalize_type,
        epochs,
        learning_rate,
        debug,
        debug_wandb,
        show_gradients
    )
    if debug_wandb:
        wandb.finish()