import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from saveDATA import *
from Parameters import *
from Evaluation import *
from DiffNWAlgo import *
from newDataLoader import *
from pathExtractor import *
from embeddingModel import *
from embeddingModel import *
from LossFunctionWithHelpers import *
from NormalizeFuncs import *
from Visualization import save_debug_visualizations

import os
import gc
import time
import wandb
import warnings

from wandb_config import init_wandb

warnings.filterwarnings("ignore")    

def interpolate_NW_matrix(NWTensor, target_shape):
    """
    Interpolates the smith matrix to match the target shape (usually alignment output shape).
    Args:
        NWTensor (torch.Tensor): Smith matrix, shape [B, H, W] or [H, W]
        target_shape (tuple): Target (H, W) shape
    Returns:
        torch.Tensor: Interpolated smith matrix, shape [B, H_new, W_new]
    """
    if NWTensor.dim() == 2:
        squeezed_NW_matrix = NWTensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif NWTensor.dim() == 3:
        squeezed_NW_matrix = NWTensor.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    else:
        raise ValueError(f"Unexpected NW matrix shape: {NWTensor.shape}")
    
    interpolated_NW_matrix = F.interpolate(squeezed_NW_matrix, size=target_shape, mode='bilinear')
    squeezed_interpolated_NW_matrix = interpolated_NW_matrix.squeeze(1)  # Remove channel dim
    
    del NWTensor, squeezed_NW_matrix, interpolated_NW_matrix
    
    return squeezed_interpolated_NW_matrix




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
def compute_batch_loss(model, image1, image2, NWTextTensor, textSimilar, DiffNWAlgo, criterion, device, 
                       loss_type='MSE', debug=False, epoch=0, batch_idx=0, dataloader_length=0, preLoss=True):
    
    tokens_a, tokens_b = model(image1, image2, show_dims=False)
    
    flip_tokens_a = torch.flip(tokens_a, dims=[1])
    flip_tokens_b = torch.flip(tokens_b, dims=[1])

    # Running the DiffNW Algorithm
    DiffNWAlgo.reset_cosine_similarity()
    diffNWimageTensor = DiffNWAlgo(x1=flip_tokens_a, x2=flip_tokens_b, show_dims=False).to(device)
    

    # Use smaller interpolation size
    new_height = min(diffNWimageTensor.shape[1], NWTextTensor.shape[1])
    new_width = min(diffNWimageTensor.shape[2], NWTextTensor.shape[2])
    new_size = (new_height, new_width)

    diffNWimageTensor = interpolate_NW_matrix(diffNWimageTensor, new_size).to(device)
    ImageSimilar = interpolate_NW_matrix(DiffNWAlgo.ImageSimilar, new_size).to(device)
    NWTextTensor = interpolate_NW_matrix(NWTextTensor, new_size).to(device)
    TextSimilar = interpolate_NW_matrix(TextSimilar, new_size).to(device)

#---------------------------------------------------------------------------------------------------------
    # Normalize matrices or paths before loss computation
    if preLoss:
        # Normalize and smooth alignment outputs
        NWTextFinal = normalize_func(NWTextTensor, normalize_type)
        diffNWimageFinal = normalize_func(diffNWimageTensor, normalize_type)
    #------------------------------------------------------------------------------------------------
    else:
        # Path extraction
        textNWpath, text_startPoints = diff_NW_Path(NWTextTensor, textSimilar,
                                                    match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore)
        NWTextFinal = NWTextTensor * textNWpath

        imageNWpath, _ = diff_NW_Path(diffNWimageTensor, cosine_sim,
                                    match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore, 
                                    position=text_startPoints)
        diffNWimageFinal = diffNWimageTensor * imageNWpath
#---------------------------------------------------------------------------------------------------------
    
    # Loss computation
    path_loss = criterion(NWTextFinal, diffNWimageFinal)
    loss_value = path_loss.item()
    
    ######################################################################################################################################
    # Debugging: Save visualizations for the last batch every 10 epochs

    if debug and batch_idx == dataloader_length - 1 and epoch % 10 == 0:
        save_debug_visualizations(model, image1, image2, tokens_a, tokens_b, 
                                  NWTextTensor, diffNWimageTensor, DiffNWAlgo, 
                                  loss_type, epoch, batch_idx)

    ######################################################################################################################################

    # Delete tensors to free memory
    del tokens_a, tokens_b
    del flip_tokens_a, flip_tokens_b
    del NWTextTensor, TextSimilar
    del diffNWimageTensor, ImageSimilar
    torch.cuda.empty_cache()
    
    return path_loss, loss_value, NWTextFinal, diffNWimageFinal


def Train(model, trainLoader, validLoader, DiffNW, criterion, loss_type, 
        device, normalize_type, epochs=100,
        learning_rate=1e-4, debug=False, 
        debug_wandb=True, show_gradients=False, preLoss=True):

    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (image1, image2, scoreMatrix, textSimilar, image1_name, image2_name) in enumerate(trainLoader):
            # Ensure all data is on correct device
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            scoreMatrix = scoreMatrix.to(device, non_blocking=True)
            textSimilar = textSimilar.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Compute loss using shared function
            path_loss, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                model, image1, image2, scoreMatrix, textSimilar, 
                DiffNW, criterion, device, loss_type=loss_type, 
                debug=debug, epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                preLoss=preLoss
            )
            
            # Compute accuracy
            batch_accuracy = compute_accuracy(NWImageFinal, NWTextFinal)
            train_accuracy += batch_accuracy
            

            train_loss += loss_value

            # Backpropagation and optimizer step
            path_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)

            # Final cleanup
            del path_loss
            del image1, image2
            del NWTextFinal, NWImageFinal
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
            for batch_idx, (image1, image2, scoreMatrix, textSimilar, image1_name, image2_name) in enumerate(validLoader):
                # Ensure all data is on correct device
                image1 = image1.to(device)
                image2 = image2.to(device)
                scoreMatrix = scoreMatrix.to(device)
                textSimilar = textSimilar.to(device)
                
                # Compute loss using shared function
                _, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                    model, image1, image2, scoreMatrix, textSimilar, DiffNW, criterion, device
                )
                
                # Compute accuracy
                batch_accuracy = compute_accuracy(NWImageFinal, NWTextFinal)
                val_accuracy += batch_accuracy
                
                del NWTextFinal, NWImageFinal
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
    if debug_wandb:
        init_wandb()

    cnn_transformer_model = EmbeddingModel(
        window_size=window_size,
        stride=window_size//2,
        vector_size=vector_size,
        model_arch=model_arch,
        device=device
    )

    DiffNW = DiffNWAlgo(match_score=matchScore, miss_score=mismatchScore, 
                        gap=gapScore)
    
    criterion = Loss_choice(loss_type)
    
    # try:
    loss_lst = Train(
        cnn_transformer_model,
        train_dataloader,
        valid_dataloader,
        DiffNW,
        criterion,
        loss_type,
        device,
        normalize_type,
        epochs,
        learning_rate,
        debug,
        debug_wandb,
        show_gradients,
        preLoss=preLoss
    )
    
    if debug_wandb:
        wandb.finish()