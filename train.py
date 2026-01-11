import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from saveDATA import *
from Parameters import *
from Evaluation import *
from DiffNWAlgo import *
from wandb_config import *
from newDataLoader import *
from pathExtractor import *
from embeddingModel import *
from embeddingModel import *
from LossFunctionWithHelpers import *
from NormalizeFuncs import *
from Visualization import *

import os
import gc
import time
import wandb
import warnings

from wandb_config import init_wandb, update_wandb

warnings.filterwarnings("ignore")    

def interpolate_NW_matrix(NWTensor, target_shape):
    if NWTensor.dim() == 2:
        squeezed_NW_matrix = NWTensor.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif NWTensor.dim() == 3:
        squeezed_NW_matrix = NWTensor.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    else:
        raise ValueError(f"Unexpected NW matrix shape: {NWTensor.shape}")
    
    interpolated_NW_matrix = F.interpolate(squeezed_NW_matrix, size=target_shape, mode='nearest')
    squeezed_interpolated_NW_matrix = interpolated_NW_matrix.squeeze(1)  # Remove channel dim
    
    del NWTensor, squeezed_NW_matrix, interpolated_NW_matrix
    
    return squeezed_interpolated_NW_matrix



def compute_accuracy(pred_path, target_path, threshold=0.5):
    # Binarize the paths
    pred_binary = (pred_path > threshold).float()
    target_binary = (target_path > threshold).float()
    
    # Calculate accuracy as the percentage of matching positions
    correct = (pred_binary == target_binary).float()
    accuracy = correct.sum().item() / correct.numel()
    
    return accuracy


def save_model_weights(model, epoch):
    weights_dir = os.path.join(os.path.dirname(__file__), "Weights", loss_type, model_arch)
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), weights_path)

def check_grad(grad):
    if grad is None:
        print("Gradient is None!")
    elif torch.isnan(grad).any():
        print("Gradient became NaN!")
    else:
        print(f"Gradient flowing... Sum: {grad.sum().item():.5f}")

# second line of parameters is for saving visualizations during debugging 
def compute_batch_loss(model, text1, text2, image1, image2, NWTextTensor, textSimilar, 
                       DiffNWAlgo, criterion, epoch=0, batch_idx=0, dataloader_length=0, 
                       debug=False):
    
    tokens_a, tokens_b = model(image1, image2, show_dims=False)
    
    flip_tokens_a = torch.flip(tokens_a, dims=[-2])
    flip_tokens_b = torch.flip(tokens_b, dims=[-2])

    # Running the DiffNW Algorithm
    DiffNWAlgo.reset_cosine_similarity()
    diffNWimageTensor = DiffNWAlgo(x1=flip_tokens_a, x2=flip_tokens_b, show_dims=False).to(device)
    

    # Interpolate NW matrices to match sizes
    new_size = (NWTextTensor.shape[-2], NWTextTensor.shape[-1])

    diffNWimageTensor = interpolate_NW_matrix(diffNWimageTensor, new_size).to(device)
    Interpolated_ImageSimilar = interpolate_NW_matrix(DiffNWAlgo.ImageSimilar, new_size).to(device)

    NWTextTensor = interpolate_NW_matrix(NWTextTensor, new_size).to(device)
    Interpolated_TextSimilar = interpolate_NW_matrix(textSimilar, new_size).to(device)

#---------------------------------------------------------------------------------------------------------
    # Normalize matrices or paths before loss computation
    if preprocess == 'Normalize':
        # Normalize and smooth alignment outputs
        NWTextTensor = normalize_func(NWTextTensor, normalize_type)
        diffNWimageTensor = normalize_func(diffNWimageTensor, normalize_type)
    #------------------------------------------------------------------------------------------------
    # Path extraction
    elif preprocess == 'ExtractPaths':
        # Extract paths using Needleman-Wunsch algorithm
        if Regular_ScoreMatrix_Load:
            textNWpath, text_startPoints = NW_Path(NWTextTensor, Interpolated_TextSimilar,
                                                        match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore)
        else:
        # Extract paths using Differentiable Needleman-Wunsch algorithm
            textNWpath, text_startPoints = diff_NW_Path(NWTextTensor, Interpolated_TextSimilar,
                                                        match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore)
        NWTextTensor = NWTextTensor * textNWpath

        imageNWpath, _ = diff_NW_Path(diffNWimageTensor, Interpolated_ImageSimilar,
                                    match_score=matchScore, miss_score=mismatchScore, gap_penalty=gapScore, 
                                    position=text_startPoints)
        diffNWimageTensor = diffNWimageTensor * imageNWpath
#---------------------------------------------------------------------------------------------------------
    
    # Loss computation
    loss = criterion(NWTextTensor, diffNWimageTensor, lamda=1.0) if loss_type == 'HeightDiff' else criterion(NWTextTensor, diffNWimageTensor)
    loss_value = loss.item()
    
    ######################################################################################################################################
    # Debugging: Save visualizations for the last batch every 10 epochs
    if debug and batch_idx == 0 and epoch % 10 == 0:
        original_diffNWImageSimilar = DiffNWAlgo.ImageSimilar
        interpolated_diffNWImageSimilar = Interpolated_ImageSimilar
        original_NWTextSimilar = textSimilar
        interpolated_NWTextSimilar = Interpolated_TextSimilar

        save_debug_visualizations(
            model, 
            text1, text2, 
            image1, image2, 
            tokens_a, tokens_b, 
            NWTextTensor, diffNWimageTensor,
            original_diffNWImageSimilar, interpolated_diffNWImageSimilar,
            original_NWTextSimilar, interpolated_NWTextSimilar,
            epoch, batch_idx
        )
        
        del original_diffNWImageSimilar
        del interpolated_diffNWImageSimilar
        del original_NWTextSimilar
        del interpolated_NWTextSimilar
        
        # Save model weights
        save_model_weights(model, epoch)

    ######################################################################################################################################

    # Delete tensors to free memory
    del tokens_a, tokens_b
    del flip_tokens_a, flip_tokens_b
    del Interpolated_TextSimilar
    del Interpolated_ImageSimilar
    torch.cuda.empty_cache()
    
    return loss, loss_value, NWTextTensor, diffNWimageTensor



def Train(model, trainLoader, validLoader, DiffNW, criterion):
    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (image1, image2, scoreMatrix, textSimilar, text1, text2, image1_name, image2_name) in enumerate(trainLoader):
            # Ensure all data is on correct device
            image1 = image1.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
            scoreMatrix = scoreMatrix.to(device, non_blocking=True)
            textSimilar = textSimilar.to(device, non_blocking=True)
            text1 = list(text1)
            text2 = list(text2)
            optimizer.zero_grad()
            
            # Compute loss using shared function
            path_loss, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                model, text1, text2, image1, image2, 
                scoreMatrix, textSimilar, DiffNW, 
                criterion, epoch=epoch, batch_idx=batch_idx, dataloader_length=len(trainLoader),
                debug=debug
            )
            
            # Compute accuracy
            batch_accuracy = compute_accuracy(NWImageFinal, NWTextFinal)
            train_accuracy += batch_accuracy
            

            train_loss += loss_value

            # Backpropagation and optimizer step
            path_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)
            # Optionally print gradients for debugging
            if show_gradients:
                if image1.grad is not None and image2.grad is not None:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1} - image1 gradient: {image1.grad.sum():.4f}, image2 gradient: {image2.grad.sum():.4f}", flush=True)

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
            for batch_idx, (image1, image2, scoreMatrix, textSimilar, text1, text2, image1_name, image2_name) in enumerate(validLoader):
                # Ensure all data is on correct device
                image1 = image1.to(device)
                image2 = image2.to(device)
                scoreMatrix = scoreMatrix.to(device)
                textSimilar = textSimilar.to(device)
                
                # Compute loss using shared function
                _, loss_value, NWTextFinal, NWImageFinal = compute_batch_loss(
                    model, 
                    text1, text2, 
                    image1, image2, 
                    scoreMatrix, textSimilar, 
                    DiffNW, criterion
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
            update_wandb(
                train_loss, 
                val_loss, 
                train_accuracy, 
                val_accuracy
            )
        loss_lst.append(train_loss)
        
        # Set model back to training mode
        model.train()

    return loss_lst



if __name__ == '__main__':
    if debug_wandb:
        init_wandb()

    model = EmbeddingModel(
        window_size=window_size,
        stride=window_size,
        vector_size=vector_size,
        model_arch=model_arch,
        device=device
    )
    if show_gradients:
        for param in model.parameters():
            param.register_hook(check_grad)

    DiffNW = DiffNWAlgo(
        match_score=matchScore, 
        miss_score=mismatchScore, 
        gap=gapScore,
    )
    
    criterion = Loss_choice(loss_type)
    
    # try:
    loss_lst = Train(
        model,
        train_dataloader,
        valid_dataloader,
        DiffNW,
        criterion
    )
    
    
    if debug_wandb:
        wandb.finish()