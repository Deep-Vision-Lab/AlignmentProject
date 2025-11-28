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

def saveHeatmapPlots(model, image1, image2, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx, 
                    debug_diffSWimage, debug_diffSWText, debug_SWTextSimilar,
                    matrices_epoch_dir, original_text1_batch, original_text2_batch):
    for i in range(image1.size(0)): # Iterate through items in the current batch
        # Save token vectors
        # tokens_a and tokens_b are already flipped and on the correct device
        print_elements(tokens_a[i], f'{vectors_epoch_dir}/tokens_a_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        print_elements(tokens_b[i], f'{vectors_epoch_dir}/tokens_b_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        # Generate patches for visualization
        image1_i = image1[i].unsqueeze(0)
        image2_i = image2[i].unsqueeze(0)
        model_window_size = model.window_size
        model_stride = model.stride
        windows_img1 = sliding_window(image1_i, model_window_size, model_stride).squeeze(0)
        windows_img2 = sliding_window(image2_i, model_window_size, model_stride).squeeze(0)

        y_heatmap = torch.flip(windows_img1, dims=[0]) # From image1, for Y-axis
        x_heatmap = torch.flip(windows_img2, dims=[0]) # From image2, for X-axis

        # Extract paths using diff_SW_Path for visualization
        textPath, _ = SW_Path(debug_diffSWText[i:i+1], debug_diffSWText[i:i+1], match_score=7, miss_score=-3, gap_penalty=-1)
        imagePath, _ = diff_SW_Path(debug_diffSWimage[i:i+1], debug_diffSWimage[i:i+1], match_score=7, miss_score=-3, gap_penalty=-1)
        
        # Visualize paths instead of raw matrices
        visualize_heatmaps(
            imagePath[0],  # Use extracted path
            textPath[0],   # Use extracted path
            f"{matrices_epoch_dir}/heatmaps_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png",
            patches_y=y_heatmap,
            patches_x=x_heatmap
        )

        # Only run the following if debug_SWTextSimilar is not None
        if debug_SWTextSimilar is not None:
            # New visualization: Raw Smith-Waterman matrix (before interpolation)
            # with original character sequences as axes. seq1 on X, seq2 on Y.
            debug_SWTextSimilar_i = debug_SWTextSimilar[i] # Shape [H_orig, W_orig]
            original_text1_batch_i = original_text1_batch[i] # string for seq1
            original_text2_batch_i = original_text2_batch[i] # string for seq2

            # Use SW_Path instead of makeTracerouteMatrixBinary
            smith_original_path_tensor, _ = SW_Path(debug_SWTextSimilar_i.unsqueeze(0), 
                                                   debug_SWTextSimilar_i.unsqueeze(0),
                                                   match_score=7, miss_score=-3, gap_penalty=-1)
            smith_original_path_np = smith_original_path_tensor[0].cpu().numpy()

            # SW matrix has text1 along rows, text2 along columns.
            # To put text1 (seq1) on X-axis and text2 (seq2) on Y-axis, we need to transpose.
            scores_matrix_to_plot = debug_SWTextSimilar_i.T
            path_matrix_to_plot = torch.tensor(smith_original_path_np).T

            # Labels for axes: X-axis for seq1 (original_text1), Y-axis for seq2 (original_text2)
            x_char_labels = ['ø'] + list(original_text1_batch_i.replace(" ", "")) # 'ø' for the initial empty string state
            y_char_labels = ['ø'] + list(original_text2_batch_i.replace(" ", ""))
            filename_char_level_smith_dual = f"{matrices_epoch_dir}/raw_char_smith_scores_path_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png"
            visualize_dual_char_heatmaps(
                scores_matrix_to_plot,
                path_matrix_to_plot,
                x_labels=x_char_labels,
                y_labels=y_char_labels,
                image_path=filename_char_level_smith_dual,
                title1=f"Raw SW Scores (Seq1 vs Seq2) - Item {i}",
                title2=f"Raw SW Path (Seq1 vs Seq2) - Item {i}"
            )
            del debug_SWTextSimilar_i, original_text1_batch_i, original_text2_batch_i
            del smith_original_path_tensor, smith_original_path_np
            del scores_matrix_to_plot, path_matrix_to_plot
            del x_char_labels, y_char_labels, filename_char_level_smith_dual
        del windows_img1, windows_img2, y_heatmap, x_heatmap
        del textPath, imagePath
        del image1_i, image2_i
            

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


def compute_batch_loss(model, image1, image2, diffSWText, textSimilar, DiffSW, criterion, device):
    """
    Compute loss for a single batch (used in both training and validation).
    
    Args:
        model: The embedding model
        image1, image2: Input images
        diffSWText, textSimilar: Text similarity matrices
        DiffSW: DiffSW algorithm instance
        criterion: Loss function
        device: Device to run computation on
    
    Returns:
        path_loss: Loss tensor (for backprop)
        loss_value (float): Computed loss value
        diffSWText_final: Final text alignment path
        diffSWimage_final: Final image alignment path
    """
    # Forward pass
    tokens_a, tokens_b = model(image1, image2, show_dims=False)
    
    flip_tokens_a = torch.flip(tokens_a, dims=[1])
    flip_tokens_b = torch.flip(tokens_b, dims=[1])

    # Immediately delete original tokens
    del tokens_a, tokens_b
    torch.cuda.empty_cache()

    # Running the DiffSW Algorithm
    DiffSW.reset_cosine_similarity()
    diffSWimage = DiffSW(x1=flip_tokens_a, x2=flip_tokens_b, show_dims=False).to(device)
    
    # Delete flip tokens immediately after use
    del flip_tokens_a, flip_tokens_b
    torch.cuda.empty_cache()

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
    del diffSWText, textSWpath, textSimilar

    imageSWpath, _ = diff_SW_Path(diffSWimage, cosine_sim,
                                match_score=2, miss_score=-3, gap_penalty=-1, 
                                position=text_startPoints)
    diffSWimage_final = diffSWimage * imageSWpath
    del diffSWimage, imageSWpath, cosine_sim, text_startPoints

    # Loss computation
    path_loss = criterion(diffSWText_final, diffSWimage_final)
    
    loss_value = path_loss.item()
    
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
                model, image1, image2, diffSWText, textSimilar, DiffSW, criterion, device
            )
            
            # Compute accuracy
            batch_accuracy = compute_accuracy(diffSWimage_final, diffSWText_final)
            train_accuracy += batch_accuracy
            
            del diffSWText_final, diffSWimage_final
            
            end_time = time.time()
            if end_time - begin_time > 60:
                print(f"⚠️ Warning: Batch computation took {end_time - begin_time:.2f} seconds in batch {batch_idx}", flush=True)
                print(f"Image1 name: {image1_name}", flush=True)  
                print(f"Image2 name: {image2_name}", flush=True)

            train_loss += loss_value

            # Backpropagation and optimizer step
            path_loss.backward()
            optimizer.step()
            del path_loss

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {train_loss / (batch_idx + 1):.4f}", flush=True)

            # Final cleanup
            del image1, image2
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
                image1 = image1.to(device, non_blocking=True)
                image2 = image2.to(device, non_blocking=True)
                diffSWText = diffSWText.to(device, non_blocking=True)
                textSimilar = textSimilar.to(device, non_blocking=True)
                
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