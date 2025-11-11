import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from DiffSWAlgo import DiffSWAlgo
from embeddingModel import *
from DiffSWAlgo import *
from newDataLoader import *
from pathExtractor import *
from saveDATA import *
from LossFunctionWithHelpers import *
from Evaluation import *
from embeddingModel import *

import os
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
        interpolated_smith_matrix (torch.Tensor): The smith matrix after interpolation (for scaling).
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



import torch
import time

def Train(model, trainLoader, criterion, loss_type, device, normalize_type, epochs=100,
        learning_rate=1e-4, debug=False, gradient_accumulation_steps=1, 
        debug_wandb=True, show_gradients=False):

    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_lst = []
    
    # Enable memory monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(epochs):
        epoch_loss = 0
        accumulated_loss = 0

        for batch_idx, (image1, image2, diffSWText, textSimilar, image1_name, image2_name) in enumerate(trainLoader):
            try:
                # Monitor memory at start
                if torch.cuda.is_available() and batch_idx % 5 == 0:
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"Batch {batch_idx}: Current: {current_mem:.2f}GB, Peak: {peak_mem:.2f}GB", flush=True)
                
                # Ensure all data is on correct device
                image1 = image1.to(device, non_blocking=True)
                image2 = image2.to(device, non_blocking=True)
                diffSWText = diffSWText.to(device, non_blocking=True)
                textSimilar = textSimilar.to(device, non_blocking=True)
                
                if batch_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    # Clear any cached gradients
                    torch.cuda.empty_cache()
                
                tokens_a, tokens_b = model(image1, image2, show_dims=False)
                flip_tokens_a = torch.flip(tokens_a, dims=[1])
                flip_tokens_b = torch.flip(tokens_b, dims=[1])

                # Immediately delete original tokens
                del tokens_a, tokens_b
                torch.cuda.empty_cache()

                begin_time = time.time()
                # Debug: Save original text sequences for visualization

                # JAX computation
                DiffSW = DiffSWAlgo(match_score=2, miss_score=-3, gap=-1).to(device)
                diffSWimage = DiffSW(x1=flip_tokens_a, x2=flip_tokens_b)
                
                end_time = time.time()
                if end_time - begin_time > 60:
                    print(f"⚠️ Warning: DiffSW computation took {end_time - begin_time:.2f} seconds in batch {batch_idx}", flush=True)
                    print(f"Image1 name: {image1_name}", flush=True)  
                    print(f"Image2 name: {image2_name}", flush=True)
                # Delete flip tokens immediately after use
                del flip_tokens_a, flip_tokens_b
                torch.cuda.empty_cache()

                # Use smaller interpolation size
                new_size = min(16, diffSWText.shape[-1])  # Cap at 16x16 for memory
                new_size = (new_size, new_size)
                
                diffSWimage = interpolate_SW_matrix(diffSWimage, new_size)
                cosine_sim = interpolate_SW_matrix(DiffSW.cosine_similarity, new_size)
                diffSWText = interpolate_SW_matrix(diffSWText, new_size)
                textSimilar = interpolate_SW_matrix(textSimilar, new_size)

                # Delete DiffSW object and clear cache
                del DiffSW
                torch.cuda.empty_cache()

                # Path extraction
                textSWpath, text_startPoints = diff_SW_Path(diffSWText, textSimilar,
                                                          match_score=2, miss_score=-3, gap_penalty=-1)
                diffSWText_final = diffSWText * textSWpath
                del diffSWText, textSWpath, textSimilar
                torch.cuda.empty_cache()

                imageSWpath, _ = diff_SW_Path(diffSWimage, cosine_sim,
                                            match_score=2, miss_score=-3, gap_penalty=-1, 
                                            position=text_startPoints)
                diffSWimage_final = diffSWimage * imageSWpath
                del diffSWimage, imageSWpath, cosine_sim, text_startPoints
                torch.cuda.empty_cache()

                # Loss computation
                path_loss = criterion(diffSWText_final, diffSWimage_final)
                scaled_loss = path_loss / gradient_accumulation_steps
                
                del diffSWText_final, diffSWimage_final
                torch.cuda.empty_cache()

                epoch_loss += path_loss.item()
                accumulated_loss += path_loss.item()

                # Backpropagation
                scaled_loss.backward()
                del path_loss, scaled_loss
                torch.cuda.empty_cache()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(trainLoader) - 1:
                    optimizer.step()

                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {accumulated_loss:.4f}", flush=True)
                accumulated_loss = 0

                # Final cleanup - don't delete input images until the very end
                # The gradients should be computed by now
                del image1, image2
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ OOM at epoch {epoch}, batch {batch_idx}: {e}", flush=True)
                    if torch.cuda.is_available():
                        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
                    torch.cuda.empty_cache()
                    return loss_lst
                else:
                    raise e

        epoch_loss = epoch_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}', flush=True)
        loss_lst.append(epoch_loss)

    return loss_lst



if __name__ == '__main__':
    loss_type = 'MSE' # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 'Dice', 'Wasserstein']
    model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'Transformer']
    window_size = 32
    vector_size = 64
    normalize_type = '' # ['min_max', 'mean_std']
    epochs = 100
    learning_rate = 1e-4
    gradient_accumulation_steps = 1  # Accumulate gradients over 4 batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = True # Set to True to save patches and heatmaps for debugging
    debug_wandb = False # Set to True to log training to Weights & Biases
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
                "normalizing method ": normalize_type,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": True
            })

    cnn_transformer_model = EmbeddingModel(
        window_size=window_size,
        stride=window_size//2,
        vector_size=vector_size,
        model_arch=model_arch,
        use_checkpointing=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )  # In Train.py


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
        criterion,
        loss_type,
        device,
        normalize_type,
        epochs,
        learning_rate,
        debug,
        gradient_accumulation_steps,
        debug_wandb,
        show_gradients
    )
    if debug_wandb:
        wandb.finish()