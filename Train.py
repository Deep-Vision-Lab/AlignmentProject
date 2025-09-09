import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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

def Train(model, DiffSW, trainLoader, criterion, 
        loss_type, device, normalize_type, epochs=100,
        learning_rate=1e-4, debug=False, 
        gradient_accumulation_steps=1, debug_wandb=True,
        show_gradients=False):

    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()  # Add GradScaler for mixed precision
    loss_lst = []
    print(f"Using device: {device}")
    print("Train DataLoader length:", len(trainLoader))
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print("Using automatic mixed precision training")

    for epoch in range(epochs):
        # Prepare output directories
        weights_dir = f'Weights/{loss_type}/{model.model_arch}'
        os.makedirs(weights_dir, exist_ok=True)

        epoch_loss = 0
        accumulated_loss = 0

        for batch_idx, (image1, image2, diffSWText, textSimilar) in enumerate(trainLoader):
            # Only zero gradients at the start of accumulation cycle
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Forward pass with autocast for mixed precision
            with torch.cuda.amp.autocast():
                tokens_a, tokens_b = model(image1, image2, show_dims=False)
                flip_tokens_a = torch.flip(tokens_a, dims=[1])
                flip_tokens_b = torch.flip(tokens_b, dims=[1])
                diffSWimage = DiffSW(x1=flip_tokens_a, x2=flip_tokens_b)


            ######################################################################################################################################
            
            # Interpolate smith matrix to match alignment output shape
            new_size = diffSWText.shape[-2:]
            diffSWimage = interpolate_SW_matrix(diffSWimage,
                                                   new_size)
            DiffSW.cosine_similarity = interpolate_SW_matrix(DiffSW.cosine_similarity,
                                                                    new_size)
            assert diffSWText.shape == diffSWimage.shape == DiffSW.cosine_similarity.shape, \
                f"Shapes after interpolation do not match! diffSWimage: {diffSWimage.shape}, \
                diffSWText: {diffSWText.shape}, cosine_similarity: {DiffSW.cosine_similarity.shape}"

            ######################################################################################################################################
            # Extracting the path
            
            textSWpath, text_startPoints = SW_Path(diffSWText,
                                                   textSimilar,match_score=7,
                                                   miss_score=-3, gap_penalty=-1)
            diffSWText = diffSWText * textSWpath

            imageSWpath, _ = diff_SW_Path(diffSWimage,
                                                   DiffSW.cosine_similarity,match_score=7,
                                                   miss_score=-3, gap_penalty=-1, position=text_startPoints)
            diffSWimage = diffSWimage * imageSWpath
            ######################################################################################################################################

            # Optionally smooth and normalize alignment output
            # alignment_output = smooth_and_normalize_matrix(alignment_output, normalize_type)
            # interpolated_smith_matrix = smooth_and_normalize_matrix(interpolated_smith_matrix, normalize_type)

            ####################################################################################################################################
            
            # Compute loss and scale by accumulation steps with autocast
            with torch.cuda.amp.autocast():
                path_loss = criterion(diffSWText, diffSWimage)
                scaled_loss = path_loss / gradient_accumulation_steps

            epoch_loss += path_loss.item()
            accumulated_loss += path_loss.item()

            # Backpropagation with gradient scaling
            scaler.scale(scaled_loss).backward()

            # Update weights only after accumulating enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(trainLoader) - 1:
                scaler.step(optimizer)
                scaler.update()

            if show_gradients:
                # print gradients for debugging
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(trainLoader)}, Accumulated Loss: {accumulated_loss:.4f}, image1.grad: {image1.grad.sum()}, image2.grad: {image2.grad.sum()}")
            else:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(trainLoader)}, Accumulated Loss: {accumulated_loss:.4f}")
            accumulated_loss = 0  # Reset accumulated loss for next cycle

            

            ######################################################################################################################################
            # Free memory

            if debug and batch_idx == len(trainLoader) - 1 : # Save visualizations every 10 batches if in debug mode
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
            
            del image1, image2 
            del tokens_a, tokens_b
            del flip_tokens_a, flip_tokens_b
            del diffSWText, diffSWimage
            del textSWpath, textSimilar
            del imageSWpath, DiffSW.cosine_similarity, DiffSW.align
            del new_size, text_startPoints
            del path_loss, scaled_loss
            torch.cuda.empty_cache()

            ######################################################################################################################################

        # Epoch summary
        epoch_loss = epoch_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}')
        loss_lst.append(epoch_loss)
        if debug_wandb:
            wandb.log({"Loss": epoch_loss}, step=epoch, commit=True)

        # Save model every 10 epochs
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f'Weights/{loss_type}/{model.model_arch}/model_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch + 1}.")

    print('Training complete!')
    return loss_lst



if __name__ == '__main__':
    loss_type = 'MSE' # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 'Dice', 'Wasserstein']
    model_arch = 'CNN' # ['CNN-Transformer', 'CNN', 'Transformer']
    window_size = 32
    vector_size = 64
    normalize_type = '' # ['min_max', 'mean_std']
    epochs = 100
    learning_rate = 1e-4
    gradient_accumulation_steps = 2  # Accumulate gradients over 4 batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False # Set to True to save patches and heatmaps for debugging
    debug_wandb = False # Set to True to log training to Weights & Biases
    show_gradients = False # Set to True to print gradients for debugging
    
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
        use_checkpointing=True
    ).to(device)  # In Train.py

    DiffSW = DiffSWAlgo(match_score=7, miss_score=-3, gap=-1).to(device)

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
        DiffSW,
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