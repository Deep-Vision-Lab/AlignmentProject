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
from Visualization import *
from embeddingModel import *

import os
import wandb
import warnings


warnings.filterwarnings("ignore")

def saveHeatmapPlots(model, image_a, image_b, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx, 
                    cloned_alignment_output_for_viz, cloned_processed_smith_for_viz, smith_matrix_for_char_level_viz,
                    matrices_epoch_dir, original_text1_batch, original_text2_batch):
    for i in range(image_a.size(0)): # Iterate through items in the current batch
        # Save token vectors
        # tokens_a and tokens_b are already flipped and on the correct device
        print_elements(tokens_a[i], f'{vectors_epoch_dir}/tokens_a_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        print_elements(tokens_b[i], f'{vectors_epoch_dir}/tokens_b_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.xlsx')
        # Generate patches for visualization
        current_img_a_for_patches = image_a[i].unsqueeze(0).cpu()
        current_img_b_for_patches = image_b[i].unsqueeze(0).cpu()
        model_window_size = model.window_size
        model_stride = model.stride
        patches_a_for_viz = sliding_window(current_img_a_for_patches, model_window_size, model_stride).squeeze(0)
        patches_b_for_viz = sliding_window(current_img_b_for_patches, model_window_size, model_stride).squeeze(0)

        patches_y_heatmap = torch.flip(patches_a_for_viz, dims=[0]) # From image_a, for Y-axis
        patches_x_heatmap = torch.flip(patches_b_for_viz, dims=[0]) # From image_b, for X-axis

        # Visualize heatmaps
        visualize_heatmaps(
            cloned_alignment_output_for_viz[i].detach().cpu(),
            cloned_processed_smith_for_viz[i].detach().cpu(),
            f"{matrices_epoch_dir}/heatmaps_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png",
            patches_y=patches_y_heatmap.detach().cpu(),
            patches_x=patches_x_heatmap.detach().cpu()
        )

        # Only run the following if smith_matrix_for_char_level_viz is not None
        if smith_matrix_for_char_level_viz is not None:
            # New visualization: Raw Smith-Waterman matrix (before interpolation)
            # with original character sequences as axes. seq1 on X, seq2 on Y.
            current_smith_original_item = smith_matrix_for_char_level_viz[i] # Shape [H_orig, W_orig]
            current_original_text1 = original_text1_batch[i] # string for seq1
            current_original_text2 = original_text2_batch[i] # string for seq2

            # Extract the traceback path from the original Smith-Waterman matrix
            smith_original_scores_np = current_smith_original_item.cpu().numpy()

            # makeTracerouteMatrixBinary expects a batch, so add a dimension and remove it
            smith_original_path_np = makeTracerouteMatrixBinary(np.expand_dims(smith_original_scores_np, axis=0))[0]

            # SW matrix has text1 along rows, text2 along columns.
            # To put text1 (seq1) on X-axis and text2 (seq2) on Y-axis, we need to transpose.
            scores_matrix_to_plot = current_smith_original_item.T 
            path_matrix_to_plot = torch.tensor(smith_original_path_np).T

            # Labels for axes: X-axis for seq1 (original_text1), Y-axis for seq2 (original_text2)
            x_char_labels = ['ø'] + list(current_original_text1.replace(" ", "")) # 'ø' for the initial empty string state
            y_char_labels = ['ø'] + list(current_original_text2.replace(" ", ""))
            filename_char_level_smith_dual = f"{matrices_epoch_dir}/raw_char_smith_scores_path_epoch_{epoch+1}_batch_{batch_idx}_item_{i}.png"
            visualize_dual_char_heatmaps(
                scores_matrix_to_plot.cpu(),
                path_matrix_to_plot.cpu(),
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



import gc
import torch

def Train(model, DiffSW, trainLoader, criterion, loss_type, device, normalize_type, epochs=100, learning_rate=1e-4, debug=False):
    model.train()
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    loss_lst = []
    print(f"Using device: {device}")
    print("Train DataLoader length:", len(trainLoader))

    for epoch in range(epochs):
        # Prepare output directories
        weights_dir = f'Weights/{loss_type}/{model.model_arch}'
        os.makedirs(weights_dir, exist_ok=True)

        epoch_loss = 0

        for batch_idx, (image_a, image_b, diffSWText, textSimilar) in enumerate(trainLoader):
            optimizer.zero_grad()
            
            # Forward pass
            tokens_a, tokens_b = model(image_a, image_b, show_dims=False, debug=debug)
            flip_tokens_a = torch.flip(tokens_a, dims=[1])
            flip_tokens_b = torch.flip(tokens_b, dims=[1])
            diffSWimage = DiffSW(x1=flip_tokens_a, x2=flip_tokens_b)

            del flip_tokens_a, flip_tokens_b

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
            
            textSWpath, text_startPoints = diff_SW_Path(diffSWText,
                                                   textSimilar,match_score=7,
                                                   miss_score=-3, gap_penalty=-1)
            diffSWText = diffSWText * textSWpath
            del textSWpath, textSimilar

            imageSWpath, _ = diff_SW_Path(diffSWimage,
                                                   DiffSW.cosine_similarity,match_score=7,
                                                   miss_score=-3, gap_penalty=-1, position=text_startPoints)
            diffSWimage = diffSWimage * imageSWpath
            del imageSWpath, DiffSW.cosine_similarity
            ######################################################################################################################################

            # Optionally smooth and normalize alignment output
            # alignment_output = smooth_and_normalize_matrix(alignment_output, normalize_type)
            # interpolated_smith_matrix = smooth_and_normalize_matrix(interpolated_smith_matrix, normalize_type)

            ####################################################################################################################################

            if debug and batch_idx % 10 == 9: # Save visualizations every 10 batches if in debug mode
                # Prepare directories for saving visualizations
                vectors_epoch_dir = f'Visualizations/{loss_type}/{model.model_arch}/Vectors/Epoch_{epoch+1}'
                matrices_epoch_dir = f'Visualizations/{loss_type}/{model.model_arch}/Matrices/Epoch_{epoch+1}'
                os.makedirs(vectors_epoch_dir, exist_ok=True)
                os.makedirs(matrices_epoch_dir, exist_ok=True)

                # Clone tensors for visualization to avoid affecting gradients
                cloned_alignment_output_for_viz = diffSWimage.clone().detach()
                cloned_processed_smith_for_viz = diffSWText.clone().detach()

                # smith_matrix_for_char_level_viz is only available during training if calc_cosine=False in Alignment
                if hasattr(DiffSW, 'similarity_matrix'):
                    smith_matrix_for_char_level_viz = DiffSW.similarity_matrix.clone().detach()
                else:
                    smith_matrix_for_char_level_viz = None

                # original_text1_batch and original_text2_batch are only available during training if calc_cosine=False in Alignment
                if hasattr(DiffSW, 'original_text1_batch') and hasattr(DiffSW, 'original_text2_batch'):
                    original_text1_batch = DiffSW.original_text1_batch
                    original_text2_batch = DiffSW.original_text2_batch
                else:
                    original_text1_batch = None
                    original_text2_batch = None

                # Save heatmap visualizations
                saveHeatmapPlots(model, image_a, image_b, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx,
                                cloned_alignment_output_for_viz, cloned_processed_smith_for_viz, smith_matrix_for_char_level_viz,
                                matrices_epoch_dir, original_text1_batch, original_text2_batch)

                del cloned_alignment_output_for_viz, cloned_processed_smith_for_viz
                del smith_matrix_for_char_level_viz, vectors_epoch_dir, matrices_epoch_dir
            ####################################################################################################################################
            del image_a, image_b 
            del tokens_a, tokens_b
            
            # Compute loss and accuracy
            path_loss = criterion(diffSWText, diffSWimage)
            del diffSWText, diffSWimage

            epoch_loss += path_loss.item()

            # Backpropagation
            path_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(trainLoader)}, Loss: {path_loss.item():.4f}")
            del path_loss
            
            # print gradients for debugging
            # print(f"image_a.grad: {image_a.grad.sum()}, image_b.grad: {image_b.grad.sum()}, ")

            ######################################################################################################################################
            # Free memory

            del new_size, text_startPoints
            torch.cuda.empty_cache()

            ######################################################################################################################################

        # Epoch summary
        epoch_loss = epoch_loss / len(trainLoader)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}')
        loss_lst.append(epoch_loss)
        wandb.log({"Loss": epoch_loss}, step=epoch, commit=True)

        # Save model every 10 epochs
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f'Weights/{loss_type}/{model.model_arch}/model_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch + 1}.")

    print('Training complete!')
    return loss_lst



if __name__ == '__main__':
    loss_type = 'MSE' # ['HeightDiff', 'MSE', 'GuidedAttention', 'KL-Divergence', 'Dice', 'Wasserstein]
    model_arch = 'CNN' # ['CNN-Transformer','CNN','Transformer']
    window_size = 32
    vector_size = 64
    normalize_type = '' # ['min_max', 'mean_std']
    epochs = 100
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug = False # Set to True to save patches and heatmaps for debugging
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="AlignmentCNN-TransformerProject",
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
    cnn_transformer_model = EmbeddingModel(window_size=window_size, stride=window_size//2, 
                                           vector_size=vector_size, model_arch=model_arch).to(device)# In Train.py

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
    loss_lst = Train(cnn_transformer_model, DiffSW, train_dataloader,
                        criterion, loss_type, device, normalize_type,epochs,
                        learning_rate,debug)
    #     del cnn_transformer_model
    # except Exception as e: 
    #     del cnn_transformer_model
    #     for obj in gc.get_objects():
    #         try:
    #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #                 print(type(obj), obj.size(), obj.device)
    #         except:
    #             pass
    wandb.finish()
    
    # epochs = range(1, len(loss_lst) + 1)
    # plt.plot(epochs, loss_lst, marker='o', label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Epochs')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout() # Not needed if using wandb
    # plt.savefig(f'Results/{loss_type}/Training_loss.png')

    # print(f'**********************************Starting the Validation**********************************')
    # print('')
    
    # Example of how you might call Evaluate for validation after training
    # Evaluate(cnn_transformer_model, alignment_model, valid_dataloader, criterion, window_size, loss_type, device, normalize_type)