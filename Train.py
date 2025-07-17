import warnings
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from embeddingModel import EmbeddingModel
from AlignmentAlgo import Alignment
from newDataLoader import train_dataloader, valid_dataloader, batch_size
from pathExtractor import *
from saveDATA import *
from LossFunctionWithHelpers import *
from Evaluation import Evaluate
from Visualization import visualize_heatmaps, visualize_single_heatmap_with_text_labels, visualize_dual_char_heatmaps
from embeddingModel import sliding_window # For generating patches for visualization
import os # For creating directories
import sys

import wandb

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


def Train(model, alignment_model, trainLoader, criterion, loss_type, device, normalize_type, epochs=100, learning_rate=1e-4):
    model.to(device)
    model.train()
    loss_lst = []

    print(f"Using device: {device}")
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    
    print("Train DataLoader length:", len(trainLoader))
    for epoch in range(epochs):
        # Set default values for epoch directories
        vectors_epoch_dir = None
        matrices_epoch_dir = None
        # Create directories for this epoch's outputs
        # if epoch % 10 == 9:
        vectors_epoch_dir = f'TrainResults/{loss_type}/VectorsPerEpoch/{model.model_arch}/epoch_{epoch+1}'
        matrices_epoch_dir = f'TrainResults/{loss_type}/ScoreMatricesPerEpoch/{model.model_arch}/epoch_{epoch+1}'
        os.makedirs(vectors_epoch_dir, exist_ok=True)
        os.makedirs(matrices_epoch_dir, exist_ok=True)

        epoch_loss = 0
        total_correct = 0
        total_elements = 0
        for batch_idx, (image_a, image_b, smith_matrix_original_batch, seq1_tokenized, seq2_tokenized, original_text1_batch, original_text2_batch) in enumerate(trainLoader):            
            optimizer.zero_grad()

            image_a = image_a.to(device)
            image_a.retain_grad()
            image_b = image_b.to(device)
            image_b.retain_grad()
            
            # smith_matrix_original_batch is already on device if dataloader sends it there, or needs .to(device)
            current_smith_matrix = smith_matrix_original_batch.to(device) 
            current_smith_matrix.retain_grad()
            
            tokens_a, tokens_b = model(image_a, image_b)
            tokens_a, tokens_b = tokens_a.to(device), tokens_b.to(device)
            
            tokens_a = torch.flip(tokens_a, dims=[1])
            tokens_b = torch.flip(tokens_b, dims=[1])
            
            tokens_a.retain_grad()
            tokens_b.retain_grad()
            
            alignment_output = alignment_model(tokens_a, tokens_b).to(device)

            batch_correct = 0
            batch_total = 0


            ###################################################################################
            # 1. Interpolate smith_matrix_loaded to match alignment_output_raw's dimensions
            # Use current_smith_matrix for interpolation
            if current_smith_matrix.dim() == 2: # Should not happen with batching
                # [H, W] -> [1, 1, H, W]
                interpolated_smith_matrix = current_smith_matrix.unsqueeze(0).unsqueeze(0)
            elif current_smith_matrix.dim() == 3: # Batch of matrices [B, H, W]
                # [C, H, W] -> [1, C, H, W]
                interpolated_smith_matrix = current_smith_matrix.unsqueeze(1) # [B, 1, H, W]
            
            new_size = alignment_output.shape[-2:]
            # The 'linear' mode for interpolate is for 3D tensors. For 4D tensors (B, C, H, W),
            # 'bilinear' should be used. align_corners=False is recommended to avoid warnings.
            interpolated_smith_matrix = F.interpolate(interpolated_smith_matrix, size=new_size, mode='bilinear')
            interpolated_smith_matrix = interpolated_smith_matrix.squeeze(1) # [B, H_new, W_new]
            
            assert interpolated_smith_matrix.shape == alignment_output.shape, \
                f"Shapes after interpolation do not match! alignment_output: {alignment_output.shape}, interpolated_smith_matrix: {interpolated_smith_matrix.shape}"
            ###################################################################################

            # 3a. Normalize alignment_output
            if normalize_type == 'min_max':
                alignment_min = alignment_output.min()
                alignment_max = alignment_output.max()
                alignment_output = 2 * (alignment_output - alignment_min) / (alignment_max - alignment_min + 1e-8) - 1
                
                # Scale normalized tensor to match smith_matrix's max value
                # At this point, interpolated_smith_matrix is the one to compare against for scaling
                smith_max = interpolated_smith_matrix.max() # Max of the interpolated version
                alignment_output = alignment_output * smith_max
                del alignment_min, alignment_max, smith_max
            
            elif normalize_type == 'mean_std':
                alignment_mean = alignment_output.mean(dim=(1, 2), keepdim=True)
                alignment_std = alignment_output.std(dim=(1, 2), keepdim=True)
                alignment_output = (alignment_output - alignment_mean) / (alignment_std + 1e-8)
                
                del alignment_mean, alignment_std

            ###################################################################################
            # Extract traceback paths
            
            # Path is extracted from the scaled alignment_output
            alignment_np = alignment_output.detach().cpu().numpy()
            alignment_path = torch.tensor(makeTracerouteMatrixBinary(alignment_np), dtype=torch.float32, device=device)
            
            # Path is extracted from the interpolated smith_matrix
            smith_np = interpolated_smith_matrix.detach().cpu().numpy()
            smith_path = torch.tensor(makeTracerouteMatrixBinary(smith_np), dtype=torch.float32, device=device)
            
            ###################################################################################
            # Smooth paths and apply path masks
            
            # interpolated_smith_matrix is processed into its final target form (smoothed path)
            processed_smith_matrix_for_loss = smooth_path(smith_path)
            del smith_path
            
            # alignment_output (which was already scaled) is masked by its own extracted path
            alignment_output = alignment_output * alignment_path
            del alignment_path
            

            if batch_idx == len(trainLoader) - 1:
                # Save token vectors and score matrices for every batch
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Saving data...")

                # Clone the relevant tensors for visualization.
                # alignment_output is path-masked.
                # smith_matrix is the smoothed path version.
                # Use processed_smith_matrix_for_loss for the existing heatmap.
                cloned_alignment_output_for_viz = alignment_output.clone().detach()
                cloned_processed_smith_for_viz = processed_smith_matrix_for_loss.clone().detach()
                smith_matrix_for_char_level_viz = current_smith_matrix.clone().detach()
                saveHeatmapPlots(model, image_a, image_b, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx, cloned_alignment_output_for_viz, 
                                cloned_processed_smith_for_viz, smith_matrix_for_char_level_viz, matrices_epoch_dir, original_text1_batch, 
                                original_text2_batch)
                
                # Path Visualization testing
                # cloned_alignment_output_for_viz = alignment_path.clone().detach()
                # cloned_processed_smith_for_viz = smith_path.clone().detach()
                # saveHeatmapPlots(model, image_a, image_b, tokens_a, tokens_b, vectors_epoch_dir, epoch, batch_idx, cloned_alignment_output_for_viz, 
                #                 cloned_processed_smith_for_viz, smith_matrix_for_char_level_viz, matrices_epoch_dir, original_text1_batch, 
                #                 original_text2_batch)
            
            ###################################################################################

            # Compute loss
            path_loss = criterion(alignment_output, processed_smith_matrix_for_loss)

            # Accuracy Calculation
            correct = (torch.abs(alignment_output - processed_smith_matrix_for_loss) < 0.1).float()
            batch_correct += correct.sum().item()
            # Make sure smith_matrix for numel is the one used in loss
            batch_total += processed_smith_matrix_for_loss.numel() 
            
            del alignment_output, processed_smith_matrix_for_loss, interpolated_smith_matrix, current_smith_matrix

            epoch_loss += path_loss.item()
             
            # Backpropagation
            path_loss.backward()
            optimizer.step()

            # print(f'image A gradient: {image_a.grad.sum()}')
            # print(f'image B gradient: {image_b.grad.sum()}')


            batch_accuracy = (batch_correct / batch_total) if batch_total > 0 else 0
            total_correct += batch_correct
            total_elements += batch_total

            # Print stats every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {path_loss.item()}, Accuracy: {batch_accuracy * 100:.2f}%')
            
            del path_loss
        
            # gradient_a = image_a.grad.sum()
            # gradient_b = image_b.grad.sum()
            # print(f'image_a gradient: {gradient_a}')
            # print(f'image_b gradient: {gradient_b}')

            # wandb.log({"Image A Gradient": torch.abs(gradient_a),
            #             "Image B Gradient": torch.abs(gradient_b)})

            # Delete tensors from the last batch of the epoch
            del image_a, image_b, tokens_a, tokens_b 
            torch.cuda.empty_cache()  # Release unused memory from GPU cache
        
        
        # Compute epoch accuracy
        epoch_accuracy = (total_correct / total_elements) * 100 if total_elements > 0 else 0
        epoch_loss = epoch_loss/len(trainLoader)

        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        loss_lst.append(epoch_loss)
        wandb.log({"Loss": epoch_loss, "Accuracy": epoch_accuracy}, step=epoch, commit=True)

        if epoch % 10 == 9:
            torch.save(cnn_transformer_model.state_dict(), f'Weights/{loss_type}/{model.model_arch}/model_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch + 1}.") # Changed from "New best model" as there's no best model tracking here

    print('Training complete!')
    return loss_lst


if __name__ == '__main__':
    loss_type = 'HeightDiff' # ['HeightDiff', 'MSE']
    model_arch = 'Transformer' # ['CNN-Transformer','CNN','Transformer']
    window_size = 32
    vector_size = 64
    normalize_type = 'mean_std' # ['min_max', 'mean_std']
    epochs = 300
    learning_rate = 1e-3

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_transformer_model = EmbeddingModel(window_size=window_size, stride=window_size//2, 
                                           vector_size=vector_size,model_arch=model_arch).to(device)# In Train.py
                                           
    alignment_model = Alignment(match_score=2, miss_score=-3).to(device)
   
    if loss_type == 'HeightDiff':
        criterion = compute_path_loss
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()


    loss_lst = Train(cnn_transformer_model, alignment_model, train_dataloader,
                      criterion, loss_type, device, normalize_type,epochs,learning_rate)
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
    Evaluate(cnn_transformer_model, alignment_model, valid_dataloader, criterion, window_size, loss_type, device, normalize_type)