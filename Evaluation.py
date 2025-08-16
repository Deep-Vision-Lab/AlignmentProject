import warnings
import torch.nn as nn
import torch.nn.functional as F
from SmithWaterman import SmithWaterman
from saveDATA import *
from Visualization import *
from embeddingModel import EmbeddingModel
from AlignmentAlgo import Alignment # Assuming sliding_window is not here
from newDataLoader import test_dataloader
from pathExtractor import *
from LossFunctionWithHelpers import *

warnings.filterwarnings("ignore")

def Evaluate(model, alignment_model, dataloader, criterion, window_size,loss_type, device, normalize_type):
    from embeddingModel import sliding_window # Import here to access model's window_size/stride easily
    torch.cuda.empty_cache()
    """Evaluates the model on the test set."""
    print(f"Using device: {device}")


    model.eval()
    alignment_model.eval()

    test_loss = 0
    total_correct = 0
    total_elements = 0

    print("Test DataLoader length:", len(dataloader))

    vectors_similarity_file_path = f'Results/{loss_type}/Vectors_similarity/vec_similar_test.txt'
    open(vectors_similarity_file_path, 'w').close()

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (image_a, image_b, smith_matrix, seq1, seq2) in enumerate(dataloader):
            image_a, image_b, smith_matrix = image_a.to(device), image_b.to(device), smith_matrix.to(device)
            tokens_a, tokens_b = model(image_a, image_b)
            tokens_a, tokens_b = tokens_a.to(device), tokens_b.to(device)
            tokens_a = torch.flip(tokens_a, dims=[1])
            tokens_b = torch.flip(tokens_b, dims=[1])

            alignment_output = alignment_model(tokens_a, tokens_b)

            ###################################################################################
            # Interpolation
            
            # Ensure smith_matrix is [N, C, H, W]
            if smith_matrix.dim() == 2:
                # [H, W] -> [1, 1, H, W]
                smith_matrix = smith_matrix.unsqueeze(0).unsqueeze(0)
            elif smith_matrix.dim() == 3:
                # [C, H, W] -> [1, C, H, W]
                smith_matrix = smith_matrix.unsqueeze(0)
            # Resize to match alignment_output's H and W
            new_size = alignment_output.shape[-2:]  # get (H, W)
            smith_matrix = F.interpolate(smith_matrix, size=new_size, mode='nearest')
            # Optionally remove channel dimension
            smith_matrix = smith_matrix.squeeze(0).squeeze(0)
            assert smith_matrix.shape == alignment_output.shape, f"Shapes do not match! \
                                                        \nalignment_cuda shape: {alignment_output.shape} \
                                                        \nsmith_cuda shape: {smith_matrix.shape}"

            ###################################################################################
               
            # Compute loss
            path_loss = criterion(alignment_output, smith_matrix)

            routes = makeTracerouteMatrix(alignment_output)
            for i, _ in enumerate(alignment_output):            
                # Generate patches for visualization
                # Ensure model has window_size and stride attributes, or pass them explicitly
                # Assuming model is an instance of EmbeddingModel and has these attributes
                current_img_a_for_patches = image_a[i].unsqueeze(0).cpu() # sliding_window expects batch
                current_img_b_for_patches = image_b[i].unsqueeze(0).cpu()
                
                # Access window_size and stride from the model instance if they are attributes
                # If not, you might need to pass them to Evaluate or get them from config
                model_window_size = model.window_size
                model_stride = model.stride

                patches_a_for_viz = sliding_window(current_img_a_for_patches, model_window_size, model_stride).squeeze(0)
                patches_b_for_viz = sliding_window(current_img_b_for_patches, model_window_size, model_stride).squeeze(0)

                # The heatmap axes correspond to the flipped tokens, so flip patches accordingly
                patches_y_heatmap = torch.flip(patches_a_for_viz, dims=[0]) # From image_a, for Y-axis
                patches_x_heatmap = torch.flip(patches_b_for_viz, dims=[0]) # From image_b, for X-axis

                visualize_heatmaps(alignment_output[i].cpu(), smith_matrix[i].cpu(), # Pass tensors before path extraction
                                f"Results/{loss_type}/Matrices_plots/heatmaps_epoch_batch_{batch_idx}_{i}.png",
                                    patches_y=patches_y_heatmap, patches_x=patches_x_heatmap)
                visualize_heatmaps(alignment_output[i].cpu(), torch.tensor(routes[i]).cpu(), # Pass tensors before path extraction
                                f"Results/{loss_type}/Matrices_plots/heatmaps_routes_epoch_batch_{batch_idx}_{i}.png",
                                    patches_y=patches_y_heatmap, patches_x=patches_x_heatmap)
                buildAlignedImages(image_a[i], image_b[i], routes[i], window_size,
                                f'Results/{loss_type}/Lines_plots/lines_{batch_idx}_{i}')
            
            del alignment_output
            del image_a, image_b, smith_matrix, tokens_a, tokens_b  # Delete the tensors
            torch.cuda.empty_cache()

            # Print stats every 10 batches
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {path_loss.item()}')
            test_loss += path_loss.item()
            del path_loss

        # Compute epoch accuracy
        test_loss = test_loss / len(dataloader)
        print(f'Loss: {test_loss:.4f}')


from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    normalize_type = 'mean_std' # ['min_max', 'mean_std']
    loss_type = 'MSE'  # ['HeightDiff', 'MSE']
    model_arch = 'CNN' # ['CNN-Transformer','CNN','Transformer']
    window_size = 64
    vector_size = 128

    criterion = None
    if loss_type == 'HeightDiff':
        criterion = HeightDiff_loss
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'GuidedAttention':
        criterion = guided_attention_loss
    elif loss_type == 'CrossEntropy':
        criterion = kl_divergence_loss
    elif loss_type == 'Dice':
        criterion = dice_loss
    elif loss_type == 'Wasserstein':
        criterion = wasserstein_distance
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_transformer_model = EmbeddingModel(window_size=window_size, stride=window_size, 
                                           vector_size=vector_size,model_arch=model_arch).to(device)
    cnn_transformer_model.load_state_dict(torch.load(f"Weights/{loss_type}/model_epoch_100.pth",
                                                      map_location=device))
    sw = SmithWaterman(match_score=3, mismatch_penalty=-1, gap_penalty=-2).to(device)

    Evaluate(cnn_transformer_model, sw, test_dataloader, 
             criterion, window_size, loss_type, device, normalize_type)