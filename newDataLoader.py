import os.path

import torch
import torch.nn.functional as F # Added for interpolate and conv2d
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from newDataSet import TextLineModern, window_size

from DiffSWAlgo import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4

data_dir = "DataSet/Synthetic"  # Directory for the new dataset
# Define paths for NewDataSet
new_dataset = {
    "images": os.path.join(data_dir, "images"),
    "score_matrices": os.path.join(data_dir, "score_matrices"),
    "similarity_matrices": os.path.join(data_dir, "similarity_matrices"),
    "texts":  os.path.join(data_dir, "texts")
}

class ToTensorWithGrad:
    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        return tensor.requires_grad_()

class ScoreMapping:
    def __call__(self, tensor):
        # Map 1 to 7 and 0 to -3
        return torch.where(tensor == 1, torch.tensor(7.0), torch.where(tensor == 0, torch.tensor(-3.0), tensor))

# Define transformations for images
transform = transforms.Compose([
    ToTensorWithGrad(),
    transforms.Resize((128, 1024)),
    # ScoreMapping()
])

# Create the full dataset
full_dataset = TextLineModern(
    datasetPaths=None,  # Not used for NewDataSet
    fonts=None,         # Not used for NewDataSet
    patchHeight=None,   # Not used for NewDataSet
    patchWidth=None,    # Not used for NewDataSet
    numberWords=None,   # Not used for NewDataSet
    new_dataset=new_dataset,
    transform=transform
)

# Split the dataset into 80% training and 20% testing
train_size = int(0.6 * len(full_dataset))
valid_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

# test_size = 2
# valid_size = len(full_dataset) - train_size - test_size

train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])

# Function to pad matrices to the maximum size in the batch
def pad_matrices(matrices, smooth=False, kernel_size=5, sigma=1.0):
    """
    Resizes matrices in a batch to a common square dimension using interpolation.
    Optionally applies Gaussian smoothing after interpolation.
    The target dimension is the maximum dimension found across all matrices in the batch.
    """
    if not matrices:
        return torch.empty(0) # Or handle as an error, though DataLoader usually provides non-empty batches.

    max_dim = max(max(mat.shape) for mat in matrices)

    gaussian_kernel_2d = None
    if smooth:
        # This check is technically redundant due to the initial 'if not matrices:'
        # but ensures device can be accessed if matrices is guaranteed non-empty here.
        if matrices:
            device = matrices[0].device # Assume all matrices in the list are on the same device
            
            # Create 1D Gaussian kernel centered at 0
            _x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=torch.float32)
            _gauss1d = torch.exp(-_x.pow(2) / (2 * sigma**2))
            _gauss1d /= _gauss1d.sum() # Normalize 1D kernel

            # Create 2D kernel from outer product of 1D kernel
            # This results in a kernel that sums to 1
            gaussian_kernel_2d = torch.outer(_gauss1d, _gauss1d)
            # Reshape for conv2d: [out_channels, in_channels, H, W]
            gaussian_kernel_2d = gaussian_kernel_2d.unsqueeze(0).unsqueeze(0)

    processed_matrices = []
    for mat in matrices:
        # Add batch and channel dimensions for interpolate: [H, W] -> [1, 1, H, W]
        mat_unsqueezed = mat.unsqueeze(0).unsqueeze(0)
        # Interpolate to [1, 1, max_dim, max_dim]
        processed_mat = F.interpolate(mat_unsqueezed, size=(max_dim, max_dim), mode='nearest')
        
        if smooth and gaussian_kernel_2d is not None:
            current_kernel = gaussian_kernel_2d.to(processed_mat.device) # Ensure kernel is on correct device
            padding = kernel_size // 2
            processed_mat = F.conv2d(processed_mat, current_kernel, padding=padding)

        # Remove batch and channel dimensions: [1, 1, max_dim, max_dim] -> [max_dim, max_dim]
        processed_matrices.append(processed_mat.squeeze(0).squeeze(0))
    return torch.stack(processed_matrices, dim=0)

# Define a custom collate function to handle variable-sized smith matrices
def custom_collate_fn(batch):
    """
    Custom collate function to handle the batching of the dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_a, images_b, score_matrix, similar_matrix = zip(*batch)
    
    # Stack image tensors
    images_a = torch.stack(images_a, dim=0).to(device)
    images_a.retain_grad()
    images_b = torch.stack(images_b, dim=0).to(device)
    images_b.retain_grad()
    
    # Pad and stack smith matrices
    # Smoothing can be enabled here if desired, e.g., pad_matrices(smith_matrices, smooth=True)
    score_matrix = pad_matrices(score_matrix, smooth=False) # Defaulting to no smoothing for now
    similar_matrix = pad_matrices(similar_matrix, smooth=False) # Defaulting to no smoothing for now
    # alignment_model = DiffSWAlgo(match_score=7, miss_score=-3, 
    #                                             gap=-1).to(device)
    # SW_matrices = alignment_model(similarity_matrix=similar_matrix,
    #                                             calc_cosine=False).to(device)

    return (
        images_a,
        images_b,
        score_matrix,
        similar_matrix
    )


# Create DataLoaders for training and testing
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
)

valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
)



if __name__ == "__main__":
    # Test the DataLoader
    for batch in train_dataloader:
        images_a, images_b, SW_matrices, similar_matrix = batch
        print(f'Batch images_a shape: {images_a.shape}')
        print(f'Batch images_b shape: {images_b.shape}')
        print(f'Batch SW_matrices shape: {SW_matrices.shape}')
        print(f'Batch similar_matrix shape: {similar_matrix.shape}')
        break  # Just test one batch