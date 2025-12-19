import torch
import torch.nn.functional as F # Added for interpolate and conv2d
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from newDataSet import TextLineModern, window_size

from Parameters import *
from DiffNWAlgo import *


data_dir = "DataSet/Synthetic"  # Directory for the new dataset
# Define paths for NewDataSet
new_dataset = {
    "images": os.path.join(data_dir, "images"),
    "matrices": os.path.join(data_dir, "matrices"),
    "diffmatrices": os.path.join(data_dir, "diffmatrices"),
    "similarity_matrices": os.path.join(data_dir, "similarity_matrices"),
    "texts":  os.path.join(data_dir, "texts")
}

class ToTensorWithGrad:
    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        return tensor.requires_grad_()


# Define transformations for images
transform = transforms.Compose([
    ToTensorWithGrad(),
    transforms.Resize((128, 1024)),
    # ScoreMapping()
])

# Create the full dataset
full_dataset = TextLineModern(  # Not used for NewDataSet
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
        return torch.empty(0)

    # Ensure all matrices are on the same device
    device = matrices[0].device
    max_dim = max(max(mat.shape) for mat in matrices)

    gaussian_kernel_2d = None
    if smooth:
        # Create 1D Gaussian kernel centered at 0
        _x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=torch.float32)
        _gauss1d = torch.exp(-_x.pow(2) / (2 * sigma**2))
        _gauss1d /= _gauss1d.sum()

        # Create 2D kernel from outer product of 1D kernel
        gaussian_kernel_2d = torch.outer(_gauss1d, _gauss1d)
        # Reshape for conv2d: [out_channels, in_channels, H, W]
        gaussian_kernel_2d = gaussian_kernel_2d.unsqueeze(0).unsqueeze(0)

    processed_matrices = []
    for mat in matrices:
        # Ensure matrix is on the correct device
        mat = mat.to(device)
        
        # Add batch and channel dimensions for interpolate: [H, W] -> [1, 1, H, W]
        mat_unsqueezed = mat.unsqueeze(0).unsqueeze(0)
        # Interpolate to [1, 1, max_dim, max_dim]
        processed_mat = F.interpolate(mat_unsqueezed, size=(max_dim, max_dim), mode='nearest')
        
        if smooth and gaussian_kernel_2d is not None:
            current_kernel = gaussian_kernel_2d.to(processed_mat.device)
            padding = kernel_size // 2
            processed_mat = F.conv2d(processed_mat, current_kernel, padding=padding)

        # Remove batch and channel dimensions: [1, 1, max_dim, max_dim] -> [max_dim, max_dim]
        processed_matrices.append(processed_mat.squeeze(0).squeeze(0))
    
    # Stack all processed matrices and ensure they're on the correct device
    result = torch.stack(processed_matrices, dim=0).to(device)
    return result

# Define a custom collate function to handle variable-sized smith matrices
def custom_collate_fn(batch):
    """Custom collate function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_a, images_b, matrices, similar_matrix, images1_names, images2_names = zip(*batch)

    # Stack on CPU first, then move to device
    images_a = torch.stack(images_a, dim=0)
    images_b = torch.stack(images_b, dim=0)
    
    # Convert matrices to tensors on CPU
    matrices_cpu = []
    similar_matrix_cpu = []
    
    for matrix, sim_mat in zip(matrices, similar_matrix):
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        if not isinstance(sim_mat, torch.Tensor):
            sim_mat = torch.tensor(sim_mat, dtype=torch.float32)
            
        matrices_cpu.append(matrix)
        similar_matrix_cpu.append(sim_mat)
    
    # Pad matrices (still on CPU)
    matrices = pad_matrices(matrices_cpu, smooth=False)
    similar_matrix = pad_matrices(similar_matrix_cpu, smooth=False)
    
    # Now move everything to device at once
    images_a = images_a.to(device, non_blocking=True)
    images_b = images_b.to(device, non_blocking=True)
    matrices = matrices.to(device, non_blocking=True)
    similar_matrix = similar_matrix.to(device, non_blocking=True)
    
    # Set requires_grad only after moving to device
    images_a.requires_grad_(True)
    images_b.requires_grad_(True)
    matrices.requires_grad_(True)
    similar_matrix.requires_grad_(True)

    return images_a, images_b, matrices, similar_matrix, images1_names, images2_names


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
        images_a, images_b, NW_matrices, similar_matrix = batch
        print(f'Batch images_a shape: {images_a.shape}')
        print(f'Batch images_b shape: {images_b.shape}')
        print(f'Batch NW_matrices shape: {NW_matrices.shape}')
        print(f'Batch similar_matrix shape: {similar_matrix.shape}')
        break  # Just test one batch