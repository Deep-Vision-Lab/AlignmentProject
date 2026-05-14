import torch
import random
import torch.nn.functional as F # Added for interpolate and conv2d
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from newDataSet import TextLineModern, window_size

from Parameters import *
from DiffNWAlgo import *


_default_data_dir = f'DataSet/Synthetic_{lang}_{num_samples}'

class ToTensorWithGrad:
    def __call__(self, img):
        tensor = transforms.ToTensor()(img)
        return tensor.requires_grad_()


transform = transforms.Compose([
    ToTensorWithGrad(),
    transforms.Resize((128, 1024)),
])


def build_dataloaders(data_dir=None):
    """Build train/valid/test DataLoaders for the given dataset directory.

    Args:
        data_dir: Path to the dataset root (must contain images/ and texts/ subdirs).
                  Defaults to DataSet/Synthetic_{lang}_{num_samples} from Parameters.py.

    Returns:
        (train_dataloader, valid_dataloader, test_dataloader)
    """
    if data_dir is None:
        data_dir = _default_data_dir

    dataset_paths = {
        "images": os.path.join(data_dir, "images"),
        "matrices": os.path.join(data_dir, "matrices"),
        "diffNWmatrices": os.path.join(data_dir, "diffNWmatrices"),
        "similarity_matrices": os.path.join(data_dir, "similarity_matrices"),
        "texts": os.path.join(data_dir, "texts"),
    }

    full_dataset = TextLineModern(new_dataset=dataset_paths, transform=transform)

    train_sz = int(0.6 * len(full_dataset))
    valid_sz = int(0.2 * len(full_dataset))
    test_sz = len(full_dataset) - train_sz - valid_sz

    train_ds, valid_ds, test_ds = random_split(full_dataset, [train_sz, valid_sz, test_sz])

    _train = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    _valid = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    _test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=True,  collate_fn=custom_collate_fn)
    return _train, _valid, _test


# Module-level defaults (used when newDataLoader is imported without a custom data_dir)
new_dataset = {
    "images": os.path.join(_default_data_dir, "images"),
    "matrices": os.path.join(_default_data_dir, "matrices"),
    "diffNWmatrices": os.path.join(_default_data_dir, "diffNWmatrices"),
    "similarity_matrices": os.path.join(_default_data_dir, "similarity_matrices"),
    "texts": os.path.join(_default_data_dir, "texts"),
}

full_dataset = TextLineModern(new_dataset=new_dataset, transform=transform)

train_size = int(0.6 * len(full_dataset))
valid_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])


# Function to pad matrices to the maximum size in the batch
def pad_matrices(matrices, smooth=False, kernel_size=5, sigma=1.0):
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
    """
    Custom collate function for contrastive learning with in-batch negative sampling.
    
    Each dataset sample is a positive pair (text, img).
    This collate function samples `num_negatives` in-batch negatives per sample.
    
    Returns:
        texts: tuple of B strings (positive texts)
        images: [B, C, H, W] tensor (positive images, on GPU)
        neg_texts: list of B lists, each containing num_negatives negative text strings
        neg_indices: [B, num_negatives] tensor of indices into the batch for negative images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack batch: each sample is (text, img)
    texts1, images1, texts2, images2 = zip(*batch)

    # Stack images on CPU first, then move to device
    images = torch.stack(images1, dim=0)
    images = images.to(device, non_blocking=True)
    images.requires_grad_(True)

    # Use actual batch size (last batch may be smaller than the global batch_size)
    actual_batch_size = len(texts1)

    pos_texts = texts1  # Positive texts are the original batch texts

    # ---- Hard Negative Generators (applied to the POSITIVE text) ----
    def _hard_neg_crop(text):
        """Return the first ~50% of the positive text (by words)."""
        words = text.split()
        if len(words) < 2:
            return text
        cutoff = max(1, len(words) // 2)
        return ' '.join(words[:cutoff])

    def _hard_neg_drop(text):
        """Delete one random word from the positive text."""
        words = text.split()
        if len(words) < 2:
            return text
        idx = random.randint(0, len(words) - 1)
        return ' '.join(words[:idx] + words[idx + 1:])

    def _hard_neg_shuffle(text):
        """Swap two random words in the positive text."""
        words = text.split()
        if len(words) < 2:
            return text
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        return ' '.join(words)

    _hard_neg_fns = [_hard_neg_crop, _hard_neg_drop, _hard_neg_shuffle]

    # ---- Random in-batch negative with optional length crop ----
    def _maybe_crop(text):
        """Randomly crop negative text to 50-100% of its length."""
        if random.random() < 0.3 and len(text) > 3:
            crop_ratio = random.uniform(0.5, 0.9)
            crop_len = max(2, int(len(text) * crop_ratio))
            start = random.randint(0, len(text) - crop_len)
            return text[start:start + crop_len]
        return text

    # ---- Build negatives: 2 hard + (num_negatives-2) random in-batch ----
    num_hard = min(2, num_negatives)
    num_random = num_negatives - num_hard

    neg_texts = []
    for i in range(actual_batch_size):
        sample_negs = []
        # Hard negatives (derived from positive text)
        for h in range(num_hard):
            fn = random.choice(_hard_neg_fns)
            sample_negs.append(fn(pos_texts[i]))
        # Random in-batch negatives
        available = [j for j in range(actual_batch_size) if j != i]
        if not available:
            available = [i]
        if len(available) >= num_random:
            sampled = random.sample(available, num_random)
        else:
            sampled = [random.choice(available) for _ in range(num_random)]
        for j in sampled:
            sample_negs.append(_maybe_crop(texts1[j]))
        neg_texts.append(sample_negs)

    return images, pos_texts, neg_texts


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
        texts, images, neg_texts, neg_indices = batch
        print(f'Batch texts length: {len(texts)}')
        print(f'Batch images shape: {images.shape}')
        print(f'Neg texts per sample: {len(neg_texts[0])} (num_negatives={num_negatives})')
        print(f'Neg indices shape: {neg_indices.shape}')
        print(f'Note: texts/images are aligned positive pairs')
        print(f'      neg_texts/neg_indices are in-batch negative samples')
        break  # Just test one batch