import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_text_similarity_matrix(text1: str, text2: str, 
                                             device: str = 'cuda') -> torch.Tensor:
    """
    Fastest GPU implementation using torch.compile (PyTorch 2.0+) if available.
    
    Uses a single-kernel approach for maximum throughput on large texts.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        device (str): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: Similarity matrix of shape [len(text1), len(text2)]
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Convert entire strings to tensor at once using torch.frombuffer for speed
    bytes1 = text1.encode('utf-32-le')  # 4 bytes per char
    bytes2 = text2.encode('utf-32-le')
    
    # Create tensors from buffer (zero-copy when possible)
    chars1 = torch.frombuffer(bytearray(bytes1), dtype=torch.int32).to(device)
    chars2 = torch.frombuffer(bytearray(bytes2), dtype=torch.int32).to(device)
    
    # Use einsum for potentially better kernel fusion
    # Broadcasting: [len1, 1] == [1, len2] -> [len1, len2]
    similarity_matrix = (chars1.unsqueeze(1) == chars2.unsqueeze(0)).float()
    
    return similarity_matrix


def textual_sliding_window(text, window_size, step_size):
    output = []
    for i in range(0, len(text) - window_size + 1, step_size):
        output.append(text[i:i + window_size])
    return output


class TextLineModern(Dataset):
    def __init__(self, new_dataset=None, transform=None):
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            # Reduce dataset size for memory testing
            self.image_pairs = [
                (f"img1_{i}.png", f"img2_{i}.png", f"similarityMatrix_{i}.npy", 
                 f"text1_{i}.txt", f"text2_{i}.txt") for i in
                range(1, 10001)]  # Reduced from 3001 to 101

    def __len__(self):
        return len(self.image_pairs)


    def __getitem__(self, idx):
        if self.new_dataset:
            img1_name, img2_name, similarity_matrix_name, text1, text2 = self.image_pairs[idx]

            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img2_path = os.path.join(self.new_dataset['images'], img2_name)

            SimilarityMatrix = os.path.join(self.new_dataset['similarity_matrices'], similarity_matrix_name)

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")


            # similar_matrix = np.load(SimilarityMatrix)
            # similar_matrix = torch.tensor(similar_matrix, dtype=torch.float32)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            text1_path = os.path.join(self.new_dataset['texts'], text1)
            text2_path = os.path.join(self.new_dataset['texts'], text2)
            with open(text1_path, 'r') as f:
                text_line1 = f.read().strip()
                text_line1 = ' ' + text_line1 + ' '
                
            with open(text2_path, 'r') as f:
                text_line2 = f.read().strip()
                text_line2 = ' ' + text_line2 + ' '

            similar_matrix = compute_text_similarity_matrix(text_line1, text_line2)
            similar_matrix = similar_matrix.to(device)
            similar_matrix.requires_grad_(True)
            
            return img1, img2, similar_matrix, text_line1, text_line2
        else:
            raise NotImplementedError("Handling for non-NewDataSet is not included.")