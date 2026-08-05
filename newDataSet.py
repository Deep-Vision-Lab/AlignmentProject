import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
import random

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_STOCHASTIC_TRANSFORM_TOKENS = (
    "random",
    "jitter",
    "noise",
    "blur",
    "erasing",
    "affine",
    "perspective",
    "crop",
    "rotation",
)


def _deterministic_only(transform):
    """Keep deterministic preprocessing while dropping online augmentation.

    Synthetic images are already augmented on disk. The dataloader must only
    perform stable preprocessing such as resize, tensor conversion and
    normalization, exactly as it did for the previous generated dataset.
    """
    if transform is None:
        return None

    children = getattr(transform, "transforms", None)
    if children is not None:
        kept = []
        for child in children:
            filtered = _deterministic_only(child)
            if filtered is not None:
                kept.append(filtered)
        return tv_transforms.Compose(kept)

    name = transform.__class__.__name__.lower()
    if any(token in name for token in _STOCHASTIC_TRANSFORM_TOKENS):
        return None
    return transform


def compute_text_similarity_matrix(text1: str, text2: str, device: str = 'cuda') -> torch.Tensor:
    """Fast equality matrix between two strings."""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    bytes1 = text1.encode('utf-32-le')
    bytes2 = text2.encode('utf-32-le')
    chars1 = torch.frombuffer(bytearray(bytes1), dtype=torch.int32).to(device)
    chars2 = torch.frombuffer(bytearray(bytes2), dtype=torch.int32).to(device)
    return (chars1.unsqueeze(1) == chars2.unsqueeze(0)).float()


def textual_sliding_window(text, window_size, step_size):
    output = []
    for i in range(0, len(text) - window_size + 1, step_size):
        output.append(text[i:i + window_size])
    return output


class TextLineModern(Dataset):
    def __init__(self, new_dataset=None, transform=None, num_samples_override=None):
        self.new_dataset = new_dataset
        self.transform = _deterministic_only(transform)

        if new_dataset:
            if num_samples_override is not None:
                self.num_samples = num_samples_override
            else:
                images_dir = new_dataset['images']
                detected = len([
                    f for f in os.listdir(images_dir)
                    if f.startswith('img1_') and f.endswith('.png')
                ])
                self.num_samples = detected if detected > 0 else num_samples

    def __len__(self):
        return self.num_samples if self.new_dataset else 0

    def __getitem__(self, idx):
        if self.new_dataset:
            sample_idx = idx + 1

            img1_name = f"img1_{sample_idx}.png"
            text1_name = f"text1_{sample_idx}.txt"
            img2_name = f"img2_{sample_idx}.png"
            text2_name = f"text2_{sample_idx}.txt"

            img1_path = os.path.join(self.new_dataset['images'], img1_name)
            img1 = Image.open(img1_path).convert("RGB")
            if self.transform:
                img1 = self.transform(img1)

            text1_path = os.path.join(self.new_dataset['texts'], text1_name)
            with open(text1_path, 'r', encoding='utf-8') as file:
                text1 = ' ' + file.read().strip() + ' '

            img2_path = os.path.join(self.new_dataset['images'], img2_name)
            img2 = Image.open(img2_path).convert("RGB")
            if self.transform:
                img2 = self.transform(img2)

            text2_path = os.path.join(self.new_dataset['texts'], text2_name)
            with open(text2_path, 'r', encoding='utf-8') as file:
                text2 = ' ' + file.read().strip() + ' '

            return text1, img1, text2, img2
        raise NotImplementedError("Handling for non-NewDataSet is not included.")
