import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_text_similarity_matrix(text1: str, text2: str, device: str = "cuda") -> torch.Tensor:
    """Fast equality matrix between two strings."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    bytes1 = text1.encode("utf-32-le")
    bytes2 = text2.encode("utf-32-le")
    chars1 = torch.frombuffer(bytearray(bytes1), dtype=torch.int32).to(device)
    chars2 = torch.frombuffer(bytearray(bytes2), dtype=torch.int32).to(device)
    return (chars1.unsqueeze(1) == chars2.unsqueeze(0)).float()


def textual_sliding_window(text, window_size, step_size):
    return [text[i:i + window_size] for i in range(0, len(text) - window_size + 1, step_size)]


class TextLineModern(Dataset):
    def __init__(self, new_dataset=None, transform=None, num_samples_override=None):
        self.new_dataset = new_dataset
        self.transform = transform

        if new_dataset:
            if num_samples_override is not None:
                self.num_samples = num_samples_override
            else:
                images_dir = new_dataset["images"]
                detected = len([
                    f for f in os.listdir(images_dir)
                    if f.startswith("img1_") and f.endswith(".png")
                ])
                self.num_samples = detected if detected > 0 else num_samples

    def __len__(self):
        return self.num_samples if self.new_dataset else 0

    def _read_image(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _read_text(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return " " + f.read().strip() + " "

    def __getitem__(self, idx):
        if not self.new_dataset:
            raise NotImplementedError("Handling for non-DataSet is not included.")

        sample_idx = idx + 1  # 1-based file naming
        images_dir = self.new_dataset["images"]
        texts_dir = self.new_dataset["texts"]

        img1_path = os.path.join(images_dir, f"img1_{sample_idx}.png")
        text1_path = os.path.join(texts_dir, f"text1_{sample_idx}.txt")
        img1 = self._read_image(img1_path)
        text1 = self._read_text(text1_path)

        if not use_image_pair_contrastive:
            return text1, img1

        img2_path = os.path.join(images_dir, f"img2_{sample_idx}.png")
        text2_path = os.path.join(texts_dir, f"text2_{sample_idx}.txt")
        if not os.path.exists(img2_path) or not os.path.exists(text2_path):
            # Keep old behavior if the dataset is single-line only.
            return text1, img1

        return {
            "text1": text1,
            "image1": img1,
            "text2": self._read_text(text2_path),
            "image2": self._read_image(img2_path),
        }
