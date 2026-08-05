import os
from collections import OrderedDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

from Parameters import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_STOCHASTIC_TRANSFORM_TOKENS = (
    "random", "jitter", "noise", "blur", "erasing", "affine",
    "perspective", "crop", "rotation",
)


def _deterministic_only(transform):
    """Drop online augmentation while keeping stable preprocessing."""
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


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    """Synthetic paired-line dataset with precomputed paths and transcript cache."""

    def __init__(self, new_dataset=None, transform=None, num_samples_override=None):
        self.new_dataset = new_dataset
        self.transform = _deterministic_only(transform)
        self._image_cache = OrderedDict()
        self._image_cache_size = max(0, int(os.environ.get("IMAGE_DECODE_CACHE_SIZE", "0")))
        self._preload_transcripts = _env_flag("PRELOAD_TRANSCRIPTS", True)
        self._sample_records = []
        self._direct_subword_cache = {}

        if new_dataset:
            if num_samples_override is not None:
                self.num_samples = int(num_samples_override)
            else:
                images_dir = new_dataset["images"]
                detected = len([
                    f for f in os.listdir(images_dir)
                    if f.startswith("img1_") and f.endswith(".png")
                ])
                self.num_samples = detected if detected > 0 else int(num_samples)
            self._build_records()

    def _build_records(self):
        images_dir = self.new_dataset["images"]
        texts_dir = self.new_dataset["texts"]
        paired = bool(
            use_image_pair_contrastive
            or image_text_loss_on_both_lines
            or _env_flag("LOAD_PAIRED_LINES", False)
            or _env_flag("DIRECT_SUBWORD_SUPERVISION", False)
        )
        for zero_index in range(self.num_samples):
            sample_index = zero_index + 1
            image1 = os.path.join(images_dir, f"img1_{sample_index}.png")
            text1_path = os.path.join(texts_dir, f"text1_{sample_index}.txt")
            image2 = os.path.join(images_dir, f"img2_{sample_index}.png")
            text2_path = os.path.join(texts_dir, f"text2_{sample_index}.txt")
            has_pair = paired and os.path.exists(image2) and os.path.exists(text2_path)
            record = {
                "image1": image1,
                "text1_path": text1_path,
                "image2": image2 if has_pair else None,
                "text2_path": text2_path if has_pair else None,
            }
            if self._preload_transcripts:
                record["text1"] = self._read_text_file(text1_path)
                record["text2"] = self._read_text_file(text2_path) if has_pair else None
            self._sample_records.append(record)

    def __len__(self):
        return self.num_samples if self.new_dataset else 0

    @staticmethod
    def _read_text_file(path):
        with open(path, "r", encoding="utf-8") as file:
            return " " + file.read().strip() + " "

    def _read_image(self, path):
        cached = self._image_cache.get(path)
        if cached is not None:
            self._image_cache.move_to_end(path)
            image = cached.copy()
        else:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if self._image_cache_size > 0:
                self._image_cache[path] = image.copy()
                self._image_cache.move_to_end(path)
                while len(self._image_cache) > self._image_cache_size:
                    self._image_cache.popitem(last=False)
        if self.transform:
            image = self.transform(image)
        return image

    def _direct_subword_regions(self, image_path):
        from direct_subword_data import load_sidecar, sidecar_path

        cached = self._direct_subword_cache.get(image_path)
        if cached is not None:
            return cached
        path = sidecar_path(image_path)
        if not path.is_file():
            if _env_flag("DIRECT_SUBWORD_STRICT_BOXES", True):
                raise FileNotFoundError(
                    f"Missing {path}. Run "
                    "scripts/data/build_connected_subword_boxes_window_validated.py first."
                )
            regions = []
        else:
            regions = load_sidecar(path)
        self._direct_subword_cache[image_path] = regions
        return regions

    def __getitem__(self, idx):
        if not self.new_dataset:
            raise NotImplementedError("Handling for non-DataSet is not included.")
        record = self._sample_records[int(idx)]
        text1 = record.get("text1")
        if text1 is None:
            text1 = self._read_text_file(record["text1_path"])
        image1 = self._read_image(record["image1"])
        if record["image2"] is None:
            return text1, image1

        text2 = record.get("text2")
        if text2 is None:
            text2 = self._read_text_file(record["text2_path"])
        result = {
            "text1": text1,
            "image1": image1,
            "text2": text2,
            "image2": self._read_image(record["image2"]),
        }
        if _env_flag("DIRECT_SUBWORD_SUPERVISION", False):
            result["subwords1"] = self._direct_subword_regions(record["image1"])
            result["subwords2"] = self._direct_subword_regions(record["image2"])
            result["sample_index"] = int(idx)
        return result
