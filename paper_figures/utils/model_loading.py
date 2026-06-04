"""Utilities for loading trained models and dataset samples."""
import os
import sys
import glob
import torch
from PIL import Image
from torchvision import transforms

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ_ROOT)

from Parameters import (
    window_size, stride_ratio, vector_size, lang,
    use_bilstm, bilstm_layers, bilstm_hidden_dim,
    model_dropout, text_embedder_type,
)
from embeddingModel import EmbeddingModel
from textEmbedding import build_text_embedder

_IMG_H, _IMG_W = 128, 1024

_transform = transforms.Compose([
    transforms.Resize((_IMG_H, _IMG_W)),
    transforms.ToTensor(),
])


def get_stride():
    """Return the stride used during training."""
    return max(1, int(window_size * stride_ratio))


def _build_model(device, use_bilstm_override=None, stride_override=None):
    stride = stride_override if stride_override is not None else get_stride()
    _ubil = use_bilstm if use_bilstm_override is None else use_bilstm_override
    return EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=(lang.lower() == "arabic"),
        use_bilstm=_ubil,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        dropout=model_dropout,
    ).to(device)


def load_image_model(checkpoint_path, device, use_bilstm_override=None,
                     stride_override=None):
    """Load EmbeddingModel from a checkpoint file. Returns eval-mode model."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first:  python train.py --job_id <name>"
        )

    model = _build_model(device, use_bilstm_override, stride_override=stride_override)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [model_loading] Missing keys ({len(missing)}): {missing[:3]} ...")
    model.eval()
    model._flip_verified = True   # suppress debug_flip/ output during figure gen
    print(f"  Loaded checkpoint: {checkpoint_path}")
    return model


def load_random_model(device, use_bilstm_override=None, stride_override=None):
    """Build a randomly initialized EmbeddingModel (for before/after comparison)."""
    model = _build_model(device, use_bilstm_override, stride_override=stride_override)
    model.eval()
    model._flip_verified = True
    return model


def load_text_embedder(device):
    """Load and return the frozen text embedder (type from Parameters.py)."""
    try:
        embedder = build_text_embedder(kind=text_embedder_type, embedding_dim=vector_size)
    except Exception as e:
        print(f"  [model_loading] Warning: could not load '{text_embedder_type}' embedder: {e}")
        print("  [model_loading] Falling back to 'char' embedder.")
        embedder = build_text_embedder(kind="char", embedding_dim=vector_size)

    embedder = embedder.to(device)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad_(False)
    return embedder


def load_sample(data_dir, sample_idx, transform=True):
    """
    Load one (image, text) pair by 0-based index.

    File naming is 1-based: sample_idx 0 → img1_1.png / text1_1.txt.
    Falls back to sorted glob if the file is not found at the expected path.

    Args:
        data_dir:   Root dataset directory (contains images/ and texts/).
        sample_idx: 0-based sample index.
        transform:  If True, returns a float32 Tensor [C, H, W]; else PIL Image.

    Returns:
        (image, text_str)
    """
    file_idx = sample_idx + 1
    img_path = os.path.join(data_dir, "images", f"img1_{file_idx}.png")
    txt_path = os.path.join(data_dir, "texts",  f"text1_{file_idx}.txt")

    if not os.path.isfile(img_path):
        img_files = sorted(
            glob.glob(os.path.join(data_dir, "images", "img1_*.png")),
            key=lambda p: int(os.path.basename(p)[5:-4]),
        )
        if sample_idx >= len(img_files):
            raise IndexError(
                f"sample_idx={sample_idx} out of range "
                f"({len(img_files)} samples in {data_dir}/images/)"
            )
        img_path = img_files[sample_idx]
        file_idx = int(os.path.basename(img_path)[5:-4])
        txt_path = os.path.join(data_dir, "texts", f"text1_{file_idx}.txt")

    pil_img = Image.open(img_path).convert("RGB")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if transform:
        return _transform(pil_img), text
    return pil_img, text


def list_samples(data_dir):
    """Return sorted list of (file_idx, img_path) pairs in data_dir."""
    img_files = sorted(
        glob.glob(os.path.join(data_dir, "images", "img1_*.png")),
        key=lambda p: int(os.path.basename(p)[5:-4]),
    )
    result = []
    for p in img_files:
        try:
            idx = int(os.path.basename(p)[5:-4])
        except ValueError:
            continue
        result.append((idx, p))
    return result
