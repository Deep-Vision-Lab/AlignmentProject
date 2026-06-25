"""Utilities for loading trained models and dataset samples."""
import json
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
    transformer_num_layers, transformer_num_heads, transformer_ff_dim,
    transformer_dropout, transformer_activation, transformer_norm_first,
    transformer_positional_encoding, transformer_max_len,
)
from embeddingModel import EmbeddingModel, CrossAttentionSimilarity
from textEmbedding import build_text_embedder
from ctc_utils import CTCVocabulary
from NormalizeFuncs import normalize_func

_IMG_H, _IMG_W = 128, 1024

_transform = transforms.Compose([
    transforms.Resize((_IMG_H, _IMG_W)),
    transforms.ToTensor(),
])


def compute_stride(window_size_px, stride_ratio_value, window_overlap_mode="custom"):
    """Match train.py stride selection for figure generation."""
    window_size_px = int(window_size_px)
    mode = str(window_overlap_mode or "custom").lower()
    if mode == "no_overlap":
        return window_size_px
    if mode == "light_overlap":
        return max(1, window_size_px // 2)
    if mode == "dense_overlap":
        return max(1, window_size_px // 4)
    if mode == "custom":
        return max(1, int(window_size_px * float(stride_ratio_value)))
    raise ValueError(f"Unknown window_overlap_mode: {window_overlap_mode}")


def get_stride():
    """Return the stride used by default figure/training parameters."""
    return compute_stride(window_size, stride_ratio, "custom")


def _checkpoint_config(ckpt):
    return ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}


def load_checkpoint_config(checkpoint_path):
    """Return checkpoint model_config if present, otherwise an empty dict."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return _checkpoint_config(ckpt)


def _checkpoint_state(ckpt):
    if isinstance(ckpt, dict):
        return ckpt.get("image_model_state_dict", ckpt.get("model_state_dict", ckpt))
    return ckpt


def _build_model(device, use_bilstm_override=None, stride_override=None,
                 model_config=None):
    cfg = model_config or {}
    ws = int(cfg.get("window_size", window_size))
    sr = float(cfg.get("stride_ratio", stride_ratio))
    stride = stride_override
    if stride is None:
        stride = cfg.get("stride")
    if stride is None:
        stride = compute_stride(ws, sr, cfg.get("window_overlap_mode", "custom"))

    _vector_size = int(cfg.get("vector_size", vector_size))
    _lang = cfg.get("lang", lang)
    _ubil = cfg.get("use_bilstm", use_bilstm)
    if use_bilstm_override is not None:
        _ubil = use_bilstm_override
    _seq = str(cfg.get("sequence_encoder_type", "bilstm" if _ubil else "none")).lower()
    if use_bilstm_override is not None:
        _seq = "bilstm" if use_bilstm_override else "none"
    _layers = int(cfg.get("bilstm_layers", bilstm_layers))
    _hidden = int(cfg.get("bilstm_hidden_dim", bilstm_hidden_dim))

    return EmbeddingModel(
        window_size=ws,
        stride=int(stride),
        vector_size=_vector_size,
        device=device,
        use_flip=(str(_lang).lower() == "arabic"),
        sequence_encoder_type=_seq,
        use_bilstm=_ubil,
        bilstm_layers=_layers,
        bilstm_hidden_dim=_hidden,
        dropout=model_dropout,
        transformer_num_layers=int(cfg.get("transformer_num_layers", transformer_num_layers)),
        transformer_num_heads=int(cfg.get("transformer_num_heads", transformer_num_heads)),
        transformer_ff_dim=int(cfg.get("transformer_ff_dim", transformer_ff_dim)),
        transformer_dropout=float(cfg.get("transformer_dropout", transformer_dropout)),
        transformer_activation=cfg.get("transformer_activation", transformer_activation),
        transformer_norm_first=bool(cfg.get("transformer_norm_first", transformer_norm_first)),
        transformer_positional_encoding=cfg.get(
            "transformer_positional_encoding", transformer_positional_encoding
        ),
        transformer_max_len=int(cfg.get("transformer_max_len", transformer_max_len)),
    ).to(device)


def load_image_model(checkpoint_path, device, use_bilstm_override=None,
                     stride_override=None):
    """Load EmbeddingModel from a checkpoint file. Returns eval-mode model."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first:  python train.py --job_id <name>"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = _checkpoint_config(ckpt)
    model = _build_model(
        device,
        use_bilstm_override,
        stride_override=stride_override,
        model_config=cfg,
    )
    state = _checkpoint_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [model_loading] Missing keys ({len(missing)}): {missing[:3]} ...")
    model.eval()
    model._flip_verified = True   # suppress debug_flip/ output during figure gen
    print(f"  Loaded checkpoint: {checkpoint_path}")
    return model


def load_char_bank_if_available(checkpoint_dir_or_file, device):
    """
    Load a frozen character bank saved next to a char-pooling checkpoint.

    Returns:
        (char_bank_embeddings, char_to_idx, idx_to_char), or (None, None, None)
        with a warning when unavailable. This helper never trains or modifies
        the text embedder.
    """
    checkpoint_dir = (
        os.path.dirname(checkpoint_dir_or_file)
        if os.path.isfile(checkpoint_dir_or_file)
        else checkpoint_dir_or_file
    )
    bank_path = os.path.join(checkpoint_dir, "char_bank.json")
    if not os.path.isfile(bank_path):
        print(f"  [model_loading] Warning: char bank not found: {bank_path}")
        return None, None, None

    with open(bank_path, "r", encoding="utf-8") as handle:
        bank = json.load(handle)

    idx_to_char = list(bank.get("idx_to_char", []))
    char_to_idx = dict(bank.get("char_to_idx", {ch: i for i, ch in enumerate(idx_to_char)}))
    embeddings = bank.get("embeddings")
    if embeddings is None:
        print(
            f"  [model_loading] Warning: {bank_path} has no embeddings; "
            "top-k char-pool predictions will be disabled."
        )
        return None, char_to_idx, idx_to_char

    char_bank_embeddings = normalize_func(
        torch.tensor(embeddings, dtype=torch.float32, device=device)
    )
    return char_bank_embeddings, char_to_idx, idx_to_char


def load_cross_attention_module(checkpoint_path, device, use_for_figures=False):
    """
    Load optional cross-attention similarity for figure diagnostics.

    Returns (module, weight). By default this returns (None, 0.0) even if the
    checkpoint contains cross-attention, preserving dot-product similarity for
    fair comparisons.
    """
    if not use_for_figures:
        return None, 0.0
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = _checkpoint_config(ckpt)
    if not cfg.get("use_cross_attention", False):
        return None, 0.0
    weight = float(cfg.get("cross_attention_weight", 0.0))
    state = ckpt.get("cross_attention_state_dict") if isinstance(ckpt, dict) else None
    if state is None or weight <= 0:
        return None, 0.0
    module = CrossAttentionSimilarity(
        dim=int(cfg.get("vector_size", vector_size)),
        num_heads=int(cfg.get("cross_attention_num_heads", 4)),
        dropout=float(cfg.get("cross_attention_dropout", 0.1)),
        attention_type=cfg.get("cross_attention_type", "text_to_image"),
    ).to(device)
    module.load_state_dict(state)
    module.eval()
    print("  [model_loading] Loaded cross-attention similarity for figures.")
    return module, weight


def load_ctc_components(checkpoint_path, device, stride_override=None):
    """Load image model, CTC head, and CTCVocabulary from a CTC checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = _checkpoint_config(ckpt)
    vocab_data = ckpt.get("ctc_vocab")
    if vocab_data is None:
        vocab_path = os.path.join(os.path.dirname(checkpoint_path), "ctc_vocab.json")
        if not os.path.isfile(vocab_path):
            raise KeyError(
                "Checkpoint does not contain ctc_vocab and ctc_vocab.json was not found."
            )
        ctc_vocab = CTCVocabulary.load_json(vocab_path)
    else:
        ctc_vocab = CTCVocabulary.from_dict(vocab_data)

    model = load_image_model(checkpoint_path, device, stride_override=stride_override)
    head_state = ckpt.get("ctc_head_state_dict")
    if head_state is None:
        raise KeyError("Checkpoint does not contain ctc_head_state_dict.")
    head_in_dim = int(cfg.get("vector_size", vector_size))
    head = torch.nn.Linear(head_in_dim, len(ctc_vocab)).to(device)
    head.load_state_dict(head_state)
    head.eval()
    return model, head, ctc_vocab


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
