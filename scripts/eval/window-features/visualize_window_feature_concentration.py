#!/usr/bin/env python3
"""Visualize feature concentration and predicted/aligned token for every image window.

The script can create either a summary image for the full line, or one PNG per
model window.  The per-window mode is useful for debugging local embeddings: it
shows the window crop, the feature-concentration image, and an early-fusion
blend of both for every window.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arabic_span_text_encoder import ArabicSpanTextEncoder  # noqa: E402
from arabic_token_text_encoder import ArabicTokenTextEncoder  # noqa: E402
from embeddingModel import EmbeddingModel  # noqa: E402
from span_alignment_loss import hard_span_dtw_path  # noqa: E402
from textEmbedding import TextEmbedding  # noqa: E402

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:  # pragma: no cover - optional visualization dependency
    arabic_reshaper = None
    get_display = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def display_text(text: str) -> str:
    """Shape Arabic labels when optional bidi packages are installed."""
    if text is None:
        return ""
    text = str(text)
    if arabic_reshaper is not None and get_display is not None:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text


def safe_filename_token(text: str, max_len: int = 24) -> str:
    text = str(text or "empty").strip() or "empty"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\u0600-\u06FF]+", "", text)
    return text[:max_len] or "token"


def load_image(path: str, height: int, width: int) -> Tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((int(width), int(height)), Image.BILINEAR)
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pil, to_tensor(pil).unsqueeze(0)


def read_text(path: Optional[str], text: Optional[str]) -> str:
    if text is not None:
        return text
    if path is None:
        raise ValueError("Provide either --text or --text-path, or use --data-dir with --index.")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def infer_paths(args) -> Tuple[str, str]:
    if args.line is not None:
        if args.text_path is None and args.text is None:
            raise ValueError("When using --line, also provide --text or --text-path.")
        return args.line, args.text_path
    if args.data_dir is None or args.index is None:
        raise ValueError("Provide --line + --text/--text-path, or --data-dir + --index.")
    which = int(args.which_line)
    image_path = os.path.join(args.data_dir, "images", f"img{which}_{args.index}.png")
    text_path = os.path.join(args.data_dir, "texts", f"text{which}_{args.index}.txt")
    return image_path, text_path


def checkpoint_config(loaded) -> dict:
    if isinstance(loaded, dict):
        return dict(loaded.get("model_config") or {})
    return {}


def extract_image_state(loaded):
    if isinstance(loaded, dict):
        if "image_model_state_dict" in loaded:
            return loaded["image_model_state_dict"]
        if "model_state_dict" in loaded:
            return loaded["model_state_dict"]
    return loaded


def build_image_model(args, cfg: dict, device: str) -> EmbeddingModel:
    window_size = int(args.window_size or cfg.get("window_size", 32))
    stride = int(args.stride or cfg.get("stride", max(1, window_size // 2)))
    vector_size = int(args.vector_size or cfg.get("vector_size", 128))
    use_bilstm = bool(cfg.get("use_bilstm", True)) and not args.no_bilstm
    bilstm_layers = int(cfg.get("bilstm_layers", 1))
    bilstm_hidden_dim = cfg.get("bilstm_hidden_dim", vector_size)
    if bilstm_hidden_dim is not None:
        bilstm_hidden_dim = int(bilstm_hidden_dim)

    model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=bool(args.use_flip),
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
    ).to(device)
    return model


def build_text_encoder(args, cfg: dict, loaded, device: str):
    text_encoder_type = str(args.text_encoder_type or cfg.get("text_encoder_type", "arabic_span")).lower()
    vector_size = int(args.vector_size or cfg.get("vector_size", 128))

    if text_encoder_type == "arabic_span":
        encoder = ArabicSpanTextEncoder(
            model_name=args.arabic_text_model_name or cfg.get("arabic_text_model_name", "aubmindlab/bert-base-arabertv02"),
            output_dim=vector_size,
            max_span_chars=int(args.max_span_chars or cfg.get("max_text_span_chars", 3)),
            freeze_backbone=True,
            device=device,
            strip_text_edges=bool(cfg.get("strip_span_text_edges", True)),
            cache_size=int(cfg.get("span_feature_cache_size", 2048)),
            cache_dtype=str(cfg.get("span_feature_cache_dtype", "float16")),
        )
    elif text_encoder_type == "arabic_token":
        encoder = ArabicTokenTextEncoder(
            model_name=args.arabic_text_model_name or cfg.get("arabic_text_model_name", "aubmindlab/bert-base-arabertv02"),
            output_dim=vector_size,
            max_token_chars=int(args.max_token_chars or cfg.get("max_text_token_chars", 3)),
            freeze_backbone=True,
            device=device,
        )
    elif text_encoder_type == "char":
        encoder = TextEmbedding(embedding_dim=vector_size)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        encoder = encoder.to(device)
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder_type!r}")

    if isinstance(loaded, dict):
        state = loaded.get("text_encoder_state_dict") or loaded.get("text_embedder_state_dict")
        if state:
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"[warn] text encoder state loaded with missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
        else:
            print("[warn] checkpoint has no text encoder state; token assignments may be unreliable.", flush=True)
    encoder.eval()
    return encoder, text_encoder_type


def get_embeddings(model: EmbeddingModel, image_tensor: torch.Tensor, device: str):
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        contextual, local, ink_ratio = model(image_tensor, return_local=True, return_ink=True)
    return contextual[0].float(), local[0].float(), ink_ratio[0].float() if ink_ratio is not None else None


def feature_concentration(features: torch.Tensor, metric: str = "topk_mass", top_k: int = 8) -> np.ndarray:
    z = features.detach().float().cpu().numpy()
    abs_z = np.abs(z)
    eps = 1e-8
    dim = max(1, abs_z.shape[1])

    if metric == "hoyer":
        l1 = abs_z.sum(axis=1)
        l2 = np.sqrt((z ** 2).sum(axis=1)) + eps
        denom = max(eps, math.sqrt(dim) - 1.0)
        score = (math.sqrt(dim) - (l1 / l2)) / denom
        return np.clip(score, 0.0, 1.0)

    if metric == "energy":
        energy = np.sqrt((z ** 2).sum(axis=1))
        if np.max(energy) <= eps:
            return np.zeros_like(energy)
        return (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + eps)

    # Default: how much of the absolute activation mass is carried by the top-k dimensions.
    k = min(max(1, int(top_k)), dim)
    sorted_abs = np.sort(abs_z, axis=1)[:, ::-1]
    return sorted_abs[:, :k].sum(axis=1) / (abs_z.sum(axis=1) + eps)


def feature_heatmap_matrix(features: torch.Tensor, normalize: str = "per_window") -> np.ndarray:
    z = np.abs(features.detach().float().cpu().numpy())
    eps = 1e-8
    if normalize == "per_window":
        z = z / (z.max(axis=1, keepdims=True) + eps)
    elif normalize == "global":
        z = z / (z.max() + eps)
    elif normalize == "zscore":
        z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + eps)
        z = np.abs(z)
        z = z / (z.max() + eps)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown --feature-normalize {normalize!r}")
    return z.T


def aligned_tokens_for_windows(
    text_encoder,
    text: str,
    image_embeddings_for_alignment: torch.Tensor,
    args,
) -> Tuple[List[str], list, object]:
    encoding = text_encoder(text, use_cache=True) if callable(getattr(text_encoder, "forward", None)) else text_encoder(text)
    norm_img = F.normalize(image_embeddings_for_alignment.float(), p=2, dim=-1)
    path = hard_span_dtw_path(
        encoding,
        norm_img,
        temperature=float(args.temperature),
        max_windows=int(args.max_windows_per_span),
        window_count_penalty=float(args.window_count_penalty),
    )
    num_windows = int(norm_img.shape[0])
    labels = ["?"] * num_windows
    span_indices = [-1] * num_windows
    for step in path:
        span_idx = int(step["span_idx"])
        token = str(step.get("text", encoding.texts[span_idx]))
        for w in range(int(step["window_start"]), int(step["window_end"])):
            if 0 <= w < num_windows:
                labels[w] = token
                span_indices[w] = span_idx
    return labels, path, encoding


def window_pixel_range(window_idx: int, num_windows: int, window_size: int, stride: int, width: int, use_flip: bool) -> Tuple[int, int]:
    if use_flip:
        visual_idx = num_windows - 1 - window_idx
    else:
        visual_idx = window_idx
    x0 = int(visual_idx * stride)
    x1 = int(x0 + window_size)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    return x0, x1


def crop_window_image(pil_img: Image.Image, window_idx: int, num_windows: int, args) -> Tuple[Image.Image, int, int]:
    x0, x1 = window_pixel_range(window_idx, num_windows, args.window_size, args.stride, pil_img.width, args.use_flip)
    if x1 <= x0:
        x1 = min(pil_img.width, x0 + max(1, args.window_size))
    crop = pil_img.crop((x0, 0, x1, pil_img.height)).convert("RGB")
    return crop, x0, x1


def feature_vector_grid(feature_vec: torch.Tensor) -> np.ndarray:
    values = torch.abs(feature_vec.detach().float()).cpu().numpy()
    dim = int(values.shape[0])
    cols = int(math.ceil(math.sqrt(dim)))
    rows = int(math.ceil(dim / cols))
    grid = np.zeros((rows, cols), dtype=np.float32)
    grid.flat[:dim] = values
    if grid.max() > 1e-8:
        grid = grid / (grid.max() + 1e-8)
    return grid


def feature_vector_colored_image(feature_vec: torch.Tensor, width: int, height: int, cmap_name: str) -> Image.Image:
    grid = feature_vector_grid(feature_vec)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(grid)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((int(width), int(height)), Image.NEAREST)


def make_early_fusion(crop: Image.Image, heatmap: Image.Image, alpha: float) -> Image.Image:
    crop = crop.convert("RGB")
    heatmap = heatmap.resize(crop.size, Image.NEAREST).convert("RGB")
    alpha = max(0.0, min(1.0, float(alpha)))
    return Image.blend(crop, heatmap, alpha=alpha)


def top_feature_dims_for_one(feature_vec: torch.Tensor, top_k: int) -> str:
    z = torch.abs(feature_vec.detach().float()).cpu()
    k = min(max(1, int(top_k)), z.numel())
    idx = torch.topk(z, k=k).indices.tolist()
    return ";".join(str(int(x)) for x in idx)


def save_per_window_images(
    pil_img: Image.Image,
    labels: List[str],
    concentration: np.ndarray,
    features: torch.Tensor,
    args,
    out_dir: str,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    num_windows = len(labels)

    for i in range(num_windows):
        crop, x0, x1 = crop_window_image(pil_img, i, num_windows, args)
        # Upscale narrow windows so the visual crop is readable.
        display_w = max(int(args.per_window_image_width), int(crop.width) * int(args.per_window_scale))
        display_h = int(args.per_window_image_height)
        crop_display = crop.resize((display_w, display_h), Image.BILINEAR)
        heatmap = feature_vector_colored_image(features[i], display_w, display_h, args.cmap)
        fused = make_early_fusion(crop_display, heatmap, args.early_fusion_alpha)
        token_label = display_text(labels[i])
        top_dims = top_feature_dims_for_one(features[i], args.csv_top_features)

        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        axes = fig.subplots(1, 3)
        panels = [
            (crop_display, "window image"),
            (heatmap, "feature concentration image"),
            (fused, f"early fusion alpha={args.early_fusion_alpha:.2f}"),
        ]
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(
            f"window {i:03d} | x={x0}:{x1} | token/span={token_label} | "
            f"concentration={float(concentration[i]):.4f} | top dims={top_dims}",
            fontsize=11,
        )
        filename = f"window_{i:03d}_x{x0:04d}_{x1:04d}_{safe_filename_token(labels[i])}.png"
        fig.savefig(out_path / filename, dpi=args.per_window_dpi, bbox_inches="tight")
        plt.close(fig)


def top_feature_dims(features: torch.Tensor, top_k: int) -> List[str]:
    z = torch.abs(features.detach().float()).cpu()
    k = min(max(1, int(top_k)), z.shape[1])
    top = torch.topk(z, k=k, dim=1).indices.numpy()
    return [";".join(str(int(x)) for x in row) for row in top]


def save_window_csv(
    csv_path: str,
    labels: List[str],
    concentration: np.ndarray,
    features: torch.Tensor,
    args,
    num_windows: int,
    image_width: int,
):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    top_dims = top_feature_dims(features, args.csv_top_features)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window_index_model_order",
                "x0_visual",
                "x1_visual",
                "token_or_span",
                "concentration",
                "top_feature_dimensions",
            ]
        )
        for i in range(num_windows):
            x0, x1 = window_pixel_range(i, num_windows, args.window_size, args.stride, image_width, args.use_flip)
            writer.writerow([i, x0, x1, labels[i], f"{float(concentration[i]):.6f}", top_dims[i]])


def plot_visualization(
    pil_img: Image.Image,
    labels: List[str],
    concentration: np.ndarray,
    features: torch.Tensor,
    path: list,
    args,
    out_path: str,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img_np = np.array(pil_img)
    height, width = img_np.shape[:2]
    num_windows = len(labels)
    feat_hm = feature_heatmap_matrix(features, normalize=args.feature_normalize)

    fig_width = max(16, num_windows * 0.33)
    fig = plt.figure(figsize=(fig_width, 12), constrained_layout=False)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 0.8, 2.8, 0.3], hspace=0.45)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_np)
    ax_img.set_title(
        f"Line windows colored by {args.concentration_metric} concentration | token/span shown for each model window",
        fontsize=12,
    )
    ax_img.set_xlim(0, width)
    ax_img.set_ylim(height + 28, -18)
    ax_img.axis("off")
    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(float(np.min(concentration)), float(np.max(concentration)) + 1e-8)

    for i, score in enumerate(concentration):
        x0, x1 = window_pixel_range(i, num_windows, args.window_size, args.stride, width, args.use_flip)
        color = cmap(norm(float(score)))
        rect = plt.Rectangle((x0, 0), max(1, x1 - x0), height, fill=False, edgecolor=color, linewidth=1.4, alpha=0.95)
        ax_img.add_patch(rect)
        xc = 0.5 * (x0 + x1)
        ax_img.text(
            xc,
            height + 9,
            display_text(labels[i]),
            ha="center",
            va="top",
            fontsize=args.token_fontsize,
            rotation=90 if args.rotate_tokens else 0,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_img, fraction=0.018, pad=0.01)
    cbar.set_label("feature concentration")

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.bar(np.arange(num_windows), concentration, width=0.85)
    ax_bar.set_ylabel("concentration")
    ax_bar.set_xlim(-0.5, num_windows - 0.5)
    ax_bar.set_ylim(0, max(1.0, float(np.max(concentration)) * 1.05))
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_xticks(np.arange(num_windows))
    ax_bar.set_xticklabels([display_text(t) for t in labels], rotation=90, fontsize=args.token_fontsize)
    ax_bar.set_title("Per-window feature concentration; x-axis label is the aligned token/span")

    ax_feat = fig.add_subplot(gs[2])
    im = ax_feat.imshow(feat_hm, aspect="auto", interpolation="nearest", cmap=args.cmap)
    ax_feat.set_title(
        f"Feature activation heatmap ({args.feature_space} embeddings, normalize={args.feature_normalize})"
    )
    ax_feat.set_xlabel("window index in model order" + (" (RTL flipped)" if args.use_flip else ""))
    ax_feat.set_ylabel("feature dimension")
    ax_feat.set_xticks(np.arange(num_windows))
    ax_feat.set_xticklabels([f"{i}\n{display_text(labels[i])}" for i in range(num_windows)], rotation=90, fontsize=args.token_fontsize)
    y_step = max(1, int(features.shape[1] // 16))
    ax_feat.set_yticks(np.arange(0, features.shape[1], y_step))
    fig.colorbar(im, ax=ax_feat, fraction=0.018, pad=0.01).set_label("|feature activation|")

    for step in path:
        w0 = int(step["window_start"])
        w1 = int(step["window_end"])
        if w1 <= w0:
            continue
        ax_feat.axvline(w0 - 0.5, color="white", linewidth=0.5, alpha=0.7)
        ax_feat.axvline(w1 - 0.5, color="white", linewidth=0.5, alpha=0.7)

    ax_note = fig.add_subplot(gs[3])
    ax_note.axis("off")
    ax_note.text(
        0,
        0.4,
        "Per-window images can be enabled with --save-per-window-images. CSV can be disabled with --no-csv.",
        fontsize=10,
    )

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize feature concentration and aligned token for every image window.")
    parser.add_argument("--weights", required=True, help="Checkpoint path containing image model and text encoder state.")
    parser.add_argument("--line", default=None, help="Path to one line image. Alternative to --data-dir/--index.")
    parser.add_argument("--text", default=None, help="Text string for the line.")
    parser.add_argument("--text-path", default=None, help="Path to text file for --line.")
    parser.add_argument("--data-dir", default=None, help="Dataset root with images/ and texts/ folders.")
    parser.add_argument("--index", type=int, default=None, help="1-based dataset sample index.")
    parser.add_argument("--which-line", type=int, choices=[1, 2], default=1, help="Use img1/text1 or img2/text2 in dataset mode.")
    parser.add_argument("--output", required=True, help="Output PNG path for the optional summary image.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path. Defaults to output path with .csv suffix.")
    parser.add_argument("--no-csv", action="store_true", help="Do not save the CSV file.")
    parser.add_argument("--no-summary-image", action="store_true", help="Do not save the full-line summary image.")
    parser.add_argument("--save-per-window-images", action="store_true", help="Save one image per model window.")
    parser.add_argument("--per-window-output-dir", default=None, help="Directory for per-window PNG files.")
    parser.add_argument("--early-fusion-alpha", type=float, default=0.55, help="Blend weight for feature heatmap over window image.")
    parser.add_argument("--per-window-image-width", type=int, default=256)
    parser.add_argument("--per-window-image-height", type=int, default=256)
    parser.add_argument("--per-window-scale", type=int, default=8)
    parser.add_argument("--per-window-dpi", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--use-flip", action="store_true", help="Use RTL flipped model-window order, same as Arabic training/eval.")
    parser.add_argument("--no-bilstm", action="store_true")
    parser.add_argument("--feature-space", choices=["local", "contextual"], default="local")
    parser.add_argument("--alignment-space", choices=["local", "contextual"], default="contextual")
    parser.add_argument("--text-encoder-type", choices=["arabic_span", "arabic_token", "char"], default=None)
    parser.add_argument("--arabic-text-model-name", default=None)
    parser.add_argument("--max-span-chars", type=int, default=None)
    parser.add_argument("--max-token-chars", type=int, default=None)
    parser.add_argument("--max-windows-per-span", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--window-count-penalty", type=float, default=0.01)
    parser.add_argument("--concentration-metric", choices=["topk_mass", "hoyer", "energy"], default="topk_mass")
    parser.add_argument("--concentration-top-k", type=int, default=8)
    parser.add_argument("--feature-normalize", choices=["per_window", "global", "zscore", "none"], default="per_window")
    parser.add_argument("--csv-top-features", type=int, default=8)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--token-fontsize", type=float, default=6.0)
    parser.add_argument("--rotate-tokens", action="store_true", default=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path, text_path = infer_paths(args)
    text = read_text(text_path, args.text)

    loaded = torch.load(args.weights, map_location=args.device)
    cfg = checkpoint_config(loaded)
    if args.window_size is None:
        args.window_size = int(cfg.get("window_size", 32))
    if args.stride is None:
        args.stride = int(cfg.get("stride", max(1, args.window_size // 2)))
    if args.vector_size is None:
        args.vector_size = int(cfg.get("vector_size", 128))
    if args.max_span_chars is None:
        args.max_span_chars = int(cfg.get("max_text_span_chars", 3))
    if args.max_token_chars is None:
        args.max_token_chars = int(cfg.get("max_text_token_chars", 3))
    if args.max_windows_per_span is None:
        args.max_windows_per_span = int(cfg.get("max_windows_per_span", 4))
    if "contrastive_temperature" in cfg and args.temperature == 0.07:
        args.temperature = float(cfg.get("contrastive_temperature", 0.07))

    model = build_image_model(args, cfg, args.device)
    model.load_state_dict(extract_image_state(loaded), strict=False)
    text_encoder, _encoder_type = build_text_encoder(args, cfg, loaded, args.device)

    pil_img, image_tensor = load_image(image_path, args.height, args.width)
    contextual, local, _ink = get_embeddings(model, image_tensor, args.device)
    features = local if args.feature_space == "local" else contextual
    align_features = contextual if args.alignment_space == "contextual" else local

    labels, path, _encoding = aligned_tokens_for_windows(text_encoder, text, align_features, args)
    concentration = feature_concentration(features, metric=args.concentration_metric, top_k=args.concentration_top_k)

    if not args.no_summary_image:
        plot_visualization(pil_img, labels, concentration, features, path, args, args.output)
        print(f"saved summary figure: {args.output}")

    if args.save_per_window_images:
        per_window_dir = args.per_window_output_dir or str(Path(args.output).with_suffix("")) + "_windows"
        save_per_window_images(pil_img, labels, concentration, features, args, per_window_dir)
        print(f"saved per-window images: {per_window_dir}")

    if not args.no_csv:
        csv_path = args.csv_output or str(Path(args.output).with_suffix(".csv"))
        save_window_csv(csv_path, labels, concentration, features, args, len(labels), pil_img.width)
        print(f"saved csv: {csv_path}")

    print(f"windows={len(labels)} feature_dim={features.shape[1]} text_length={len(text)}")


if __name__ == "__main__":
    main()
