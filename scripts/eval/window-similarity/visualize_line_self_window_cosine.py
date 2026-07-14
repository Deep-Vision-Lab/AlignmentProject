#!/usr/bin/env python3
"""Visualize cosine similarity between every window of a line and every other window of the same line.

This creates a self-similarity heatmap:

    x-axis = model windows from the selected line
    y-axis = the same model windows from the same line
    cell(i, j) = cosine(window_i_embedding, window_j_embedding)

The top and left axes show the actual window image slices with visible gaps so it
is easy to map every cosine cell to the two visual windows it compares.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddingModel import EmbeddingModel  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    _RESAMPLE_NEAREST = Image.Resampling.NEAREST
    _ROTATE_90 = Image.Transpose.ROTATE_90
except AttributeError:  # Pillow<9 compatibility
    _RESAMPLE_BILINEAR = Image.BILINEAR
    _RESAMPLE_NEAREST = Image.NEAREST
    _ROTATE_90 = Image.ROTATE_90


def infer_image_path(args) -> str:
    if args.line is not None:
        return args.line
    if args.data_dir is None or args.index is None:
        raise ValueError("Provide --line or --data-dir + --index.")
    return os.path.join(args.data_dir, "images", f"img{int(args.which_line)}_{int(args.index)}.png")


def load_image(path: str, height: int, width: int) -> Tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((int(width), int(height)), _RESAMPLE_BILINEAR)
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pil, to_tensor(pil).unsqueeze(0)


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

    return EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=bool(args.use_flip),
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
    ).to(device)


def get_window_embeddings(model: EmbeddingModel, image_tensor: torch.Tensor, device: str, feature_space: str) -> torch.Tensor:
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        contextual, local = model(image_tensor, return_local=True)
    emb = local if feature_space == "local" else contextual
    return emb[0].float()


def cosine_matrix(features: torch.Tensor) -> np.ndarray:
    z = F.normalize(features.float(), p=2, dim=-1)
    sim = torch.matmul(z, z.T)
    return sim.detach().cpu().numpy()


def model_window_to_visual_range(
    window_idx: int,
    num_windows: int,
    window_size: int,
    stride: int,
    image_width: int,
    use_flip: bool,
) -> Tuple[int, int]:
    visual_idx = (num_windows - 1 - window_idx) if use_flip else window_idx
    x0 = int(visual_idx * stride)
    x1 = int(x0 + window_size)
    x0 = max(0, min(image_width, x0))
    x1 = max(0, min(image_width, x1))
    if x1 <= x0:
        x1 = min(image_width, x0 + max(1, window_size))
    return x0, x1


def crop_visual_slice(
    image: Image.Image,
    window_idx: int,
    num_windows: int,
    window_size: int,
    stride: int,
    use_flip: bool,
    mode: str,
) -> Image.Image:
    x0, x1 = model_window_to_visual_range(window_idx, num_windows, window_size, stride, image.width, use_flip)
    if mode == "window" or stride >= window_size:
        return image.crop((x0, 0, x1, image.height)).convert("RGB")

    if mode != "nonoverlap":
        raise ValueError(f"Unknown --axis-slice-mode {mode!r}")

    # Display-only slice: use the central stride-sized part of the model window so
    # neighboring thumbnails do not repeatedly show the same overlapped pixels.
    center = 0.5 * (x0 + x1)
    half = max(1.0, float(stride) / 2.0)
    sx0 = int(round(center - half))
    sx1 = int(round(center + half))
    sx0 = max(0, min(image.width - 1, sx0))
    sx1 = max(sx0 + 1, min(image.width, sx1))
    return image.crop((sx0, 0, sx1, image.height)).convert("RGB")


def hstack(images):
    width = sum(im.width for im in images)
    height = max(im.height for im in images) if images else 1
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def vstack(images):
    width = max(im.width for im in images) if images else 1
    height = sum(im.height for im in images)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def make_x_axis_strip(image: Image.Image, num_windows: int, args) -> np.ndarray:
    cells = []
    gap = int(args.window_gap_pixels)
    for idx in range(num_windows):
        crop = crop_visual_slice(
            image,
            idx,
            num_windows,
            int(args.window_size),
            int(args.stride),
            bool(args.use_flip),
            args.axis_slice_mode,
        )
        crop = crop.resize((int(args.axis_cell_pixels), int(args.x_strip_height)), _RESAMPLE_BILINEAR)
        cell = Image.new(
            "RGB",
            (int(args.axis_cell_pixels) + gap, int(args.x_strip_height)),
            (255, 255, 255),
        )
        cell.paste(crop, (0, 0))
        cells.append(cell)
    if cells:
        cells[-1] = cells[-1].crop((0, 0, int(args.axis_cell_pixels), int(args.x_strip_height)))
    return np.array(hstack(cells))


def make_y_axis_strip(image: Image.Image, num_windows: int, args) -> np.ndarray:
    cells = []
    gap = int(args.window_gap_pixels)
    for idx in range(num_windows):
        crop = crop_visual_slice(
            image,
            idx,
            num_windows,
            int(args.window_size),
            int(args.stride),
            bool(args.use_flip),
            args.axis_slice_mode,
        )
        if args.y_axis_rotate:
            crop = crop.transpose(_ROTATE_90)
        if args.y_axis_flip:
            crop = ImageOps.mirror(crop)
        crop = crop.resize((int(args.y_strip_width), int(args.axis_cell_pixels)), _RESAMPLE_BILINEAR)
        cell = Image.new(
            "RGB",
            (int(args.y_strip_width), int(args.axis_cell_pixels) + gap),
            (255, 255, 255),
        )
        cell.paste(crop, (0, 0))
        cells.append(cell)
    if cells:
        cells[-1] = cells[-1].crop((0, 0, int(args.y_strip_width), int(args.axis_cell_pixels)))
    return np.array(vstack(cells))


def make_gap_heatmap_image(sim: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray]:
    cell = int(args.axis_cell_pixels)
    gap = int(args.window_gap_pixels)
    n = int(sim.shape[0])
    size = n * cell + max(0, n - 1) * gap
    value_canvas = np.full((size, size), np.nan, dtype=np.float32)
    for i in range(n):
        y0 = i * (cell + gap)
        y1 = y0 + cell
        for j in range(n):
            x0 = j * (cell + gap)
            x1 = x0 + cell
            value_canvas[y0:y1, x0:x1] = float(sim[i, j])

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="white")
    norm = plt.Normalize(float(args.vmin), float(args.vmax))
    rgba = cmap(norm(value_canvas))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb, value_canvas


def save_self_similarity_figure(sim: np.ndarray, image: Image.Image, args, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = int(sim.shape[0])
    x_strip = make_x_axis_strip(image, n, args)
    y_strip = make_y_axis_strip(image, n, args)
    heatmap_rgb, expanded_values = make_gap_heatmap_image(sim, args)

    # Match physical sizes to the generated pixel dimensions so axis thumbnails
    # align exactly with the heatmap cells.
    fig_w = max(10.0, (args.y_strip_width + heatmap_rgb.shape[1]) / 90.0)
    fig_h = max(10.0, (args.x_strip_height + heatmap_rgb.shape[0]) / 90.0)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[args.y_strip_width, heatmap_rgb.shape[1]],
        height_ratios=[args.x_strip_height, heatmap_rgb.shape[0]],
        wspace=0.02,
        hspace=0.02,
    )

    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")
    ax_corner.text(0.5, 0.5, "line self\nwindow cosine", ha="center", va="center", fontsize=9)

    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.imshow(x_strip)
    ax_x.set_title("x-axis: line windows in model order", fontsize=11)
    ax_x.axis("off")

    ax_y = fig.add_subplot(gs[1, 0])
    ax_y.imshow(y_strip)
    ax_y.set_ylabel("y-axis: same line windows", fontsize=10)
    ax_y.axis("off")

    ax_h = fig.add_subplot(gs[1, 1])
    ax_h.imshow(heatmap_rgb)
    ax_h.set_title(
        f"Cosine similarity: each window with every window in the same line | feature_space={args.feature_space}",
        fontsize=12,
    )
    ax_h.set_xlabel("window index in model order" + (" (RTL flipped)" if args.use_flip else ""))
    ax_h.set_ylabel("window index in model order" + (" (RTL flipped)" if args.use_flip else ""))

    cell = int(args.axis_cell_pixels)
    gap = int(args.window_gap_pixels)
    centers = np.arange(n) * (cell + gap) + cell / 2.0
    if args.show_all_ticks or n <= 32:
        tick_idx = np.arange(n)
    else:
        step = max(1, int(np.ceil(n / 32)))
        tick_idx = np.arange(0, n, step)
    ax_h.set_xticks(centers[tick_idx])
    ax_h.set_yticks(centers[tick_idx])
    ax_h.set_xticklabels([str(i) for i in tick_idx], rotation=90, fontsize=6)
    ax_h.set_yticklabels([str(i) for i in tick_idx], fontsize=6)

    # Mark the self-similarity diagonal.
    diag_x = centers
    diag_y = centers
    ax_h.plot(diag_x, diag_y, color="white", linewidth=1.0, alpha=0.9)

    if args.cell_values:
        for i in range(n):
            for j in range(n):
                ax_h.text(
                    centers[j],
                    centers[i],
                    f"{sim[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=float(args.cell_value_fontsize),
                    color="black" if sim[i, j] > (args.vmin + args.vmax) / 2 else "white",
                )

    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(float(args.vmin), float(args.vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_h, fraction=0.025, pad=0.01)
    cbar.set_label("cosine similarity")

    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Self cosine similarity heatmap for all windows in one line image.")
    parser.add_argument("--weights", required=True, help="Checkpoint path with image model weights.")
    parser.add_argument("--line", default=None, help="Path to one line image. Alternative to --data-dir/--index.")
    parser.add_argument("--data-dir", default=None, help="Dataset root with images/ folder.")
    parser.add_argument("--index", type=int, default=None, help="1-based dataset sample index.")
    parser.add_argument("--which-line", type=int, choices=[1, 2], default=1, help="Use img1 or img2 in dataset mode.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--feature-space", choices=["local", "contextual"], default="local")
    parser.add_argument("--use-flip", action="store_true", help="Use RTL flipped model-window order, same as Arabic training/eval.")
    parser.add_argument("--no-bilstm", action="store_true")
    parser.add_argument("--axis-slice-mode", choices=["nonoverlap", "window"], default="nonoverlap")
    parser.add_argument("--axis-cell-pixels", type=int, default=44)
    parser.add_argument("--window-gap-pixels", type=int, default=12)
    parser.add_argument("--x-strip-height", type=int, default=84)
    parser.add_argument("--y-strip-width", type=int, default=108)
    parser.add_argument("--no-y-axis-rotate", dest="y_axis_rotate", action="store_false")
    parser.set_defaults(y_axis_rotate=True)
    parser.add_argument("--y-axis-flip", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cell-values", action="store_true", help="Write cosine value inside every cell. Useful only for small matrices.")
    parser.add_argument("--cell-value-fontsize", type=float, default=3.2)
    parser.add_argument("--show-all-ticks", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = infer_image_path(args)

    loaded = torch.load(args.weights, map_location=args.device)
    cfg = checkpoint_config(loaded)
    if args.window_size is None:
        args.window_size = int(cfg.get("window_size", 32))
    if args.stride is None:
        args.stride = int(cfg.get("stride", max(1, args.window_size // 2)))
    if args.vector_size is None:
        args.vector_size = int(cfg.get("vector_size", 128))

    pil_img, image_tensor = load_image(image_path, args.height, args.width)
    model = build_image_model(args, cfg, args.device)
    model.load_state_dict(extract_image_state(loaded), strict=False)
    features = get_window_embeddings(model, image_tensor, args.device, args.feature_space)
    sim = cosine_matrix(features)

    save_self_similarity_figure(sim, pil_img, args, args.output)
    print(f"saved self window cosine heatmap: {args.output}")
    print(f"image={image_path}")
    print(f"windows={sim.shape[0]} feature_dim={features.shape[1]} feature_space={args.feature_space}")


if __name__ == "__main__":
    main()
