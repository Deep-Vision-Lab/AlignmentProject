#!/usr/bin/env python3
"""Visualize cosine similarity between every window in line1 and every window in line2.

This creates a line-pair window similarity heatmap:

    x-axis = windows from line1
    y-axis = windows from line2
    cell(row, col) = cosine(line2_window_row_embedding, line1_window_col_embedding)

The top and left axes show the actual separated window thumbnails, so every
cosine cell is visually tied to the two windows it compares.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

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
    _ROTATE_90 = Image.Transpose.ROTATE_90
except AttributeError:  # Pillow<9 compatibility
    _RESAMPLE_BILINEAR = Image.BILINEAR
    _ROTATE_90 = Image.ROTATE_90


def infer_image_paths(args) -> Tuple[str, str]:
    if args.line1 is not None or args.line2 is not None:
        if args.line1 is None or args.line2 is None:
            raise ValueError("Provide both --line1 and --line2, or use --data-dir + --index.")
        return args.line1, args.line2

    if args.data_dir is None or args.index is None:
        raise ValueError("Provide --line1 + --line2, or --data-dir + --index.")

    image1 = os.path.join(args.data_dir, "images", f"img1_{int(args.index)}.png")
    image2 = os.path.join(args.data_dir, "images", f"img2_{int(args.index)}.png")
    return image1, image2


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


def get_window_embeddings_pair(
    model: EmbeddingModel,
    image1_tensor: torch.Tensor,
    image2_tensor: torch.Tensor,
    device: str,
    feature_space: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    images = torch.cat([image1_tensor, image2_tensor], dim=0).to(device)
    with torch.no_grad():
        contextual, local = model(images, return_local=True)
    emb = local if feature_space == "local" else contextual
    return emb[0].float(), emb[1].float()


def cosine_matrix(line1_features: torch.Tensor, line2_features: torch.Tensor) -> np.ndarray:
    """Return [line2_windows, line1_windows] cosine matrix."""
    z1 = F.normalize(line1_features.float(), p=2, dim=-1)
    z2 = F.normalize(line2_features.float(), p=2, dim=-1)
    sim = torch.matmul(z2, z1.T)
    return sim.detach().cpu().numpy()


def base_display_indices(num_windows: int, args) -> np.ndarray:
    if args.display_order == "visual" and args.use_flip:
        return np.arange(num_windows - 1, -1, -1, dtype=np.int64)
    return np.arange(num_windows, dtype=np.int64)


def axis_display_indices(n1: int, n2: int, args) -> Tuple[np.ndarray, np.ndarray]:
    base_x = base_display_indices(n1, args)
    base_y = base_display_indices(n2, args)
    x_indices = base_x[::-1].copy() if args.reverse_x_axis else base_x.copy()
    y_indices = base_y[::-1].copy() if args.reverse_y_axis else base_y.copy()
    return x_indices, y_indices


def reorder_similarity_for_display(sim: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
    return sim[np.ix_(y_indices, x_indices)]


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
    model_window_idx: int,
    num_windows: int,
    window_size: int,
    stride: int,
    use_flip: bool,
    mode: str,
) -> Image.Image:
    x0, x1 = model_window_to_visual_range(model_window_idx, num_windows, window_size, stride, image.width, use_flip)
    if mode == "window" or stride >= window_size:
        return image.crop((x0, 0, x1, image.height)).convert("RGB")

    if mode != "nonoverlap":
        raise ValueError(f"Unknown --axis-slice-mode {mode!r}")

    # Display-only slice: show the centered stride-sized part of each overlapping
    # model window so the thumbnails are separated and less visually repetitive.
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


def make_x_axis_strip(image: Image.Image, x_indices: Sequence[int], num_windows: int, args) -> np.ndarray:
    cells = []
    gap = int(args.window_gap_pixels)
    mirror = bool(args.mirror_axis_windows or args.mirror_x_axis_windows)
    for model_idx in x_indices:
        crop = crop_visual_slice(
            image,
            int(model_idx),
            num_windows,
            int(args.window_size),
            int(args.stride),
            bool(args.use_flip),
            args.axis_slice_mode,
        )
        if mirror:
            crop = ImageOps.mirror(crop)
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


def make_y_axis_strip(image: Image.Image, y_indices: Sequence[int], num_windows: int, args) -> np.ndarray:
    cells = []
    gap = int(args.window_gap_pixels)
    mirror = bool(args.mirror_axis_windows or args.mirror_y_axis_windows)
    for model_idx in y_indices:
        crop = crop_visual_slice(
            image,
            int(model_idx),
            num_windows,
            int(args.window_size),
            int(args.stride),
            bool(args.use_flip),
            args.axis_slice_mode,
        )
        if mirror:
            crop = ImageOps.mirror(crop)
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


def make_gap_heatmap_image(sim: np.ndarray, args) -> np.ndarray:
    cell = int(args.axis_cell_pixels)
    gap = int(args.window_gap_pixels)
    rows, cols = int(sim.shape[0]), int(sim.shape[1])
    height = rows * cell + max(0, rows - 1) * gap
    width = cols * cell + max(0, cols - 1) * gap
    value_canvas = np.full((height, width), np.nan, dtype=np.float32)
    for i in range(rows):
        y0 = i * (cell + gap)
        y1 = y0 + cell
        for j in range(cols):
            x0 = j * (cell + gap)
            x1 = x0 + cell
            value_canvas[y0:y1, x0:x1] = float(sim[i, j])

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="white")
    norm = plt.Normalize(float(args.vmin), float(args.vmax))
    rgba = cmap(norm(value_canvas))
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _axis_order_label(base_order: str, reverse: bool) -> str:
    suffix = " reversed" if reverse else ""
    return f"{base_order}{suffix}"


def save_pair_similarity_figure(
    sim_model: np.ndarray,
    line1_image: Image.Image,
    line2_image: Image.Image,
    args,
    out_path: str,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n2, n1 = int(sim_model.shape[0]), int(sim_model.shape[1])
    x_indices, y_indices = axis_display_indices(n1, n2, args)
    sim = reorder_similarity_for_display(sim_model, x_indices, y_indices)

    x_strip = make_x_axis_strip(line1_image, x_indices, n1, args)
    y_strip = make_y_axis_strip(line2_image, y_indices, n2, args)
    heatmap_rgb = make_gap_heatmap_image(sim, args)

    cbar_width = int(args.colorbar_width)
    fig_w = max(10.0, (args.y_strip_width + heatmap_rgb.shape[1] + cbar_width) / float(args.figure_dpi_scale))
    fig_h = max(10.0, (args.x_strip_height + heatmap_rgb.shape[0]) / float(args.figure_dpi_scale))
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[args.y_strip_width, heatmap_rgb.shape[1], cbar_width],
        height_ratios=[args.x_strip_height, heatmap_rgb.shape[0]],
        wspace=0.0,
        hspace=0.0,
    )

    base_label = "visual image order" if args.display_order == "visual" else "model order"
    x_order_label = _axis_order_label(base_label, args.reverse_x_axis)
    y_order_label = _axis_order_label(base_label, args.reverse_y_axis)
    x_mirror_label = " | x mirrored" if (args.mirror_axis_windows or args.mirror_x_axis_windows) else ""
    y_mirror_label = " | y mirrored" if (args.mirror_axis_windows or args.mirror_y_axis_windows) else ""

    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis("off")
    ax_corner.text(0.5, 0.5, "line1 ↔ line2\nwindow cosine", ha="center", va="center", fontsize=9)

    ax_x = fig.add_subplot(gs[0, 1])
    ax_x.imshow(x_strip, aspect="auto")
    ax_x.set_title(f"x-axis: line1 windows in {x_order_label}{x_mirror_label}", fontsize=11)
    ax_x.axis("off")

    ax_y = fig.add_subplot(gs[1, 0])
    ax_y.imshow(y_strip, aspect="auto")
    ax_y.set_ylabel(f"y-axis: line2 windows in {y_order_label}{y_mirror_label}", fontsize=10)
    ax_y.axis("off")

    ax_h = fig.add_subplot(gs[1, 1])
    ax_h.imshow(heatmap_rgb, aspect="auto")
    ax_h.set_title(
        f"Cosine similarity: line2 windows × line1 windows | feature_space={args.feature_space}",
        fontsize=12,
    )

    cell = int(args.axis_cell_pixels)
    gap = int(args.window_gap_pixels)
    x_centers = np.arange(n1) * (cell + gap) + cell / 2.0
    y_centers = np.arange(n2) * (cell + gap) + cell / 2.0

    if args.show_all_ticks or n1 <= 32:
        x_tick_pos = np.arange(n1)
    else:
        x_step = max(1, int(np.ceil(n1 / 32)))
        x_tick_pos = np.arange(0, n1, x_step)

    if args.show_all_ticks or n2 <= 32:
        y_tick_pos = np.arange(n2)
    else:
        y_step = max(1, int(np.ceil(n2 / 32)))
        y_tick_pos = np.arange(0, n2, y_step)

    if args.tick_labels == "model":
        x_tick_labels = [str(int(x_indices[i])) for i in x_tick_pos]
        y_tick_labels = [str(int(y_indices[i])) for i in y_tick_pos]
        label_suffix = "model idx"
    else:
        x_tick_labels = [str(int(i)) for i in x_tick_pos]
        y_tick_labels = [str(int(i)) for i in y_tick_pos]
        label_suffix = "shown idx"

    ax_h.set_xticks(x_centers[x_tick_pos])
    ax_h.set_yticks(y_centers[y_tick_pos])
    ax_h.set_xticklabels(x_tick_labels, rotation=90, fontsize=6)
    ax_h.set_yticklabels(y_tick_labels, fontsize=6)
    ax_h.set_xlabel(f"line1 x window ({label_suffix}; {x_order_label})")
    ax_h.set_ylabel(f"line2 y window ({label_suffix}; {y_order_label})")

    if args.draw_main_diagonal:
        diagonal_len = min(n1, n2)
        ax_h.plot(
            x_centers[:diagonal_len],
            y_centers[:diagonal_len],
            color="white",
            linewidth=1.0,
            alpha=0.85,
        )

    if args.cell_values:
        threshold = (float(args.vmin) + float(args.vmax)) / 2.0
        for i in range(n2):
            for j in range(n1):
                ax_h.text(
                    x_centers[j],
                    y_centers[i],
                    f"{sim[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=float(args.cell_value_fontsize),
                    color="black" if float(sim[i, j]) > threshold else "white",
                )

    ax_cbar = fig.add_subplot(gs[1, 2])
    cmap = plt.get_cmap(args.cmap)
    norm = plt.Normalize(float(args.vmin), float(args.vmax))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("cosine similarity")

    # Avoid bbox_inches='tight'; it can rescale axes independently and break the
    # thumbnail-to-cell alignment.
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Line1-to-line2 window cosine similarity heatmap.")
    parser.add_argument("--weights", required=True, help="Checkpoint path with image model weights.")
    parser.add_argument("--line1", default=None, help="Path to line1 image. Alternative to --data-dir/--index.")
    parser.add_argument("--line2", default=None, help="Path to line2 image. Alternative to --data-dir/--index.")
    parser.add_argument("--data-dir", default=None, help="Dataset root with images/ folder.")
    parser.add_argument("--index", type=int, default=None, help="1-based dataset sample index.")
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
    parser.add_argument("--display-order", choices=["model", "visual"], default="visual", help="Base order for axes/heatmap before optional axis-specific reversal.")
    parser.add_argument("--reverse-x-axis", action="store_true", help="Reverse the line1 top x-axis thumbnails and heatmap columns.")
    parser.add_argument("--reverse-y-axis", action="store_true", help="Reverse the line2 left y-axis thumbnails and heatmap rows.")
    parser.add_argument("--tick-labels", choices=["shown", "model"], default="model")
    parser.add_argument("--axis-slice-mode", choices=["nonoverlap", "window"], default="nonoverlap")
    parser.add_argument("--axis-cell-pixels", type=int, default=52)
    parser.add_argument("--window-gap-pixels", type=int, default=12)
    parser.add_argument("--x-strip-height", type=int, default=84)
    parser.add_argument("--y-strip-width", type=int, default=108)
    parser.add_argument("--mirror-axis-windows", action="store_true")
    parser.add_argument("--mirror-x-axis-windows", action="store_true")
    parser.add_argument("--mirror-y-axis-windows", action="store_true")
    parser.add_argument("--no-y-axis-rotate", dest="y_axis_rotate", action="store_false")
    parser.set_defaults(y_axis_rotate=True)
    parser.add_argument("--y-axis-flip", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cell-values", dest="cell_values", action="store_true", default=True, help="Write cosine value inside every cell.")
    parser.add_argument("--no-cell-values", dest="cell_values", action="store_false", help="Hide cosine values inside cells.")
    parser.add_argument("--cell-value-fontsize", type=float, default=4.0)
    parser.add_argument("--show-all-ticks", action="store_true")
    parser.add_argument("--draw-main-diagonal", action="store_true", help="Draw a simple shown-index diagonal as a rough positional reference.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figure-dpi-scale", type=float, default=90.0)
    parser.add_argument("--colorbar-width", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    image1_path, image2_path = infer_image_paths(args)

    loaded = torch.load(args.weights, map_location=args.device)
    cfg = checkpoint_config(loaded)
    if args.window_size is None:
        args.window_size = int(cfg.get("window_size", 32))
    if args.stride is None:
        args.stride = int(cfg.get("stride", max(1, args.window_size // 2)))
    if args.vector_size is None:
        args.vector_size = int(cfg.get("vector_size", 128))

    line1_img, line1_tensor = load_image(image1_path, args.height, args.width)
    line2_img, line2_tensor = load_image(image2_path, args.height, args.width)
    model = build_image_model(args, cfg, args.device)
    model.load_state_dict(extract_image_state(loaded), strict=False)

    line1_features, line2_features = get_window_embeddings_pair(
        model,
        line1_tensor,
        line2_tensor,
        args.device,
        args.feature_space,
    )
    sim = cosine_matrix(line1_features, line2_features)

    save_pair_similarity_figure(sim, line1_img, line2_img, args, args.output)
    print(f"saved line1-line2 window cosine heatmap: {args.output}")
    print(f"line1={image1_path}")
    print(f"line2={image2_path}")
    print(
        f"line1_windows={line1_features.shape[0]} line2_windows={line2_features.shape[0]} "
        f"feature_dim={line1_features.shape[1]} feature_space={args.feature_space}"
    )
    print(
        f"display_order={args.display_order} reverse_x_axis={args.reverse_x_axis} "
        f"reverse_y_axis={args.reverse_y_axis} tick_labels={args.tick_labels} "
        f"cell_values={args.cell_values}"
    )


if __name__ == "__main__":
    main()
