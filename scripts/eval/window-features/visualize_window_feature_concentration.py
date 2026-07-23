#!/usr/bin/env python3
"""Truthful per-window Grad-CAM labels for Arabic span alignment."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import visualize_window_feature_concentration_legacy as legacy  # noqa: E402
from embeddingModel import EmbeddingModel  # noqa: E402
from span_alignment_loss import hard_span_dtw_path  # noqa: E402


def build_model(args, config, device):
    default_flip = bool(
        config.get("use_flip", str(config.get("lang", "")).lower() == "arabic")
    )
    requested_flip = default_flip if args.use_flip is None else bool(args.use_flip)
    model = EmbeddingModel(
        window_size=int(args.window_size or config.get("window_size", 32)),
        stride=int(args.stride or config.get("stride", 16)),
        vector_size=int(args.vector_size or config.get("vector_size", 128)),
        device=device,
        use_flip=requested_flip,
        use_bilstm=bool(config.get("use_bilstm", True)) and not args.no_bilstm,
        bilstm_layers=int(config.get("bilstm_layers", 1)),
        bilstm_hidden_dim=int(
            config.get(
                "bilstm_hidden_dim",
                args.vector_size or config.get("vector_size", 128),
            )
        ),
        use_local_grouping=bool(
            config.get("use_local_window_grouping", True)
        ),
        local_group_size=int(config.get("local_group_size", 3)),
    ).to(device)
    return model


def line_embeddings(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        contextual, local, grouped, ink = model(
            image_tensor.to(device),
            return_local=True,
            return_grouped=True,
            return_ink=True,
        )
    return contextual[0].float(), local[0].float(), grouped[0].float(), ink[0].float()


def aligned_regions(text_encoder, text, features, args):
    encoding = text_encoder(text, use_cache=True)
    path = hard_span_dtw_path(
        encoding,
        F.normalize(features.float(), p=2, dim=-1),
        temperature=float(args.temperature),
        max_windows=int(args.max_windows_per_span),
        window_count_penalty=float(args.window_count_penalty),
    )
    labels = ["?"] * int(features.shape[0])
    indices = [-1] * int(features.shape[0])
    for step in path:
        span_index = int(step["span_idx"])
        for window in range(int(step["window_start"]), int(step["window_end"])):
            labels[window] = str(step["text"])
            indices[window] = span_index
    return labels, indices, path, encoding


def candidate_indices(encoding, step, max_chars):
    candidates = []
    spaces = getattr(encoding, "is_space", None)
    for index, (start, length) in enumerate(
        zip(encoding.starts, encoding.lengths)
    ):
        end = int(start) + int(length)
        if int(start) < int(step["text_start"]) or end > int(step["text_end"]):
            continue
        if int(length) > int(max_chars):
            continue
        if spaces is not None and bool(spaces[index]):
            continue
        candidates.append(index)
    if not candidates:
        candidates.append(int(step["span_idx"]))
    return candidates


def local_core_labels(encoding, path, local_features, ink_ratio, args):
    count = int(local_features.shape[0])
    labels = ["?"] * count
    indices = [-1] * count
    scores = [float("nan")] * count
    core = F.normalize(encoding.embeddings.float(), p=2, dim=-1)
    local = F.normalize(local_features.float(), p=2, dim=-1)

    for step in path:
        candidates = candidate_indices(
            encoding, step, args.local_label_max_chars
        )
        candidate_tensor = torch.as_tensor(candidates, device=local.device)
        candidate_vectors = core.index_select(0, candidate_tensor)
        lengths = local.new_tensor(
            [encoding.lengths[index] for index in candidates]
        )
        for window in range(int(step["window_start"]), int(step["window_end"])):
            if float(ink_ratio[window]) < float(args.blank_ink_threshold):
                labels[window] = "<BLANK>"
                continue
            similarities = torch.mv(candidate_vectors, local[window])
            adjusted = similarities - float(args.local_label_length_penalty) * (
                lengths - 1.0
            )
            best = int(torch.argmax(adjusted).item())
            span_index = candidates[best]
            labels[window] = str(encoding.texts[span_index])
            indices[window] = span_index
            scores[window] = float(similarities[best].item())
    return labels, indices, scores


def save_windows(
    pil_image,
    local_labels,
    local_indices,
    local_scores,
    region_labels,
    encoding,
    patches,
    model,
    ink_ratio,
    args,
    output_dir,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    layer = legacy.resolve_layer(model, args.gradcam_layer)
    core_embeddings = getattr(encoding, "embeddings", None)
    for index, local_label in enumerate(local_labels):
        span_index = int(local_indices[index])
        target = None
        if (
            args.gradcam_target == "token"
            and core_embeddings is not None
            and span_index >= 0
        ):
            target = core_embeddings[span_index]
        cam, gradcam_score, target_name = legacy.gradcam_for_patch(
            model,
            patches[index],
            layer,
            target,
            args.gradcam_target,
        )
        crop, x0, x1 = legacy.crop_window(
            pil_image, index, len(local_labels), args
        )
        width = max(
            int(args.per_window_image_width),
            crop.width * int(args.per_window_scale),
        )
        height = int(args.per_window_image_height)
        crop = crop.resize((width, height))
        heat = legacy.colorize(cam, width, height, args.cmap)
        fused = legacy.blend(crop, heat, args.early_fusion_alpha)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for axis, image, title in zip(
            axes,
            (crop, heat, fused),
            (
                "window image",
                f"Grad-CAM ({target_name})",
                f"fusion alpha={args.early_fusion_alpha:.2f}",
            ),
        ):
            axis.imshow(image)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        local_score = local_scores[index]
        figure.suptitle(
            f"window {index:03d} | x={x0}:{x1} | "
            f"local/core={legacy.display_text(local_label)} | "
            f"region/core={legacy.display_text(region_labels[index])} | "
            f"ink={float(ink_ratio[index]):.4f} | "
            f"local_sim={local_score:.4f} | gradcam={gradcam_score:.4f}",
            fontsize=10,
        )
        filename = (
            f"window_{index:03d}_x{x0:04d}_{x1:04d}_"
            f"{legacy.safe_token(local_label)}_gradcam.png"
        )
        figure.savefig(
            str(Path(output_dir) / filename),
            dpi=args.per_window_dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def save_summary(pil_image, local_labels, region_labels, ink_ratio, args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    count = len(local_labels)
    width, height = pil_image.width, pil_image.height
    figure, axis = plt.subplots(figsize=(max(16, count * 0.36), 4.0))
    axis.imshow(np.asarray(pil_image))
    axis.set_xlim(0, width)
    axis.set_ylim(height + 52, -12)
    axis.axis("off")
    axis.set_title("Local/core label (dark) and global region/core label (light)")
    for index, local_label in enumerate(local_labels):
        x0, x1 = legacy.window_pixel_range(index, count, args, width)
        axis.add_patch(
            plt.Rectangle((x0, 0), max(1, x1 - x0), height, fill=False, linewidth=1)
        )
        center = 0.5 * (x0 + x1)
        axis.text(
            center,
            height + 7,
            legacy.display_text(local_label),
            ha="center",
            va="top",
            fontsize=args.token_fontsize,
            rotation=90,
        )
        axis.text(
            center,
            height + 32,
            legacy.display_text(region_labels[index]),
            ha="center",
            va="top",
            fontsize=max(4, args.token_fontsize - 1),
            rotation=90,
            alpha=0.5,
        )
        if float(ink_ratio[index]) < float(args.blank_ink_threshold):
            axis.add_patch(
                plt.Rectangle(
                    (x0, 0), max(1, x1 - x0), height, fill=True, alpha=0.08
                )
            )
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--line")
    parser.add_argument("--text")
    parser.add_argument("--text-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--index", type=int)
    parser.add_argument("--which-line", type=int, choices=[1, 2], default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-summary-image", action="store_true")
    parser.add_argument("--save-per-window-images", action="store_true")
    parser.add_argument("--per-window-output-dir")
    parser.add_argument("--early-fusion-alpha", type=float, default=0.55)
    parser.add_argument("--per-window-image-width", type=int, default=256)
    parser.add_argument("--per-window-image-height", type=int, default=256)
    parser.add_argument("--per-window-scale", type=int, default=8)
    parser.add_argument("--per-window-dpi", type=int, default=160)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--vector-size", type=int)
    parser.add_argument(
        "--use-flip", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--no-bilstm", action="store_true")
    parser.add_argument(
        "--feature-space",
        choices=["local", "grouped", "contextual"],
        default="local",
    )
    parser.add_argument(
        "--alignment-space",
        choices=["local", "grouped", "contextual"],
        default="contextual",
    )
    parser.add_argument("--text-encoder-type")
    parser.add_argument("--arabic-text-model-name")
    parser.add_argument("--max-span-chars", type=int)
    parser.add_argument("--max-token-chars", type=int)
    parser.add_argument("--max-windows-per-span", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--window-count-penalty", type=float, default=0.05)
    parser.add_argument("--local-label-max-chars", type=int, default=2)
    parser.add_argument("--local-label-length-penalty", type=float, default=0.05)
    parser.add_argument("--blank-ink-threshold", type=float, default=0.005)
    parser.add_argument("--gradcam-layer", default="cnn_encoder.backbone.4")
    parser.add_argument("--gradcam-target", default="token")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--token-fontsize", type=float, default=6)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--rotate-tokens", action="store_true", default=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--concentration-metric")
    parser.add_argument("--concentration-top-k", type=int)
    parser.add_argument("--feature-normalize")
    parser.add_argument("--csv-top-features", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path, text_path = legacy.infer_paths(args)
    text = legacy.read_text(text_path, args.text)
    loaded = torch.load(args.weights, map_location=args.device)
    config = legacy.checkpoint_config(loaded)
    args.window_size = int(args.window_size or config.get("window_size", 32))
    args.stride = int(
        args.stride or config.get("stride", max(1, args.window_size // 2))
    )
    args.vector_size = int(args.vector_size or config.get("vector_size", 128))
    args.max_span_chars = int(
        args.max_span_chars or config.get("max_text_span_chars", 3)
    )
    args.max_token_chars = int(
        args.max_token_chars or config.get("max_text_token_chars", 3)
    )

    model = build_model(args, config, args.device)
    model.load_state_dict(legacy.image_state(loaded), strict=False)
    model.eval()
    args.resolved_use_flip = bool(model.use_flip)
    text_encoder = legacy.build_text_encoder(
        args, config, loaded, args.device
    )
    image, tensor = legacy.load_image(
        image_path, args.height, args.width
    )
    contextual, local, grouped, ink = line_embeddings(
        model, tensor, args.device
    )
    spaces = {"local": local, "grouped": grouped, "contextual": contextual}
    region_labels, _, path, encoding = aligned_regions(
        text_encoder, text, spaces[args.alignment_space], args
    )
    local_labels, local_indices, local_scores = local_core_labels(
        encoding, path, spaces[args.feature_space], ink, args
    )
    patches = legacy.model_patches(tensor, args, args.device)

    if not args.no_summary_image:
        save_summary(image, local_labels, region_labels, ink, args)
        print(f"saved summary figure: {args.output}")
    if args.save_per_window_images:
        output_dir = args.per_window_output_dir or (
            str(Path(args.output).with_suffix("")) + "_gradcam_windows"
        )
        save_windows(
            image,
            local_labels,
            local_indices,
            local_scores,
            region_labels,
            encoding,
            patches,
            model,
            ink,
            args,
            output_dir,
        )
        print(f"saved per-window Grad-CAM images: {output_dir}")


if __name__ == "__main__":
    main()
