#!/usr/bin/env python3
"""Visualize Grad-CAM and aligned token/span for every image window.

Default use through run_visualize_window_feature_concentration.sh creates one PNG
per model window: original window crop, Grad-CAM heatmap, and early-fusion
overlay.  The token/span written on each PNG is inferred by hard Span-DTW.
"""
from __future__ import annotations

import argparse
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
except Exception:
    arabic_reshaper = None
    get_display = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def display_text(text: str) -> str:
    text = "" if text is None else str(text)
    if arabic_reshaper is not None and get_display is not None:
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            pass
    return text


def safe_token(text: str, max_len: int = 24) -> str:
    text = str(text or "empty").strip() or "empty"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\u0600-\u06FF]+", "", text)
    return text[:max_len] or "token"


def read_text(path: Optional[str], text: Optional[str]) -> str:
    if text is not None:
        return text
    if path is None:
        raise ValueError("Provide --text/--text-path with --line, or use --data-dir + --index.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def infer_paths(args) -> Tuple[str, str]:
    if args.line:
        if args.text_path is None and args.text is None:
            raise ValueError("When using --line, also provide --text or --text-path.")
        return args.line, args.text_path
    if args.data_dir is None or args.index is None:
        raise ValueError("Provide --line + --text/--text-path, or --data-dir + --index.")
    image_path = os.path.join(args.data_dir, "images", f"img{args.which_line}_{args.index}.png")
    text_path = os.path.join(args.data_dir, "texts", f"text{args.which_line}_{args.index}.txt")
    return image_path, text_path


def load_image(path: str, height: int, width: int) -> Tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BILINEAR)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return pil, tfm(pil).unsqueeze(0)


def checkpoint_config(loaded) -> dict:
    return dict(loaded.get("model_config") or {}) if isinstance(loaded, dict) else {}


def image_state(loaded):
    if isinstance(loaded, dict):
        return loaded.get("image_model_state_dict") or loaded.get("model_state_dict") or loaded
    return loaded


def build_image_model(args, cfg: dict, device: str) -> EmbeddingModel:
    model = EmbeddingModel(
        window_size=int(args.window_size or cfg.get("window_size", 32)),
        stride=int(args.stride or cfg.get("stride", 16)),
        vector_size=int(args.vector_size or cfg.get("vector_size", 128)),
        device=device,
        use_flip=bool(args.use_flip),
        use_bilstm=bool(cfg.get("use_bilstm", True)) and not args.no_bilstm,
        bilstm_layers=int(cfg.get("bilstm_layers", 1)),
        bilstm_hidden_dim=int(cfg.get("bilstm_hidden_dim", args.vector_size or cfg.get("vector_size", 128))),
    ).to(device)
    return model


def build_text_encoder(args, cfg: dict, loaded, device: str):
    enc_type = str(args.text_encoder_type or cfg.get("text_encoder_type", "arabic_span")).lower()
    dim = int(args.vector_size or cfg.get("vector_size", 128))
    model_name = args.arabic_text_model_name or cfg.get("arabic_text_model_name", "aubmindlab/bert-base-arabertv02")
    if enc_type == "arabic_span":
        enc = ArabicSpanTextEncoder(
            model_name=model_name,
            output_dim=dim,
            max_span_chars=int(args.max_span_chars or cfg.get("max_text_span_chars", 3)),
            freeze_backbone=True,
            device=device,
            strip_text_edges=bool(cfg.get("strip_span_text_edges", True)),
            cache_size=int(cfg.get("span_feature_cache_size", 2048)),
            cache_dtype=str(cfg.get("span_feature_cache_dtype", "float16")),
        )
    elif enc_type == "arabic_token":
        enc = ArabicTokenTextEncoder(
            model_name=model_name,
            output_dim=dim,
            max_token_chars=int(args.max_token_chars or cfg.get("max_text_token_chars", 3)),
            freeze_backbone=True,
            device=device,
        )
    elif enc_type == "char":
        enc = TextEmbedding(embedding_dim=dim).to(device)
        for p in enc.parameters():
            p.requires_grad_(False)
    else:
        raise ValueError(f"Unknown text encoder type: {enc_type}")
    if isinstance(loaded, dict):
        state = loaded.get("text_encoder_state_dict") or loaded.get("text_embedder_state_dict")
        if state:
            enc.load_state_dict(state, strict=False)
    enc.eval()
    return enc


def get_line_embeddings(model: EmbeddingModel, image_tensor: torch.Tensor, device: str):
    model.eval()
    with torch.no_grad():
        contextual, local, _ink = model(image_tensor.to(device), return_local=True, return_ink=True)
    return contextual[0].float(), local[0].float()


def aligned_tokens_for_windows(text_encoder, text: str, align_embeddings: torch.Tensor, args):
    encoding = text_encoder(text, use_cache=True)
    norm_img = F.normalize(align_embeddings.float(), p=2, dim=-1)
    path = hard_span_dtw_path(
        encoding,
        norm_img,
        temperature=float(args.temperature),
        max_windows=int(args.max_windows_per_span),
        window_count_penalty=float(args.window_count_penalty),
    )
    labels = ["?"] * int(norm_img.shape[0])
    span_indices = [-1] * int(norm_img.shape[0])
    for step in path:
        span_idx = int(step["span_idx"])
        token = str(step.get("text", encoding.texts[span_idx]))
        for w in range(int(step["window_start"]), int(step["window_end"])):
            if 0 <= w < len(labels):
                labels[w] = token
                span_indices[w] = span_idx
    return labels, span_indices, path, encoding


def window_pixel_range(i: int, n: int, args, width: int) -> Tuple[int, int]:
    visual_i = n - 1 - i if args.use_flip else i
    x0 = max(0, min(width, int(visual_i * args.stride)))
    x1 = max(0, min(width, int(x0 + args.window_size)))
    return x0, x1


def crop_window(pil_img: Image.Image, i: int, n: int, args):
    x0, x1 = window_pixel_range(i, n, args, pil_img.width)
    if x1 <= x0:
        x1 = min(pil_img.width, x0 + max(1, int(args.window_size)))
    return pil_img.crop((x0, 0, x1, pil_img.height)).convert("RGB"), x0, x1


def model_patches(image_tensor: torch.Tensor, args, device: str) -> torch.Tensor:
    patches = image_tensor.to(device).unfold(3, int(args.window_size), int(args.stride))
    patches = patches.permute(0, 3, 1, 2, 4).contiguous()
    if args.use_flip:
        patches = torch.flip(patches, dims=[1])
    return patches[0]


def resolve_layer(model: EmbeddingModel, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        options = ", ".join(k for k in modules if k.startswith("cnn_encoder.backbone"))
        raise ValueError(f"Unknown --gradcam-layer {name!r}. Backbone layers: {options}")
    return modules[name]


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx <= mn + 1e-8 else (x - mn) / (mx - mn + 1e-8)


def gradcam_for_patch(model: EmbeddingModel, patch: torch.Tensor, layer, target_vec, target_mode: str):
    holder = {}

    def hook(_module, _inp, out):
        holder["act"] = out
        out.retain_grad()

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        patch = patch.detach().clone().unsqueeze(0).to(model.device)
        patch.requires_grad_(True)
        feat = model.vision_norm(model.cnn_encoder(patch))
        norm_feat = F.normalize(feat.float(), p=2, dim=-1)
        actual_target = target_mode
        if target_mode == "token" and target_vec is not None:
            target = F.normalize(target_vec.detach().float().to(norm_feat.device).view(1, -1), p=2, dim=-1)
            score = torch.sum(norm_feat * target)
        elif target_mode == "top_feature":
            dim = int(torch.argmax(torch.abs(feat.detach()[0])).item())
            score = feat[0, dim]
            actual_target = f"top_feature:{dim}"
        else:
            score = torch.sum(feat.float() ** 2)
            actual_target = "energy"
        score.backward()
        act = holder.get("act")
        if act is None or act.grad is None:
            raise RuntimeError("Grad-CAM failed to capture activation/gradient. Try --gradcam-layer cnn_encoder.backbone.6")
        weights = act.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=patch.shape[-2:], mode="bilinear", align_corners=False)
        return normalize01(cam[0, 0].detach().cpu().numpy()), float(score.detach().cpu().item()), actual_target
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def colorize(cam: np.ndarray, width: int, height: int, cmap_name: str) -> Image.Image:
    rgb = (plt.get_cmap(cmap_name)(normalize01(cam))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((int(width), int(height)), Image.BILINEAR)


def blend(crop: Image.Image, heat: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(crop.convert("RGB"), heat.resize(crop.size, Image.BILINEAR).convert("RGB"), max(0.0, min(1.0, float(alpha))))


def save_per_window_gradcams(pil_img, labels, span_indices, encoding, patches, model, args, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    layer = resolve_layer(model, args.gradcam_layer)
    span_emb = getattr(encoding, "embeddings", None)
    for i in range(len(labels)):
        target_vec = None
        if args.gradcam_target == "token" and span_emb is not None and int(span_indices[i]) >= 0:
            target_vec = span_emb[int(span_indices[i])]
        cam, score, target_name = gradcam_for_patch(model, patches[i], layer, target_vec, args.gradcam_target)
        crop, x0, x1 = crop_window(pil_img, i, len(labels), args)
        w = max(int(args.per_window_image_width), crop.width * int(args.per_window_scale))
        h = int(args.per_window_image_height)
        crop = crop.resize((w, h), Image.BILINEAR)
        heat = colorize(cam, w, h, args.cmap)
        fused = blend(crop, heat, args.early_fusion_alpha)
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        axes = fig.subplots(1, 3)
        for ax, img, title in zip(
            axes,
            [crop, heat, fused],
            ["window image", f"Grad-CAM ({target_name})", f"early fusion alpha={args.early_fusion_alpha:.2f}"],
        ):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        token = display_text(labels[i])
        fig.suptitle(f"window {i:03d} | x={x0}:{x1} | token/span={token} | gradcam_score={score:.4f}", fontsize=11)
        fname = f"window_{i:03d}_x{x0:04d}_{x1:04d}_{safe_token(labels[i])}_gradcam.png"
        fig.savefig(Path(out_dir) / fname, dpi=args.per_window_dpi, bbox_inches="tight")
        plt.close(fig)


def save_summary(pil_img, labels, args, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img = np.asarray(pil_img)
    n, width, height = len(labels), pil_img.width, pil_img.height
    fig_width = max(16, n * 0.33)
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    ax.imshow(img)
    ax.set_xlim(0, width)
    ax.set_ylim(height + 28, -12)
    ax.axis("off")
    ax.set_title("Window tokens used for per-window Grad-CAM")
    for i, lab in enumerate(labels):
        x0, x1 = window_pixel_range(i, n, args, width)
        ax.add_patch(plt.Rectangle((x0, 0), max(1, x1 - x0), height, fill=False, linewidth=1.0))
        ax.text(0.5 * (x0 + x1), height + 8, display_text(lab), ha="center", va="top", fontsize=args.token_fontsize, rotation=90 if args.rotate_tokens else 0)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Create per-window Grad-CAM visualizations with aligned token/span labels.")
    p.add_argument("--weights", required=True)
    p.add_argument("--line", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--text-path", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--which-line", type=int, choices=[1, 2], default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--no-summary-image", action="store_true")
    p.add_argument("--save-per-window-images", action="store_true")
    p.add_argument("--per-window-output-dir", default=None)
    p.add_argument("--early-fusion-alpha", type=float, default=0.55)
    p.add_argument("--per-window-image-width", type=int, default=256)
    p.add_argument("--per-window-image-height", type=int, default=256)
    p.add_argument("--per-window-scale", type=int, default=8)
    p.add_argument("--per-window-dpi", type=int, default=160)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--vector-size", type=int, default=None)
    p.add_argument("--use-flip", action="store_true")
    p.add_argument("--no-bilstm", action="store_true")
    p.add_argument("--feature-space", choices=["local", "contextual"], default="local")  # compatibility
    p.add_argument("--alignment-space", choices=["local", "contextual"], default="contextual")
    p.add_argument("--text-encoder-type", choices=["arabic_span", "arabic_token", "char"], default=None)
    p.add_argument("--arabic-text-model-name", default=None)
    p.add_argument("--max-span-chars", type=int, default=None)
    p.add_argument("--max-token-chars", type=int, default=None)
    p.add_argument("--max-windows-per-span", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--window-count-penalty", type=float, default=0.01)
    p.add_argument("--gradcam-layer", default="cnn_encoder.backbone.7")
    p.add_argument("--gradcam-target", choices=["token", "energy", "top_feature"], default="token")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--token-fontsize", type=float, default=6.0)
    p.add_argument("--rotate-tokens", action="store_true", default=True)
    p.add_argument("--dpi", type=int, default=180)
    # Deprecated compatibility args accepted but ignored.
    p.add_argument("--csv-output", default=None)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--concentration-metric", default="topk_mass")
    p.add_argument("--concentration-top-k", type=int, default=8)
    p.add_argument("--feature-normalize", default="per_window")
    p.add_argument("--csv-top-features", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    image_path, text_path = infer_paths(args)
    text = read_text(text_path, args.text)
    loaded = torch.load(args.weights, map_location=args.device)
    cfg = checkpoint_config(loaded)
    args.window_size = int(args.window_size or cfg.get("window_size", 32))
    args.stride = int(args.stride or cfg.get("stride", max(1, args.window_size // 2)))
    args.vector_size = int(args.vector_size or cfg.get("vector_size", 128))
    args.max_span_chars = int(args.max_span_chars or cfg.get("max_text_span_chars", 3))
    args.max_token_chars = int(args.max_token_chars or cfg.get("max_text_token_chars", 3))
    if "contrastive_temperature" in cfg and args.temperature == 0.07:
        args.temperature = float(cfg.get("contrastive_temperature", 0.07))

    model = build_image_model(args, cfg, args.device)
    model.load_state_dict(image_state(loaded), strict=False)
    model.eval()
    text_encoder = build_text_encoder(args, cfg, loaded, args.device)

    pil_img, image_tensor = load_image(image_path, args.height, args.width)
    contextual, local = get_line_embeddings(model, image_tensor, args.device)
    align_features = contextual if args.alignment_space == "contextual" else local
    labels, span_indices, _path, encoding = aligned_tokens_for_windows(text_encoder, text, align_features, args)
    patches = model_patches(image_tensor, args, args.device)

    if not args.no_summary_image:
        save_summary(pil_img, labels, args, args.output)
        print(f"saved summary figure: {args.output}")

    if args.save_per_window_images:
        out_dir = args.per_window_output_dir or str(Path(args.output).with_suffix("")) + "_gradcam_windows"
        save_per_window_gradcams(pil_img, labels, span_indices, encoding, patches, model, args, out_dir)
        print(f"saved per-window Grad-CAM images: {out_dir}")

    print(f"windows={len(labels)} gradcam_layer={args.gradcam_layer} target={args.gradcam_target} text_length={len(text)}", flush=True)


if __name__ == "__main__":
    main()
