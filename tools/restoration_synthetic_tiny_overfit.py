#!/usr/bin/env python3
"""Tiny-overfit eight REAL synthetic windows.

This complements the controlled point-12 test. It takes an actual synthetic
manuscript line, finds the most informative 8-window region, and asks the local
CNN+decoder to memorize those exact RGB windows. This isolates whether the local
bottleneck can represent real training-style windows before DTW/context are
involved.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restoration_recommended_components import denormalize_imagenet_windows
from restoration_window_seq2seq import (
    WindowSequenceCNNEncoder,
    WindowSequenceStrokeDecoder,
)
from zero_shot_preprocessing import ManuscriptLinePreprocessor

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WINDOW_COUNT = 8
WINDOW_SIZE = 32
STRIDE = 16
SEGMENT_WIDTH = WINDOW_SIZE + (WINDOW_COUNT - 1) * STRIDE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def resolve_image(dataset: Path, index: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = dataset / "images" / f"img1_{index}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"img1_{index} not found under {dataset / 'images'}")


def tensor_image(x):
    x = x.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def choose_segment(rgb_tensor: torch.Tensor):
    """Pick the 8-window segment with greatest foreground contrast."""
    gray = rgb_tensor.mean(dim=1, keepdim=True)
    darkness = (1.0 - gray).clamp_min(0.0)
    width = int(rgb_tensor.shape[-1])
    best_x = 0
    best_score = -1.0
    for x0 in range(0, width - SEGMENT_WIDTH + 1, STRIDE):
        score = float(darkness[..., x0:x0 + SEGMENT_WIDTH].mean())
        if score > best_score:
            best_score = score
            best_x = x0
    return best_x, best_score


def save_sheet(path: Path, target, restored, swapped):
    patch_w, patch_h = 32 * 3, 128 * 3
    label_h = 26
    canvas = Image.new("RGB", (WINDOW_COUNT * patch_w, 3 * (patch_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, tensor) in enumerate(
        [("target", target), ("restored", restored), ("swap", swapped)]
    ):
        for i in range(WINDOW_COUNT):
            y0 = row * (patch_h + label_h)
            draw.text((i * patch_w + 2, y0 + 3), f"{name} W{i}", fill="black")
            canvas.paste(
                tensor_image(tensor[0, i]).resize((patch_w, patch_h)),
                (i * patch_w, y0 + label_h),
            )
    canvas.save(path)


def main():
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset).expanduser().resolve()
    image_path = resolve_image(dataset, args.index)

    source = Image.open(image_path).convert("RGB")
    processor = ManuscriptLinePreprocessor(
        size=(128, 1024),
        training=False,
        augment=False,
        binarize=False,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        autocontrast=False,
    )
    prepared, geometry = processor.preprocess_with_metadata(source)
    rgb = transforms.ToTensor()(prepared).unsqueeze(0)
    x0, darkness = choose_segment(rgb)
    segment_rgb = rgb[..., x0:x0 + SEGMENT_WIDTH].contiguous()
    segment = transforms.Normalize(MEAN, STD)(segment_rgb[0]).unsqueeze(0)

    prepared.save(output / "01_full_prepared_synthetic_line.png")
    tensor_image(segment_rgb[0]).save(output / "02_selected_real_8window_region.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment = segment.to(device)
    encoder = WindowSequenceCNNEncoder(
        input_height=128,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        embed_dim=32,
        base_channels=16,
    ).to(device)
    decoder = WindowSequenceStrokeDecoder(
        dim=32,
        output_height=128,
        output_width=WINDOW_SIZE,
        channels=64,
    ).to(device)

    normalized_windows = encoder.extract_windows(segment)
    target = denormalize_imagenet_windows(normalized_windows)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-3,
        weight_decay=0.0,
    )
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        local = encoder(segment).squeeze(2).transpose(1, 2).contiguous()
        restored = decoder(local)
        loss = F.l1_loss(restored, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    encoder.eval()
    decoder.eval()
    with torch.inference_mode():
        local = encoder(segment).squeeze(2).transpose(1, 2).contiguous()
        restored = decoder(local)
        swapped_local = local.clone()
        swapped_local[:, 0] = local[:, -1]
        swapped_local[:, -1] = local[:, 0]
        swapped = decoder(swapped_local)

    output_diversity = float(restored.std(dim=1).mean())
    target_diversity = float(target.std(dim=1).mean())
    swap_delta = float((restored - swapped).abs().mean())
    final_ratio = losses[-1] / max(losses[0], 1e-12)

    save_sheet(output / "03_real_targets_restored_swapped.png", target, restored, swapped)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(losses)), losses)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("L1 loss")
    ax.set_title("Tiny overfit on 8 real synthetic windows")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "04_loss_curve.png", dpi=170)
    plt.close(fig)
    np.savetxt(output / "losses.csv", np.asarray(losses), delimiter=",")

    passed = (
        losses[-1] < losses[0] * 0.75
        and output_diversity > 1e-4
        and swap_delta > 1e-4
    )
    summary = f"""POINT 12 — REAL SYNTHETIC TINY OVERFIT
======================================

image              : {image_path}
selected x-range   : {x0}:{x0 + SEGMENT_WIDTH}
segment darkness   : {darkness:.6f}
device             : {device}

initial L1         : {losses[0]:.8f}
final L1           : {losses[-1]:.8f}
final/initial      : {final_ratio:.6f}
target diversity   : {target_diversity:.8f}
output diversity   : {output_diversity:.8f}
feature swap delta : {swap_delta:.8f}

status             : {'PASS' if passed else 'FAIL'}

Interpretation:
- If this FAILS, the local encoder/decoder itself cannot memorize 8 actual
  synthetic windows. Do not blame DTW/context yet.
- If this PASSES but the trained Stage-A checkpoint reconstructs poorly, the
  issue is training/pretraining configuration rather than representational capacity.
- If Stage A is good but Stage B becomes poor, alignment training is destroying
  the restoration representation.
"""
    (output / "summary.txt").write_text(summary, encoding="utf-8")
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "selected_x0": int(x0),
                "selected_x1": int(x0 + SEGMENT_WIDTH),
                "initial_l1": float(losses[0]),
                "final_l1": float(losses[-1]),
                "loss_ratio": float(final_ratio),
                "target_diversity": float(target_diversity),
                "output_diversity": float(output_diversity),
                "swap_delta": float(swap_delta),
                "passed": bool(passed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
