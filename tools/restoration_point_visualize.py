#!/usr/bin/env python3
"""Save human-readable diagnostics for restoration recommendation points.

This tool is called by the no-flag shell wrappers in scripts/restoration_points/.
It writes PNG/JSON/TXT artifacts under:
    Results/Diagnostics/restoration_points/point_XX/

The visual artifacts are diagnostic proofs of each architectural property. They
are not all quality metrics: unless a point explicitly loads/trains weights,
model-dependent pictures are labeled as structural/untrained diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embeddingModel import EmbeddingModel, sliding_window
from restoration_epoch_probe import _hard_dtw
from restoration_recommended_components import (
    LocalContextFusion,
    contrastive_margin_from_costs,
    denormalize_imagenet_windows,
)
from restoration_window_seq2seq import (
    WindowSequenceCNNEncoder,
    WindowSequenceStrokeDecoder,
)
from vlm_restoration_positive_dtw import (
    attach_restoration_dtw_stages,
    stroke_restoration_loss,
)
from zero_shot_preprocessing import ManuscriptLinePreprocessor

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def normalize(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - MEAN.to(rgb)) / STD.to(rgb)


def tensor_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_contact_sheet(rows, path: Path, titles=None):
    """rows: list[list[tensor/PIL]], all images same-ish size."""
    converted = []
    for row in rows:
        converted.append([
            img if isinstance(img, Image.Image) else tensor_image(img)
            for img in row
        ])
    max_h = max(img.height for row in converted for img in row)
    max_w = max(img.width for row in converted for img in row)
    cols = max(len(row) for row in converted)
    label_h = 28 if titles else 0
    canvas = Image.new("RGB", (cols * max_w, len(converted) * (max_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(converted):
        for c, img in enumerate(row):
            y = r * (max_h + label_h) + label_h
            canvas.paste(img, (c * max_w, y))
            if titles and r < len(titles) and c < len(titles[r]):
                draw.text((c * max_w + 4, r * (max_h + label_h) + 4), titles[r][c], fill="black")
    canvas.save(path)


def save_matrix(matrix, path: Path, title: str, path_cells=None, xlabel=None, ylabel=None):
    arr = np.asarray(matrix, dtype=np.float32)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, aspect="auto")
    fig.colorbar(im, ax=ax)
    if path_cells:
        ys = [p[0] for p in path_cells]
        xs = [p[1] for p in path_cells]
        ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(out: Path, text: str):
    (out / "summary.txt").write_text(text.rstrip() + "\n", encoding="utf-8")


def recommended_model(vector_size=32):
    model = EmbeddingModel(
        window_size=32,
        stride=16,
        vector_size=vector_size,
        device="cpu",
        use_flip=False,
        input_height=128,
        vit_layers=1,
        vit_heads=4,
        vit_mlp_dim=max(64, vector_size * 2),
        vit_dropout=0.0,
        vit_max_tokens=64,
        vit_position_base_tokens=7,
        vit_binarize_input=False,
    )
    config = SimpleNamespace(
        restoration_decoder_channels=64,
        restoration_contrast_scale=0.15,
        restoration_semantic_adapter="identity",
        restoration_local_encoder="cnn_seq2seq",
        restoration_training_stage="align",
    )
    return attach_restoration_dtw_stages(model, config).eval()


def structured_line(width=96, variant=0):
    x = torch.ones(1, 3, 128, width)
    starts = [8, 32, 58] if variant == 0 else [12, 39, 61]
    ys = [35, 64, 87] if variant == 0 else [40, 58, 92]
    for i, (sx, sy) in enumerate(zip(starts, ys)):
        ex = min(width, sx + 22)
        x[:, :, sy:sy+5, sx:ex] = 0.08 + 0.05 * i
        dy = max(1, sy - 11)
        dx = min(width - 4, sx + 8)
        x[:, :, dy:dy+3, dx:dx+3] = 0.02
    return x


def point01(out: Path):
    arr = np.full((80, 300, 3), 255, dtype=np.uint8)
    arr[18:55, 20:82] = np.asarray([180, 20, 20], dtype=np.uint8)
    arr[8:12, 76:80] = np.asarray([25, 25, 25], dtype=np.uint8)
    arr[25:62, 220:282] = np.asarray([20, 45, 185], dtype=np.uint8)
    source = Image.fromarray(arr, mode="RGB")
    processor = ManuscriptLinePreprocessor(
        size=(128, 256), training=False, augment=False, binarize=False,
        preserve_aspect=True, crop_foreground=True,
        target_ink_height_ratio=0.72, autocontrast=False,
    )
    processed, meta = processor.preprocess_with_metadata(source)
    boxed = source.copy()
    draw = ImageDraw.Draw(boxed)
    box = (int(meta["crop_left"]), int(meta["crop_top"]), int(meta["crop_right"]) - 1, int(meta["crop_bottom"]) - 1)
    draw.rectangle(box, outline="red", width=2)
    source.save(out / "01_original_line.png")
    boxed.save(out / "02_detected_crop_box.png")
    processed.save(out / "03_cropped_resized_rgb.png")
    (out / "geometry.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    write_summary(out, f"""Point 01 visual result
Crop box: {box}
Processed size: {processed.size}
Internal blank gap remains inside one crop rectangle.
Inspect 02_detected_crop_box.png to make sure the small upper dot is retained.
Inspect 03_cropped_resized_rgb.png to confirm RGB/color information survives.
Full metadata is in geometry.json.
""")


def point02(out: Path):
    torch.manual_seed(2)
    model = recommended_model()
    rgb = structured_line(64, 0)
    image = normalize(rgb)
    with torch.no_grad():
        bundle = model(image, return_training_bundle=True)
    expected = denormalize_imagenet_windows(sliding_window(image, 32, 16))
    target = bundle["restoration_target"]
    recon = bundle["restoration"]
    diff = float((target - expected).abs().max())
    tensor_image(rgb).save(out / "01_model_input_line.png")
    save_contact_sheet(
        [[target[0, i], recon[0, i]] for i in range(target.shape[1])],
        out / "02_target_vs_current_reconstruction.png",
        [[f"target w{i}", f"reconstruction w{i}"] for i in range(target.shape[1])],
    )
    write_summary(out, f"""Point 02 visual result
Max absolute difference between restoration_target and exact de-normalized input windows: {diff:.8g}
The left image in each row of 02_target_vs_current_reconstruction.png is the exact RGB target.
The right image is the current decoder output. This diagnostic does NOT claim reconstruction quality unless trained weights are loaded.
""")


def point04(out: Path):
    torch.manual_seed(4)
    enc = WindowSequenceCNNEncoder(input_height=128, window_size=32, stride=16, embed_dim=32, base_channels=16).eval()
    line = structured_line(64, 0)
    patch = enc.extract_windows(line)[0, 0].unsqueeze(0)
    shapes = [("input", int(patch.shape[-2]), int(patch.shape[-1]))]
    x = patch
    with torch.no_grad():
        for i, block in enumerate(enc.encoder):
            x = block(x)
            shapes.append((f"block_{i}", int(x.shape[-2]), int(x.shape[-1])))
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    stages = np.arange(len(shapes))
    ax.plot(stages, [s[1] for s in shapes], marker="o", label="height")
    ax.plot(stages, [s[2] for s in shapes], marker="o", label="width")
    ax.set_xticks(stages)
    ax.set_xticklabels([s[0] for s in shapes], rotation=30)
    ax.set_ylabel("pixels")
    ax.set_title("Spatial resolution through local CNN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_spatial_resolution.png", dpi=160)
    plt.close(fig)
    save_matrix(x[0].mean(0).cpu().numpy(), out / "02_final_8x8_feature_map.png", "Mean final local feature map")
    (out / "spatial_shapes.json").write_text(json.dumps([{"stage":a,"height":b,"width":c} for a,b,c in shapes], indent=2), encoding="utf-8")
    write_summary(out, "Point 04 visual result\n" + "\n".join(f"{a}: {b}x{c}" for a,b,c in shapes) + "\nWidth is deliberately preserved in the first two downsampling stages.")


def point05(out: Path):
    torch.manual_seed(5)
    dec = WindowSequenceStrokeDecoder(dim=32, output_height=128, output_width=32, channels=64).eval()
    tokens = torch.randn(1, 2, 32)
    with torch.no_grad():
        original = dec(tokens)
        swapped = dec(tokens.flip(1))
    delta = float((original - swapped).abs().mean())
    save_contact_sheet(
        [[original[0,0], original[0,1]], [swapped[0,0], swapped[0,1]]],
        out / "01_feature_swap_reconstruction.png",
        [["original token A", "original token B"], ["after swap slot 0", "after swap slot 1"]],
    )
    write_summary(out, f"""Point 05 visual result
Mean absolute reconstruction change after swapping encoded vectors: {delta:.8f}
In 01_feature_swap_reconstruction.png, the second row should exchange the first row's decoder outputs.
This is a structural dependency check, not a trained reconstruction-quality claim.
""")


def point06(out: Path):
    torch.manual_seed(6)
    model = recommended_model()
    vit = model.vit_encoder
    local = torch.randn(1, 5, vit.embed_dim)
    valid = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        first = vit.encoder(local + vit._position_tokens(5), src_key_padding_mask=~valid)
        changed = local.clone()
        changed[:, 0] += 25.0
        second = vit.encoder(changed + vit._position_tokens(5), src_key_padding_mask=~valid)
    delta = torch.linalg.vector_norm((second-first).float(), dim=-1)[0].cpu().numpy()
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(delta)), delta)
    ax.set_xlabel("window token")
    ax.set_ylabel("context-vector L2 change")
    ax.set_title("Changing window 0 changes contextual representations")
    fig.tight_layout()
    fig.savefig(out / "01_neighbor_influence.png", dpi=160)
    plt.close(fig)
    np.savetxt(out / "context_change_per_token.csv", delta, delimiter=",")
    write_summary(out, f"Point 06 visual result\nChanged local token: 0\nContext-vector L2 change per token: {delta.tolist()}\nNon-zero change on other tokens demonstrates sequence context.")


def point07(out: Path):
    torch.manual_seed(7)
    fusion = LocalContextFusion(32).eval()
    local = torch.randn(1, 6, 32)
    context = torch.randn(1, 6, 32)
    with torch.no_grad():
        fused = fusion(local, context)
        local_changed = fusion(local + 2.0, context)
        context_changed = fusion(local, context - 2.0)
    local_effect = torch.linalg.vector_norm((fused-local_changed).float(), dim=-1)[0].cpu().numpy()
    context_effect = torch.linalg.vector_norm((fused-context_changed).float(), dim=-1)[0].cpu().numpy()
    norms = torch.linalg.vector_norm(fused.float(), dim=-1)[0].cpu().numpy()
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    idx = np.arange(len(norms))
    width = 0.35
    ax.bar(idx-width/2, local_effect, width, label="change local")
    ax.bar(idx+width/2, context_effect, width, label="change context")
    ax.set_xlabel("window")
    ax.set_ylabel("fused-vector L2 change")
    ax.set_title("Fused representation depends on both inputs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_fusion_dependency.png", dpi=160)
    plt.close(fig)
    write_summary(out, f"Point 07 visual result\nFused vector norms: {norms.tolist()}\nEffect of changing local: {local_effect.tolist()}\nEffect of changing context: {context_effect.tolist()}\nNorms should be approximately 1.0 and both effects should be non-zero.")


def point08(out: Path):
    positive = torch.tensor(0.55, requires_grad=True)
    negatives = [torch.tensor(0.25), torch.tensor(0.35)]
    contrastive = contrastive_margin_from_costs(positive, negatives, margin=0.20)
    contrastive.backward()
    grad = float(positive.grad)
    settings = SimpleNamespace(restoration_pixel_weight=1.0, restoration_edge_weight=0.5, restoration_structure_weight=0.25)
    target = structured_line(64, 0).unfold(3, 32, 16).permute(0,3,1,2,4).contiguous()
    faithful = target.clone()
    collapsed = target[:, :1].repeat(1, target.shape[1], 1, 1, 1)
    valid = torch.ones(target.shape[0], target.shape[1], 1, target.shape[-2], target.shape[-1])
    faithful_loss, *_ = stroke_restoration_loss(settings, faithful, target, valid)
    collapsed_loss, *_ = stroke_restoration_loss(settings, collapsed, target, valid)
    vals = [float(faithful_loss), float(collapsed_loss), float(contrastive.detach())]
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.bar(["faithful recon", "collapsed recon", "contrastive"], vals)
    ax.set_ylabel("loss")
    ax.set_title("Reconstruction and contrastive training signals")
    fig.tight_layout()
    fig.savefig(out / "01_loss_signals.png", dpi=160)
    plt.close(fig)
    write_summary(out, f"""Point 08 visual result
Faithful reconstruction loss: {vals[0]:.8f}
Collapsed reconstruction loss: {vals[1]:.8f}
Contrastive margin loss: {vals[2]:.8f}
Gradient on positive DTW cost from contrastive term: {grad:.8f}
The collapsed reconstruction should cost more than the faithful reconstruction.
""")


def point09(out: Path):
    a = np.zeros((4,1), dtype=np.float32)
    b = np.zeros((1,4), dtype=np.float32)
    pa, ca = _hard_dtw(a, 0.05)
    pb, cb = _hard_dtw(b, 0.05)
    save_matrix(a, out / "01_many_windows_to_one_letter.png", "4 windows -> 1 letter", pa, "letter", "window")
    save_matrix(b, out / "02_one_window_to_many_letters.png", "1 window -> 4 letters", pb, "letter", "window")
    write_summary(out, f"Point 09 visual result\nMany-windows->one-letter path: {pa}; cost={ca}\nOne-window->many-letters path: {pb}; cost={cb}\nInspect both route overlays to verify vertical and horizontal repetitions are representable.")


def point10(out: Path):
    first = np.asarray([[0.0,9.0],[0.0,9.0],[9.0,0.0]], dtype=np.float32)
    second = np.asarray([[0.0,9.0],[9.0,0.0],[9.0,0.0]], dtype=np.float32)
    p1, c1 = _hard_dtw(first, 0.05)
    p2, c2 = _hard_dtw(second, 0.05)
    save_matrix(first, out / "01_cost_matrix_A_route.png", "Cost matrix A and recomputed route", p1, "text", "window")
    save_matrix(second, out / "02_cost_matrix_B_route.png", "Cost matrix B and recomputed route", p2, "text", "window")
    write_summary(out, f"Point 10 visual result\nRoute A: {p1}; cost={c1}\nRoute B: {p2}; cost={c2}\nRoutes differ: {p1 != p2}\nThis shows the diagnostic route follows the current cost matrix rather than a fixed epoch route.")


def point11(out: Path):
    torch.manual_seed(11)
    enc = WindowSequenceCNNEncoder(input_height=128, window_size=32, stride=16, embed_dim=32, base_channels=16).eval()
    line = structured_line(64, 0)
    with torch.no_grad():
        vectors = enc.spatial_vectors(line, vectors_per_window=4)
    v = vectors[0,0]
    sim = (v @ v.T).cpu().numpy()
    save_matrix(sim, out / "01_four_spatial_vectors_cosine.png", "K=4 vectors within first physical window", xlabel="spatial vector", ylabel="spatial vector")
    norms = torch.linalg.vector_norm(v, dim=-1).cpu().numpy()
    write_summary(out, f"Point 11 visual result\nReturned tensor shape: {tuple(vectors.shape)}\nNorms for first window's K=4 vectors: {norms.tolist()}\n01_four_spatial_vectors_cosine.png shows whether the optional sub-window vectors carry different information.")


def point13(out: Path):
    torch.manual_seed(13)
    model = recommended_model()
    line1_rgb = structured_line(96, 0)
    line2_rgb = structured_line(96, 1)
    line1 = normalize(line1_rgb)
    line2 = normalize(line2_rgb)
    with torch.no_grad():
        fused1, local1, valid1 = model(line1, return_local=True, return_ink=True)
        fused2, local2, valid2 = model(line2, return_local=True, return_ink=True)
        similarity = fused1[0] @ fused2[0].T
    tensor_image(line1_rgb).save(out / "01_line_A.png")
    tensor_image(line2_rgb).save(out / "02_line_B.png")
    save_matrix(similarity.cpu().numpy(), out / "03_image_image_similarity.png", "Image-only fused-vector similarity", xlabel="line B window", ylabel="line A window")
    write_summary(out, f"""Point 13 visual result
Line A fused/local shape: {tuple(fused1.shape)} / {tuple(local1.shape)}
Line B fused/local shape: {tuple(fused2.shape)} / {tuple(local2.shape)}
Valid masks: A={valid1[0].tolist()} B={valid2[0].tolist()}
Similarity matrix shape: {tuple(similarity.shape)}
No text embeddings are used here.
Important: this visualization uses fresh diagnostic weights unless the full evaluation script is run with a trained checkpoint, so inspect it as a pipeline proof, not final alignment quality.
""")


POINTS = {
    1: point01,
    2: point02,
    4: point04,
    5: point05,
    6: point06,
    7: point07,
    8: point08,
    9: point09,
    10: point10,
    11: point11,
    13: point13,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=int, required=True)
    args = parser.parse_args()
    point = int(args.point)
    if point not in POINTS:
        raise SystemExit(f"No generic visualizer for point {point}; that point has its own diagnostic tool.")
    out = ROOT / "Results" / "Diagnostics" / "restoration_points" / f"point_{point:02d}"
    out.mkdir(parents=True, exist_ok=True)
    POINTS[point](out)
    print(f"Visual results saved to: {out}")
    print(f"Open: {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
