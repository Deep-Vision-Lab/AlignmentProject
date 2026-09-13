#!/usr/bin/env python3
"""No-flag probe for recommendation 3: pretrained Restormer suitability.

Expected assets:
  third_party/Restormer/                    official swz30/Restormer checkout
  Weights/Pretrained/Restormer/*.pth        one official 3-channel checkpoint

When assets are present the script loads Restormer, captures its latent encoder
feature, performs one identity-restoration fine-tuning step, and saves visible
results to Results/Diagnostics/restoration_points/point_03/.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tools":
    ROOT = ROOT.parent
OUT_DIR = ROOT / "Results" / "Diagnostics" / "restoration_points" / "point_03"


@dataclass(frozen=True)
class RestormerAssets:
    repo: Path | None
    checkpoint: Path | None
    ready: bool
    message: str


def find_restormer_assets(root: Path | None = None) -> RestormerAssets:
    root = ROOT if root is None else Path(root)
    repo_candidates = [root / "third_party" / "Restormer", root / "Restormer"]
    repo = next(
        (
            item
            for item in repo_candidates
            if (item / "basicsr" / "models" / "archs" / "restormer_arch.py").is_file()
        ),
        None,
    )
    weight_candidates = []
    folders = [
        root / "Weights" / "Pretrained" / "Restormer",
        root / "pretrained_models" / "Restormer",
    ]
    if repo is not None:
        folders.append(repo / "pretrained_models")
    for folder in folders:
        if folder.is_dir():
            weight_candidates.extend(sorted(folder.rglob("*.pth")))
    checkpoint = weight_candidates[0] if weight_candidates else None
    ready = repo is not None and checkpoint is not None
    if ready:
        message = f"Restormer ready: repo={repo} checkpoint={checkpoint}"
    else:
        missing = []
        if repo is None:
            missing.append(
                "official repo (clone https://github.com/swz30/Restormer.git to third_party/Restormer)"
            )
        if checkpoint is None:
            missing.append(
                "official 3-channel pretrained .pth under Weights/Pretrained/Restormer/"
            )
        message = "Restormer probe SKIPPED: missing " + " and ".join(missing)
    return RestormerAssets(repo, checkpoint, ready, message)


def load_restormer(assets: RestormerAssets, device="cpu"):
    if not assets.ready or assets.repo is None or assets.checkpoint is None:
        raise RuntimeError(assets.message)
    sys.path.insert(0, str(assets.repo))
    module = importlib.import_module("basicsr.models.archs.restormer_arch")
    Restormer = module.Restormer
    checkpoint_name = assets.checkpoint.name.lower()
    # The official Restormer demo uses BiasFree LayerNorm for Real Denoising
    # and Gaussian Color Denoising checkpoints.
    layer_norm_type = (
        "BiasFree"
        if (
            "real_denoising" in checkpoint_name
            or "gaussian_color_denoising" in checkpoint_name
        )
        else "WithBias"
    )
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type=layer_norm_type,
        dual_pixel_task=False,
    ).to(device)
    model._alignment_probe_layer_norm_type = layer_norm_type
    payload = torch.load(assets.checkpoint, map_location=device)
    if isinstance(payload, dict):
        state = payload.get("params", payload.get("state_dict", payload))
    else:
        state = payload
    cleaned = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    incompatible = model.load_state_dict(cleaned, strict=False)
    loaded_count = len(cleaned) - len(incompatible.unexpected_keys)
    if loaded_count <= 0:
        raise RuntimeError(f"No compatible Restormer tensors loaded from {assets.checkpoint}")
    return model


def _probe_windows(device):
    """Prefer two informative windows from the real synthetic dataset."""
    dataset = os.environ.get("SYNTHETIC_DIAG_DATASET", "").strip()
    index = os.environ.get("SYNTHETIC_DIAG_INDEX", "").strip()
    if dataset and index:
        root = Path(dataset).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        image = None
        for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = root / "images" / f"img1_{int(index)}{suffix}"
            if candidate.is_file():
                image = candidate
                break
        if image is not None:
            from zero_shot_preprocessing import ManuscriptLinePreprocessor
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
            prepared, _ = processor.preprocess_with_metadata(Image.open(image).convert("RGB"))
            arr = np.asarray(prepared, dtype=np.float32) / 255.0
            line = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            patches = line.unfold(3, 32, 16).permute(0, 3, 1, 2, 4).contiguous()[0]
            darkness = (1.0 - patches.mean(dim=(1, 2, 3))).cpu()
            chosen = torch.topk(darkness, k=min(2, int(patches.shape[0]))).indices
            windows = patches.index_select(0, chosen).to(device)
            if windows.shape[0] == 2:
                return windows, f"real synthetic line {image}, windows={chosen.tolist()}"

    x = torch.ones(2, 3, 128, 32, device=device)
    x[0, :, 42:48, 4:27] = 0.10
    x[0, :, 31:36, 20:24] = 0.15
    x[1, :, 72:78, 3:25] = 0.15
    x[1, :, 57:62, 7:11] = 0.10
    return x, "fallback manuscript-like artificial windows"


def _tensor_image(x):
    x = x.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_pair_sheet(inputs, outputs):
    w, h, label_h = 32, 128, 24
    canvas = Image.new("RGB", (2 * w, 2 * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (name, tensor) in enumerate([("input", inputs), ("restored", outputs)]):
        for i in range(2):
            y0 = r * (h + label_h)
            draw.text((i * w + 2, y0 + 3), f"{name} {i}", fill="black")
            canvas.paste(_tensor_image(tensor[i]), (i * w, y0 + label_h))
    canvas.save(OUT_DIR / "01_input_vs_pretrained_output.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = find_restormer_assets()
    print(assets.message, flush=True)
    if not assets.ready:
        (OUT_DIR / "summary.txt").write_text(
            "Point 03 pretrained Restormer result\nSTATUS: SKIPPED\n"
            + assets.message + "\n",
            encoding="utf-8",
        )
        print(f"Result saved to: {OUT_DIR / 'summary.txt'}")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_restormer(assets, device=device)
    model.train()
    windows = _manuscript_like_windows(device)

    latent = {}
    handle = model.latent.register_forward_hook(
        lambda _module, _inputs, output: latent.__setitem__("value", output)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    restored = model(windows)
    if restored.shape != windows.shape:
        raise RuntimeError(
            f"Restormer output shape {tuple(restored.shape)} != input {tuple(windows.shape)}"
        )
    loss_before = F.l1_loss(restored, windows)
    loss_before.backward()
    grad_norm = sum(
        float(p.grad.detach().abs().mean())
        for p in model.parameters()
        if p.grad is not None
    )
    optimizer.step()
    handle.remove()

    if "value" not in latent:
        raise RuntimeError("Restormer latent encoder feature was not captured")
    feature = latent["value"].detach().float().mean(dim=(-2, -1))
    if feature.shape[0] != 2 or torch.allclose(feature[0], feature[1]):
        raise RuntimeError("Restormer latent features did not distinguish the two windows")
    if grad_norm <= 0:
        raise RuntimeError("Restormer fine-tuning step produced no gradients")

    _save_pair_sheet(windows.detach(), restored.detach())
    cosine = float(F.cosine_similarity(feature[0:1], feature[1:2], dim=-1)[0])
    summary = (
        "Point 03 pretrained Restormer result\n"
        "STATUS: PASS\n"
        f"checkpoint={assets.checkpoint}\n"
        f"layer_norm_type={getattr(model, '_alignment_probe_layer_norm_type', 'unknown')}\n"
        f"output_shape={tuple(restored.shape)}\n"
        f"latent_shape={tuple(feature.shape)}\n"
        f"latent_cosine_between_two_windows={cosine:.8f}\n"
        f"identity_L1_before_step={float(loss_before.detach()):.8f}\n"
        f"gradient_signal={grad_norm:.8g}\n"
        f"window_source={window_source}\n\n"
        "Inspect 01_input_vs_pretrained_output.png to judge the pretrained output yourself.\n"
        "A poor reconstruction here means pretraining alone is not sufficient for manuscript windows.\n"
    )
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(
        "PASS point 3: pretrained Restormer loaded; "
        f"output={tuple(restored.shape)} latent={tuple(feature.shape)} "
        f"identity_L1={float(loss_before.detach()):.6f} grad_signal={grad_norm:.6g}",
        flush=True,
    )
    print(f"probe windows: {window_source}", flush=True)
    print(
        "Interpretation: pretraining is usable as an initialization, but the "
        "manuscript identity/restoration objective still needs fine-tuning.",
        flush=True,
    )
    print(f"Visual results saved to: {OUT_DIR}")
    print(f"Open: {OUT_DIR / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
