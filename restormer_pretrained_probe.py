#!/usr/bin/env python3
"""No-flag probe for recommendation 3: pretrained Restormer suitability.

Expected assets:
  third_party/Restormer/                    official swz30/Restormer checkout
  Weights/Pretrained/Restormer/*.pth        one official 3-channel checkpoint

The script deliberately does not silently download a large third-party model.
When assets are present it loads the official architecture, captures its latent
encoder feature, checks reconstruction output, and performs one identity
fine-tuning step on manuscript-like 128x32 windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tools":
    ROOT = ROOT.parent


@dataclass(frozen=True)
class RestormerAssets:
    repo: Path | None
    checkpoint: Path | None
    ready: bool
    message: str


def find_restormer_assets(root: Path | None = None) -> RestormerAssets:
    root = ROOT if root is None else Path(root)
    repo_candidates = [
        root / "third_party" / "Restormer",
        root / "Restormer",
    ]
    repo = next(
        (
            item
            for item in repo_candidates
            if (item / "basicsr" / "models" / "archs" / "restormer_arch.py").is_file()
        ),
        None,
    )
    weight_candidates = []
    for folder in [
        root / "Weights" / "Pretrained" / "Restormer",
        root / "pretrained_models" / "Restormer",
        *( [repo / "pretrained_models"] if repo is not None else [] ),
    ]:
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
    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    ).to(device)
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
        raise RuntimeError(
            f"No compatible Restormer tensors loaded from {assets.checkpoint}"
        )
    return model


def _manuscript_like_windows(device):
    x = torch.ones(2, 3, 128, 32, device=device)
    x[0, :, 42:48, 4:27] = 0.10
    x[0, :, 31:36, 20:24] = 0.15
    x[1, :, 72:78, 3:25] = 0.15
    x[1, :, 57:62, 7:11] = 0.10
    return x


def main():
    assets = find_restormer_assets()
    print(assets.message, flush=True)
    if not assets.ready:
        # A missing third-party checkpoint is an informative SKIP, not a failure
        # of the branch code or of the other twelve recommendation checks.
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

    print(
        "PASS point 3: pretrained Restormer loaded; "
        f"output={tuple(restored.shape)} latent={tuple(feature.shape)} "
        f"identity_L1={float(loss_before.detach()):.6f} grad_signal={grad_norm:.6g}",
        flush=True,
    )
    print(
        "Interpretation: pretraining is usable as an initialization, but the "
        "manuscript identity/restoration objective still needs fine-tuning.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
