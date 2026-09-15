#!/usr/bin/env python3
"""Download/check the pretrained weights required by this branch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(
    os.environ.get("PRETRAINED_CACHE_DIR", PROJECT_ROOT / "Pretrained")
).expanduser().resolve()
TORCH_HOME = Path(
    os.environ.get("TORCH_HOME", CACHE_ROOT / "torch")
).expanduser().resolve()
HF_HOME = Path(
    os.environ.get("HF_HOME", CACHE_ROOT / "huggingface")
).expanduser().resolve()

os.environ["PRETRAINED_CACHE_DIR"] = str(CACHE_ROOT)
os.environ["TORCH_HOME"] = str(TORCH_HOME)
os.environ["HF_HOME"] = str(HF_HOME)

from torchvision.models import ResNet18_Weights, resnet18
from transformers import ViTModel

DEFAULT_TINY_VIT_MODEL = "facebook/deit-tiny-patch16-224"


def resnet_checkpoint_path() -> Path:
    filename = Path(urlparse(ResNet18_Weights.DEFAULT.url).path).name
    return TORCH_HOME / "hub" / "checkpoints" / filename


def check_only(model_name: str) -> None:
    checkpoint = resnet_checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing cached ResNet-18 ImageNet weights: {checkpoint}. "
            "Run this script once without --check-only on a node with internet."
        )
    try:
        model = ViTModel.from_pretrained(model_name, local_files_only=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"Missing cached ViT-Tiny weights for '{model_name}' under "
            f"HF_HOME={HF_HOME}. Run this script once without --check-only "
            "on a node with internet."
        ) from exc
    del model


def download(model_name: str) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TORCH_HOME.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)

    # torchvision downloads ResNet-18 to TORCH_HOME/hub/checkpoints.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    del model

    # Transformers downloads the ImageNet-pretrained DeiT-Tiny checkpoint.
    model = ViTModel.from_pretrained(model_name)
    del model

    check_only(model_name)
    print(f"PRETRAINED_CACHE ready={CACHE_ROOT}")
    print(f"RESNET18 weights={resnet_checkpoint_path()}")
    print(f"VIT_TINY model={model_name} hf_home={HF_HOME}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--vit-model",
        default=os.environ.get(
            "TINY_VIT_PRETRAINED_MODEL", DEFAULT_TINY_VIT_MODEL
        ),
    )
    args = parser.parse_args()
    if args.check_only:
        check_only(args.vit_model)
    else:
        download(args.vit_model)


if __name__ == "__main__":
    main()
