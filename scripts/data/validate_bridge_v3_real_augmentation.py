#!/usr/bin/env python3
"""Validate Bridge V3 real-anchor appearance augmentation bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

EXPECTED_VARIANTS = {
    "binarized",
    "gaussian_noise",
    "gaussian_blur",
    "blur_noise",
    "binarized_noise",
    "contrast_gamma",
    "salt_pepper",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _different(a: Image.Image, b: Image.Image) -> bool:
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    policy = metadata.get("real_line_augmentation") or {}
    if policy.get("enabled") is not True or policy.get("geometric") is not False:
        raise RuntimeError("Bridge V3 real-line augmentation metadata is missing/invalid")
    if set(policy.get("variant_names") or []) != EXPECTED_VARIANTS:
        raise RuntimeError(f"Unexpected real augmentation variants: {policy.get('variant_names')}")
    if int(policy.get("variants_per_anchor", -1)) != len(EXPECTED_VARIANTS):
        raise RuntimeError("real augmentation variant count is wrong")
    if policy.get("training_variant") != "combined_contrast_gamma_blur_gaussian_noise":
        raise RuntimeError("unexpected training-facing real augmentation recipe")

    anchors = [
        json.loads(line)
        for line in (root / "anchor_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed_training = 0
    checked_variants = 0

    for row in anchors:
        anchor_id = str(row["anchor_id"])
        real = row.get("real") or {}
        augmented = _resolve(root, real.get("image", ""))
        original = _resolve(root, real.get("original_image", ""))
        aug = real.get("appearance_augmentation") or {}
        variants = list(real.get("augmentation_variants") or aug.get("variants") or [])

        if not augmented.is_file() or not original.is_file():
            raise RuntimeError(f"{anchor_id}: missing augmented/original real image")
        if aug.get("geometric_transform") is not False:
            raise RuntimeError(f"{anchor_id}: geometric augmentation is not allowed")
        if aug.get("training_variant") != "combined_contrast_gamma_blur_gaussian_noise":
            raise RuntimeError(f"{anchor_id}: wrong training variant")
        if float(aug.get("gaussian_blur_radius", 0.0)) <= 0:
            raise RuntimeError(f"{anchor_id}: Gaussian blur was not applied")
        if float(aug.get("gaussian_noise_sigma", 0.0)) <= 0:
            raise RuntimeError(f"{anchor_id}: Gaussian noise was not applied")
        if set(str(v.get("name")) for v in variants) != EXPECTED_VARIANTS:
            raise RuntimeError(f"{anchor_id}: missing real augmentation variants")

        with Image.open(augmented) as a_src, Image.open(original) as b_src:
            a = a_src.convert("RGB")
            b = b_src.convert("RGB")
            if a.size != b.size:
                raise RuntimeError(f"{anchor_id}: augmentation changed image geometry")
            if _different(a, b):
                changed_training += 1
            original_size = b.size

        for variant in variants:
            name = str(variant.get("name"))
            path = _resolve(root, variant.get("path", ""))
            if variant.get("geometric_transform") is not False:
                raise RuntimeError(f"{anchor_id}/{name}: geometric augmentation is not allowed")
            if not path.is_file():
                raise RuntimeError(f"{anchor_id}/{name}: missing variant file {path}")
            with Image.open(path) as src:
                image = src.convert("RGB")
                if image.size != original_size:
                    raise RuntimeError(f"{anchor_id}/{name}: variant changed image geometry")
                if name == "binarized":
                    values = np.unique(np.asarray(src.convert("L"), dtype=np.uint8))
                    if not set(values.tolist()).issubset({0, 255}):
                        raise RuntimeError(f"{anchor_id}: binarized variant is not binary")
            checked_variants += 1

        human_dir = root / "real" / anchor_id
        for required in (
            "real.txt",
            "augmentation.json",
            "real_binarized.png",
            "real_gaussian_noise.png",
            "real_gaussian_blur.png",
            "real_blur_noise.png",
            "real_binarized_noise.png",
            "real_contrast_gamma.png",
            "real_salt_pepper.png",
        ):
            if not (human_dir / required).is_file():
                raise RuntimeError(f"{anchor_id}: missing human real file {required}")
        if not any(human_dir.glob("real_original.*")):
            raise RuntimeError(f"{anchor_id}: missing human original real backup")

    if changed_training != len(anchors):
        raise RuntimeError(f"Only {changed_training}/{len(anchors)} training-facing real images changed")
    if checked_variants != len(anchors) * len(EXPECTED_VARIANTS):
        raise RuntimeError("real augmentation variant validation count mismatch")
    if int(metadata.get("real_augmented_count", -1)) != len(anchors):
        raise RuntimeError("metadata real_augmented_count does not match anchor count")

    print("=== BRIDGE V3 REAL AUGMENTATION TEST ===")
    print(f"anchors_checked={len(anchors)}")
    print(f"variants_checked={checked_variants}")
    print("training_variant=combined_contrast_gamma_blur_gaussian_noise")
    print("variants=binarized,gaussian_noise,gaussian_blur,blur_noise,binarized_noise,contrast_gamma,salt_pepper")
    print("geometric_transform=False")
    print("originals_preserved=True")
    print("REAL_AUGMENTATION_TEST=PASS")


if __name__ == "__main__":
    main()
