#!/usr/bin/env python3
"""Create deterministic non-geometric augmentation variants for Bridge V3 real anchors.

The source ArabicDataset is never modified. The self-contained copied real line is
preserved as ``real_original.*`` and appearance-only variants are created under
``images/real/<anchor_id>/``. Real anchors are processed concurrently using the same
BRIDGE_BUILD_WORKERS/SLURM_CPUS_PER_TASK budget as synthetic generation.

No crop, translation, rotation, resize, warp, perspective, elastic transform, or
character/stroke deletion is used.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _anchor_seed(base_seed: int, anchor_id: str) -> int:
    digest = hashlib.sha256(anchor_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(base_seed) + offset) % (2**32 - 1)


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127
    weighted_sum = float(np.dot(np.arange(256, dtype=np.float64), hist))
    weight_bg = 0.0
    sum_bg = 0.0
    best_threshold = 127
    best_between = -1.0
    for threshold in range(256):
        weight_bg += hist[threshold]
        if weight_bg <= 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += threshold * hist[threshold]
        mean_bg = sum_bg / weight_bg
        mean_fg = (weighted_sum - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_between:
            best_between = between
            best_threshold = threshold
    return int(best_threshold)


def _binarize(image: Image.Image) -> tuple[Image.Image, int]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L").convert("RGB"), threshold


def _gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr += rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _gamma(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0.0, 1.0), float(gamma)) * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _salt_pepper(image: Image.Image, probability: float, seed: int) -> Image.Image:
    if probability <= 0:
        return image.copy()
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask = rng.random(arr.shape[:2])
    half = float(probability) / 2.0
    arr[mask < half] = 0
    arr[(mask >= half) & (mask < probability)] = 255
    return Image.fromarray(arr, mode="RGB")


def _save_variant(root: Path, real_path: Path, name: str, image: Image.Image, ops: list[str], params: dict) -> dict:
    path = real_path.with_name(f"real_{name}.png")
    image.convert("RGB").save(path, format="PNG")
    return {
        "name": name,
        "path": path.relative_to(root).as_posix(),
        "operations": list(ops),
        "parameters": params,
        "geometric_transform": False,
    }


def _augment_one(record: dict, root: Path, args) -> tuple[str, dict]:
    anchor_id = str(record["anchor_id"])
    real = record["real"]
    real_path = _resolve(root, real["image"])
    if not real_path.is_file():
        raise FileNotFoundError(f"{anchor_id}: missing copied real image {real_path}")

    suffix = real_path.suffix.lower() or ".png"
    original_path = real_path.with_name(f"real_original{suffix}")
    if not original_path.exists():
        shutil.copy2(real_path, original_path)

    with Image.open(original_path) as src:
        original = src.convert("RGB")
    original_size = original.size

    seed = _anchor_seed(args.seed, anchor_id)
    rng = np.random.default_rng(seed)
    blur_radius = float(rng.uniform(args.blur_min_radius, args.blur_max_radius))
    noise_sigma = float(rng.uniform(args.noise_min_sigma, args.noise_max_sigma))
    contrast = float(rng.uniform(args.contrast_min, args.contrast_max))
    gamma = float(rng.uniform(args.gamma_min, args.gamma_max))
    salt_pepper_prob = float(rng.uniform(args.salt_pepper_min_prob, args.salt_pepper_max_prob))
    noise_seed = int(rng.integers(0, 2**32 - 1))
    combo_noise_seed = int(rng.integers(0, 2**32 - 1))
    binary_noise_seed = int(rng.integers(0, 2**32 - 1))
    salt_pepper_seed = int(rng.integers(0, 2**32 - 1))

    binary, otsu_threshold = _binarize(original)
    gaussian_noise = _gaussian_noise(original, noise_sigma, noise_seed)
    gaussian_blur = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blur_noise = _gaussian_noise(gaussian_blur, noise_sigma, combo_noise_seed)
    binarized_noise = _gaussian_noise(binary, max(1.0, noise_sigma * 0.65), binary_noise_seed)
    contrast_gamma = _gamma(ImageEnhance.Contrast(original).enhance(contrast), gamma)
    salt_pepper = _salt_pepper(original, salt_pepper_prob, salt_pepper_seed)

    training_image = _gamma(ImageEnhance.Contrast(original).enhance(contrast), gamma)
    training_image = training_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    training_image = _gaussian_noise(training_image, noise_sigma, combo_noise_seed)
    if training_image.size != original_size:
        raise RuntimeError(f"{anchor_id}: appearance augmentation changed geometry")
    training_image.save(real_path)

    variants = [
        _save_variant(root, real_path, "binarized", binary, ["otsu_binarization"], {"otsu_threshold": otsu_threshold}),
        _save_variant(root, real_path, "gaussian_noise", gaussian_noise, ["gaussian_noise"], {"sigma": round(noise_sigma, 5), "seed": noise_seed}),
        _save_variant(root, real_path, "gaussian_blur", gaussian_blur, ["gaussian_blur"], {"radius": round(blur_radius, 5)}),
        _save_variant(root, real_path, "blur_noise", blur_noise, ["gaussian_blur", "gaussian_noise"], {"radius": round(blur_radius, 5), "sigma": round(noise_sigma, 5), "seed": combo_noise_seed}),
        _save_variant(root, real_path, "binarized_noise", binarized_noise, ["otsu_binarization", "gaussian_noise"], {"otsu_threshold": otsu_threshold, "sigma": round(max(1.0, noise_sigma * 0.65), 5), "seed": binary_noise_seed}),
        _save_variant(root, real_path, "contrast_gamma", contrast_gamma, ["contrast", "gamma"], {"contrast": round(contrast, 5), "gamma": round(gamma, 5)}),
        _save_variant(root, real_path, "salt_pepper", salt_pepper, ["salt_pepper_noise"], {"probability": round(salt_pepper_prob, 6), "seed": salt_pepper_seed}),
    ]

    aug = {
        "geometric_transform": False,
        "training_variant": "combined_contrast_gamma_blur_gaussian_noise",
        "training_image": real_path.relative_to(root).as_posix(),
        "original_image": original_path.relative_to(root).as_posix(),
        "gaussian_blur_radius": round(blur_radius, 5),
        "gaussian_noise_sigma": round(noise_sigma, 5),
        "gaussian_noise_seed": combo_noise_seed,
        "contrast": round(contrast, 5),
        "gamma": round(gamma, 5),
        "base_seed": int(args.seed),
        "variants": variants,
    }
    real["original_image"] = aug["original_image"]
    real["appearance_augmentation"] = aug
    real["augmentation_variants"] = variants

    anchor_json = root / "anchors" / anchor_id / "anchor.json"
    if anchor_json.is_file():
        payload = json.loads(anchor_json.read_text(encoding="utf-8"))
        payload["real"]["original_image"] = aug["original_image"]
        payload["real"]["appearance_augmentation"] = aug
        payload["real"]["augmentation_variants"] = variants
        _write_json(anchor_json, payload)
    return anchor_id, aug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--blur-min-radius", type=float, default=0.15)
    parser.add_argument("--blur-max-radius", type=float, default=1.00)
    parser.add_argument("--noise-min-sigma", type=float, default=2.0)
    parser.add_argument("--noise-max-sigma", type=float, default=8.0)
    parser.add_argument("--contrast-min", type=float, default=0.90)
    parser.add_argument("--contrast-max", type=float, default=1.12)
    parser.add_argument("--gamma-min", type=float, default=0.88)
    parser.add_argument("--gamma-max", type=float, default=1.12)
    parser.add_argument("--salt-pepper-min-prob", type=float, default=0.001)
    parser.add_argument("--salt-pepper-max-prob", type=float, default=0.006)
    args = parser.parse_args()

    if not 0 <= args.blur_min_radius <= args.blur_max_radius:
        parser.error("invalid blur radius range")
    if not 0 <= args.noise_min_sigma <= args.noise_max_sigma:
        parser.error("invalid Gaussian-noise sigma range")
    if not 0 < args.contrast_min <= args.contrast_max:
        parser.error("invalid contrast range")
    if not 0 < args.gamma_min <= args.gamma_max:
        parser.error("invalid gamma range")
    if not 0 <= args.salt_pepper_min_prob <= args.salt_pepper_max_prob <= 0.05:
        parser.error("invalid salt-and-pepper probability range")

    root = Path(args.data_dir).expanduser().resolve()
    anchor_index_path = root / "anchor_index.jsonl"
    manifest_path = root / "dataset_manifest.jsonl"
    metadata_path = root / "metadata.json"
    if not anchor_index_path.is_file() or not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Bridge V3 must be organized before real-line augmentation")

    anchors = [json.loads(line) for line in anchor_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    requested_workers = int(os.environ.get("BRIDGE_BUILD_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1))))
    workers = max(1, min(requested_workers, len(anchors)))
    augmentation_by_anchor: dict[str, dict] = {}

    print(f"real_augmentation_workers={workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_augment_one, record, root, args) for record in anchors]
        for index, future in enumerate(futures, start=1):
            anchor_id, aug = future.result()
            augmentation_by_anchor[anchor_id] = aug
            if index % 100 == 0 or index == len(futures):
                print(f"real_augmentation_progress={index}/{len(futures)}", flush=True)

    with anchor_index_path.open("w", encoding="utf-8") as handle:
        for record in anchors:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in manifest_rows:
        anchor_id = str((row.get("bridge") or {}).get("anchor_id") or "")
        if anchor_id not in augmentation_by_anchor:
            raise RuntimeError(f"Manifest row references unknown real anchor: {anchor_id}")
        row.setdefault("bridge", {})["real_appearance_augmentation"] = augmentation_by_anchor[anchor_id]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    real_index_path = root / "real_lines_index.jsonl"
    if real_index_path.is_file():
        rows = [json.loads(line) for line in real_index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with real_index_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                aug = augmentation_by_anchor[str(row["anchor_id"])]
                row["original_image"] = aug["original_image"]
                row["appearance_augmentation"] = aug
                row["augmentation_variants"] = aug["variants"]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["real_line_augmentation"] = {
        "enabled": True,
        "geometric": False,
        "training_variant": "combined_contrast_gamma_blur_gaussian_noise",
        "variant_names": [
            "binarized",
            "gaussian_noise",
            "gaussian_blur",
            "blur_noise",
            "binarized_noise",
            "contrast_gamma",
            "salt_pepper",
        ],
        "operations": [
            "otsu_binarization",
            "gaussian_noise",
            "gaussian_blur",
            "contrast",
            "gamma",
            "salt_pepper_noise",
        ],
        "blur_radius_range": [args.blur_min_radius, args.blur_max_radius],
        "noise_sigma_range": [args.noise_min_sigma, args.noise_max_sigma],
        "contrast_range": [args.contrast_min, args.contrast_max],
        "gamma_range": [args.gamma_min, args.gamma_max],
        "salt_pepper_probability_range": [args.salt_pepper_min_prob, args.salt_pepper_max_prob],
        "seed": args.seed,
        "originals_preserved": True,
        "variants_per_anchor": 7,
        "parallel_workers": workers,
    }
    metadata["real_augmented_count"] = len(anchors)
    _write_json(metadata_path, metadata)

    print("=== BRIDGE V3 REAL-LINE AUGMENTATION BUNDLE ===")
    print(f"root={root}")
    print(f"real_lines_augmented={len(anchors)}")
    print(f"parallel_workers={workers}")
    print("training_variant=combined_contrast_gamma_blur_gaussian_noise")
    print("variants=binarized,gaussian_noise,gaussian_blur,blur_noise,binarized_noise,contrast_gamma,salt_pepper")
    print(f"blur_radius_range={[args.blur_min_radius, args.blur_max_radius]}")
    print(f"noise_sigma_range={[args.noise_min_sigma, args.noise_max_sigma]}")
    print(f"contrast_range={[args.contrast_min, args.contrast_max]}")
    print(f"gamma_range={[args.gamma_min, args.gamma_max]}")
    print(f"salt_pepper_probability_range={[args.salt_pepper_min_prob, args.salt_pepper_max_prob]}")
    print("geometric_transform=False")
    print("originals_preserved=True")
    print("REAL_AUGMENTATION=READY")


if __name__ == "__main__":
    main()
