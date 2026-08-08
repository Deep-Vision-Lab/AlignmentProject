#!/usr/bin/env python3
"""Create a bbox-independent visual preview of real-line augmentations.

The actual augmented images use the same mild training-style ranges as the real
augmentation pipeline. To make subtle changes auditable after contact-sheet
resizing, each augmentation is accompanied by an amplified absolute-difference
panel and the exact sampled transform parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _resolve(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (dataset_root / path, dataset_root.parent / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (dataset_root / path).resolve()


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _background(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    band = max(1, min(8, h // 8, w // 8))
    samples = np.concatenate([
        arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
        arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3),
    ], axis=0)
    return tuple(int(round(v)) for v in np.median(samples, axis=0))


def _photometric(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    out = image.convert("RGB")
    brightness = rng.uniform(0.88, 1.12)
    contrast = rng.uniform(0.85, 1.20)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    blur_radius = 0.0
    if rng.random() < 0.30:
        blur_radius = rng.uniform(0.15, 0.75)
        out = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(out).astype(np.float32)
    gaussian_std = 0.0
    if rng.random() < 0.85:
        gaussian_std = rng.uniform(1.5, 8.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, gaussian_std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    salt_pepper_density = 0.0
    if rng.random() < 0.45:
        salt_pepper_density = rng.uniform(0.0004, 0.0025)
        generator = np.random.default_rng(rng.randrange(2**32))
        mask = generator.random(arr.shape[:2])
        salt = mask < salt_pepper_density / 2.0
        pepper = (mask >= salt_pepper_density / 2.0) & (mask < salt_pepper_density)
        arr[salt] = 255
        arr[pepper] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="RGB"), {
        "brightness": brightness, "contrast": contrast, "blur_radius": blur_radius,
        "gaussian_std": gaussian_std, "salt_pepper_density": salt_pepper_density,
    }


def _geometry(image: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    source = image.convert("RGB")
    width, height = source.size
    sx = rng.uniform(0.94, 1.04)
    sy = rng.uniform(0.96, 1.04)
    new_w, new_h = max(1, int(round(width * sx))), max(1, int(round(height * sy)))
    resized = source.resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), _background(source))
    max_dx = max(1, int(round(width * 0.015)))
    max_dy = max(1, int(round(height * 0.035)))
    dx, dy = rng.randint(-max_dx, max_dx), rng.randint(-max_dy, max_dy)
    x, y = (width - new_w) // 2 + dx, (height - new_h) // 2 + dy
    canvas.paste(resized, (x, y))
    return canvas, {"scale_x": sx, "scale_y": sy, "dx_px": dx, "dy_px": dy}


def _difference(original: Image.Image, augmented: Image.Image, amplification: float = 8.0) -> tuple[Image.Image, dict]:
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    changed = np.asarray(augmented.convert("RGB").resize(original.size), dtype=np.float32)
    magnitude = np.abs(changed - base).mean(axis=2)
    visible = np.clip(magnitude * float(amplification), 0, 255).astype(np.uint8)
    return Image.fromarray(visible, mode="L").convert("RGB"), {
        "mean_abs_pixel_difference": float(magnitude.mean()),
        "changed_pixel_fraction_gt_2": float(np.mean(magnitude > 2.0)),
        "changed_pixel_fraction_gt_5": float(np.mean(magnitude > 5.0)),
        "difference_visual_amplification": float(amplification),
    }


def _fit_panel(image: Image.Image, panel_w: int, panel_h: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    panel.paste(copy, ((panel_w - copy.width) // 2, (panel_h - copy.height) // 2))
    return panel


def _params_text(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            rendered = f"{value:.5f}" if "density" in key else f"{value:.3f}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " | ".join(parts)


def _sheet(original: Image.Image, variants, title: str) -> Image.Image:
    col_w, panel_h, header_h, label_h = 900, 190, 70, 42
    original_h, row_h = panel_h + label_h, panel_h + label_h
    width = col_w * 2
    height = header_h + original_h + len(variants) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill="black")
    draw.text((12, 32), "Left: actual training-style augmentation | Right: absolute pixel difference x8 (verification only)", fill="black")
    draw.text((12, 50), "Difference panel is not an augmentation and will never be used for training.", fill="black")
    y = header_h
    draw.text((12, y + 7), "ORIGINAL", fill="black")
    sheet.paste(_fit_panel(original, width, panel_h), (0, y + label_h))
    y += original_h
    for name, augmented, diff_image, params, diff_stats in variants:
        draw.text((12, y + 5), f"{name}: {_params_text(params)}", fill="black")
        draw.text((col_w + 12, y + 5), f"DIFF x8: mean_abs={diff_stats['mean_abs_pixel_difference']:.3f} | >2={100.0 * diff_stats['changed_pixel_fraction_gt_2']:.1f}% | >5={100.0 * diff_stats['changed_pixel_fraction_gt_5']:.1f}%", fill="black")
        sheet.paste(_fit_panel(augmented, col_w, panel_h), (0, y + label_h))
        sheet.paste(_fit_panel(diff_image, col_w, panel_h), (col_w, y + label_h))
        y += row_h
    return sheet


def _pair_id(record: dict, fallback: int) -> str:
    return str(record.get("pair_id") or record.get("id") or f"manifest_{fallback:06d}")


def _line_path(dataset_root: Path, record: dict, side_name: str) -> Path | None:
    side = record.get(side_name)
    if not isinstance(side, dict): return None
    value = side.get("line_image_path") or side.get("image_path") or side.get("image")
    if not value: return None
    path = _resolve(dataset_root, value)
    return path if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diff-amplification", type=float, default=8.0)
    args = parser.parse_args()
    dataset_root = Path(args.data_dir).expanduser().resolve()
    manifest = dataset_root / "dataset_manifest.jsonl"
    if not manifest.is_file(): raise SystemExit(f"Manifest not found: {manifest}")
    if args.num_pairs <= 0: raise SystemExit("--num-pairs must be positive")
    if args.diff_amplification <= 0: raise SystemExit("--diff-amplification must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    records = _read_manifest(manifest)
    rng = random.Random(args.seed)
    order = list(range(len(records))); rng.shuffle(order)
    created, summary_rows = 0, []
    for manifest_index in order:
        record = records[manifest_index]
        path_a, path_b = _line_path(dataset_root, record, "A"), _line_path(dataset_root, record, "B")
        if path_a is None or path_b is None: continue
        pair_id = _pair_id(record, manifest_index + 1)
        for side_name, path in (("A", path_a), ("B", path_b)):
            local_rng = random.Random(rng.randrange(2**31))
            with Image.open(path) as opened: original = opened.convert("RGB").copy()
            photometric, photo_params = _photometric(original, local_rng)
            geometry, geometry_params = _geometry(original, local_rng)
            mixed_geometry, mixed_geometry_params = _geometry(original, local_rng)
            mixed, mixed_photo_params = _photometric(mixed_geometry, local_rng)
            mixed_params = {**mixed_geometry_params, **mixed_photo_params}
            photo_diff, photo_stats = _difference(original, photometric, args.diff_amplification)
            geometry_diff, geometry_stats = _difference(original, geometry, args.diff_amplification)
            mixed_diff, mixed_stats = _difference(original, mixed, args.diff_amplification)
            sheet = _sheet(original, [
                ("PHOTOMETRIC", photometric, photo_diff, photo_params, photo_stats),
                ("GEOMETRY", geometry, geometry_diff, geometry_params, geometry_stats),
                ("MIXED", mixed, mixed_diff, mixed_params, mixed_stats),
            ], f"{pair_id} | side {side_name} | real augmentation audit")
            filename = f"pair_{created + 1:03d}_{side_name}.png"
            preview_path = previews_dir / filename
            sheet.save(preview_path)
            summary_rows.append({
                "pair_id": pair_id, "side": side_name, "source_image": str(path), "preview": str(preview_path),
                "photometric": {"parameters": photo_params, "difference": photo_stats},
                "geometry": {"parameters": geometry_params, "difference": geometry_stats},
                "mixed": {"parameters": mixed_params, "difference": mixed_stats},
            })
        created += 1
        if created >= args.num_pairs: break
    if created == 0: raise SystemExit("No manifest pairs with readable A/B line images were found.")
    summary_path = output_dir / "preview_summary.json"
    summary_path.write_text(json.dumps({
        "dataset": str(dataset_root), "pairs_previewed": created, "images_previewed": len(summary_rows),
        "modes": ["original", "photometric", "geometry", "mixed"], "bbox_required": False,
        "actual_augmentation_strength": "training-style mild ranges",
        "difference_visual_amplification": args.diff_amplification, "items": summary_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {created} real-pair augmentation previews")
    print(f"Preview directory: {previews_dir}")
    print(f"Summary: {summary_path}")
    print("Each augmented image now has an amplified DIFF x8 verification panel.")


if __name__ == "__main__":
    main()
