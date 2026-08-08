#!/usr/bin/env python3
"""Create a bbox-independent visual preview of real-line augmentations.

This preview intentionally avoids bbox-dependent injection. It reads real A/B
line images from dataset_manifest.jsonl and creates contact sheets showing the
original beside photometric, geometry, and mixed augmentations.
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _background(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    band = max(1, min(8, h // 8, w // 8))
    samples = np.concatenate(
        [
            arr[:band].reshape(-1, 3),
            arr[-band:].reshape(-1, 3),
            arr[:, :band].reshape(-1, 3),
            arr[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(int(round(v)) for v in np.median(samples, axis=0))


def _photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    out = image.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.88, 1.12))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.20))
    if rng.random() < 0.30:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))

    arr = np.asarray(out).astype(np.float32)
    if rng.random() < 0.85:
        std = rng.uniform(1.5, 8.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, std, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    if rng.random() < 0.45:
        density = rng.uniform(0.0004, 0.0025)
        generator = np.random.default_rng(rng.randrange(2**32))
        mask = generator.random(arr.shape[:2])
        salt = mask < density / 2.0
        pepper = (mask >= density / 2.0) & (mask < density)
        arr[salt] = 255
        arr[pepper] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _geometry(image: Image.Image, rng: random.Random) -> Image.Image:
    source = image.convert("RGB")
    width, height = source.size
    sx = rng.uniform(0.94, 1.04)
    sy = rng.uniform(0.96, 1.04)
    new_w = max(1, int(round(width * sx)))
    new_h = max(1, int(round(height * sy)))
    resized = source.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (width, height), _background(source))
    max_dx = max(1, int(round(width * 0.015)))
    max_dy = max(1, int(round(height * 0.035)))
    dx = rng.randint(-max_dx, max_dx)
    dy = rng.randint(-max_dy, max_dy)
    x = (width - new_w) // 2 + dx
    y = (height - new_h) // 2 + dy
    canvas.paste(resized, (x, y))
    return canvas


def _fit_panel(image: Image.Image, panel_w: int, panel_h: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    panel.paste(copy, ((panel_w - copy.width) // 2, (panel_h - copy.height) // 2))
    return panel


def _sheet(original: Image.Image, variants: list[tuple[str, Image.Image]], title: str) -> Image.Image:
    panel_w = 1100
    panel_h = 220
    header_h = 44
    label_h = 30
    rows = 1 + len(variants)
    sheet = Image.new("RGB", (panel_w, header_h + rows * (panel_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")

    entries = [("ORIGINAL", original)] + variants
    y = header_h
    for label, image in entries:
        draw.text((12, y + 6), label, fill="black")
        panel = _fit_panel(image, panel_w, panel_h)
        sheet.paste(panel, (0, y + label_h))
        y += panel_h + label_h
    return sheet


def _pair_id(record: dict, fallback: int) -> str:
    return str(record.get("pair_id") or record.get("id") or f"manifest_{fallback:06d}")


def _line_path(dataset_root: Path, record: dict, side_name: str) -> Path | None:
    side = record.get(side_name)
    if not isinstance(side, dict):
        return None
    value = side.get("line_image_path") or side.get("image_path") or side.get("image")
    if not value:
        return None
    path = _resolve(dataset_root, value)
    return path if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = Path(args.data_dir).expanduser().resolve()
    manifest = dataset_root / "dataset_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    if args.num_pairs <= 0:
        raise SystemExit("--num-pairs must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(manifest)
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)

    created = 0
    summary_rows = []
    for manifest_index in order:
        record = records[manifest_index]
        path_a = _line_path(dataset_root, record, "A")
        path_b = _line_path(dataset_root, record, "B")
        if path_a is None or path_b is None:
            continue

        pair_id = _pair_id(record, manifest_index + 1)
        for side_name, path in (("A", path_a), ("B", path_b)):
            local_rng = random.Random(rng.randrange(2**31))
            with Image.open(path) as opened:
                original = opened.convert("RGB").copy()
            photometric = _photometric(original, local_rng)
            geometry = _geometry(original, local_rng)
            mixed = _photometric(_geometry(original, local_rng), local_rng)
            sheet = _sheet(
                original,
                [
                    ("PHOTOMETRIC", photometric),
                    ("GEOMETRY", geometry),
                    ("MIXED", mixed),
                ],
                f"{pair_id} | side {side_name} | bbox-independent real augmentation preview",
            )
            filename = f"pair_{created + 1:03d}_{side_name}.png"
            sheet.save(previews_dir / filename)
            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "side": side_name,
                    "source_image": str(path),
                    "preview": str(previews_dir / filename),
                }
            )

        created += 1
        if created >= args.num_pairs:
            break

    if created == 0:
        raise SystemExit("No manifest pairs with readable A/B line images were found.")

    (output_dir / "preview_summary.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_root),
                "pairs_previewed": created,
                "images_previewed": len(summary_rows),
                "modes": ["original", "photometric", "geometry", "mixed"],
                "bbox_required": False,
                "items": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Created {created} real-pair augmentation previews")
    print(f"Preview directory: {previews_dir}")
    print(f"Summary: {output_dir / 'preview_summary.json'}")


if __name__ == "__main__":
    main()
