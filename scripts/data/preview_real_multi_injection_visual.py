#!/usr/bin/env python3
"""Preview 1/2/3 shared real-handwriting injections without bbox annotations.

This is a VISUAL-ONLY inspection tool. It does not modify the training dataset.
It reads real A/B line pairs and their transcripts, finds exact shared Arabic
words in donor pairs, estimates the RTL horizontal location of each shared word
from transcript position plus foreground bounds, crops the word independently
from donor A and donor B, and pastes those two different real handwriting crops
into another real A/B pair. Each injected component is outlined in red.

The purpose is to inspect the desired multi-component injection behaviour while
page-level bbox-to-line mapping is still unresolved. Do not use these estimated
regions as training annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageOps

_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_TOKEN = re.compile(r"[\u0600-\u06ff]+")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (root / path, root.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _strip_marks(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return _ARABIC_DIACRITICS.sub("", text.replace("ـ", ""))


def _tokens(text: str) -> list[str]:
    return [token for token in _ARABIC_TOKEN.findall(_strip_marks(text)) if token]


def _pair_id(record: dict, fallback: int) -> str:
    return str(record.get("pair_id") or record.get("id") or f"manifest_{fallback:06d}")


def _side_paths(root: Path, record: dict, side_name: str) -> tuple[Path, Path] | None:
    side = record.get(side_name)
    if not isinstance(side, dict):
        return None
    image_value = side.get("line_image_path") or side.get("image_path") or side.get("image")
    text_value = side.get("text_original_path") or side.get("text_path")
    if not image_value or not text_value:
        return None
    image = _resolve(root, image_value)
    text = _resolve(root, text_value)
    if not image.is_file() or not text.is_file():
        return None
    return image, text


def _background(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    band = max(1, min(8, h // 8, w // 8))
    samples = np.concatenate(
        [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
         arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)],
        axis=0,
    )
    return tuple(int(round(v)) for v in np.median(samples, axis=0))


def _foreground_box(image: Image.Image) -> tuple[int, int, int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate([gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
    bg = float(np.median(border)) if border.size else 255.0
    # Manuscript ink is normally darker than the background. Keep a generous threshold.
    threshold = max(0.0, bg - 22.0)
    mask = gray < threshold
    ys, xs = np.where(mask)
    if xs.size < 8:
        return 0, 0, image.width, image.height
    pad_x = max(2, int(round(image.width * 0.01)))
    pad_y = max(1, int(round(image.height * 0.05)))
    return (
        max(0, int(xs.min()) - pad_x),
        max(0, int(ys.min()) - pad_y),
        min(image.width, int(xs.max()) + 1 + pad_x),
        min(image.height, int(ys.max()) + 1 + pad_y),
    )


def _token_span(tokens: list[str], index: int, image: Image.Image) -> tuple[int, int, int, int]:
    if not tokens or index < 0 or index >= len(tokens):
        raise ValueError("invalid token index")
    x0, y0, x1, y1 = _foreground_box(image)
    ink_w = max(1, x1 - x0)
    # Approximate visual width from character count, with a small inter-word gap.
    weights = [max(1.0, float(len(token))) for token in tokens]
    gap = 0.45
    total = sum(weights) + gap * max(0, len(tokens) - 1)
    before = sum(weights[:index]) + gap * index
    after = before + weights[index]
    # Arabic transcript order is RTL: token 0 begins at the right side.
    right = x1 - int(round((before / total) * ink_w))
    left = x1 - int(round((after / total) * ink_w))
    pad = max(3, int(round(ink_w * 0.008)))
    return max(x0, left - pad), y0, min(x1, right + pad), y1


def _shared_words(text_a: str, text_b: str, min_chars: int = 4) -> list[tuple[str, int, int]]:
    a = _tokens(text_a)
    b = _tokens(text_b)
    positions_b: dict[str, list[int]] = {}
    for j, token in enumerate(b):
        positions_b.setdefault(token, []).append(j)
    result = []
    seen = set()
    for i, token in enumerate(a):
        if len(token) < min_chars or token in seen:
            continue
        if token in positions_b:
            result.append((token, i, positions_b[token][0]))
            seen.add(token)
    return result


def _load_pair(root: Path, record: dict, fallback: int) -> dict | None:
    side_a = _side_paths(root, record, "A")
    side_b = _side_paths(root, record, "B")
    if side_a is None or side_b is None:
        return None
    text_a = side_a[1].read_text(encoding="utf-8").strip()
    text_b = side_b[1].read_text(encoding="utf-8").strip()
    return {
        "pair_id": _pair_id(record, fallback),
        "image_a": side_a[0], "image_b": side_b[0],
        "text_a": text_a, "text_b": text_b,
        "tokens_a": _tokens(text_a), "tokens_b": _tokens(text_b),
        "shared": _shared_words(text_a, text_b),
    }


def _crop_word(pair: dict, shared: tuple[str, int, int]) -> tuple[str, Image.Image, Image.Image]:
    word, ia, ib = shared
    with Image.open(pair["image_a"]) as opened:
        image_a = opened.convert("RGB")
    with Image.open(pair["image_b"]) as opened:
        image_b = opened.convert("RGB")
    rect_a = _token_span(pair["tokens_a"], ia, image_a)
    rect_b = _token_span(pair["tokens_b"], ib, image_b)
    crop_a = image_a.crop(rect_a)
    crop_b = image_b.crop(rect_b)
    return word, crop_a, crop_b


def _ink_crop(image: Image.Image) -> Image.Image:
    box = _foreground_box(image)
    crop = image.crop(box)
    return crop if crop.width > 2 and crop.height > 2 else image


def _paste_component(base: Image.Image, crop: Image.Image, center_fraction: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    out = base.convert("RGB").copy()
    crop = _ink_crop(crop.convert("RGB"))
    max_h = max(8, int(round(out.height * 0.78)))
    max_w = max(20, int(round(out.width * 0.24)))
    fitted = ImageOps.contain(crop, (max_w, max_h), Image.Resampling.LANCZOS)
    target_x = int(round(center_fraction * out.width - fitted.width / 2))
    target_y = int(round((out.height - fitted.height) / 2))
    target_x = max(0, min(out.width - fitted.width, target_x))
    target_y = max(0, min(out.height - fitted.height, target_y))
    rect = (target_x, target_y, target_x + fitted.width, target_y + fitted.height)
    draw = ImageDraw.Draw(out)
    draw.rectangle(rect, fill=_background(out))
    out.paste(fitted, (target_x, target_y))
    return out, rect


def _inject(base_a: Image.Image, base_b: Image.Image, components: list[tuple[str, Image.Image, Image.Image]]) -> tuple[Image.Image, Image.Image, list[dict]]:
    out_a = base_a.copy()
    out_b = base_b.copy()
    n = len(components)
    if n == 1:
        centers = [0.50]
    elif n == 2:
        centers = [0.32, 0.68]
    else:
        centers = [0.22, 0.50, 0.78]
    metadata = []
    rects_a = []
    rects_b = []
    for center, (word, crop_a, crop_b) in zip(centers, components):
        out_a, rect_a = _paste_component(out_a, crop_a, center)
        out_b, rect_b = _paste_component(out_b, crop_b, center)
        rects_a.append(rect_a)
        rects_b.append(rect_b)
        metadata.append({"shared_text": word, "rect_A": list(rect_a), "rect_B": list(rect_b)})
    draw_a = ImageDraw.Draw(out_a)
    draw_b = ImageDraw.Draw(out_b)
    for rect in rects_a:
        draw_a.rectangle(rect, outline=(220, 30, 30), width=3)
    for rect in rects_b:
        draw_b.rectangle(rect, outline=(220, 30, 30), width=3)
    return out_a, out_b, metadata


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def _sheet(original_a: Image.Image, original_b: Image.Image, variants: list[tuple[str, Image.Image, Image.Image, list[dict]]], title: str) -> Image.Image:
    panel_w, panel_h = 1000, 180
    label_h, header_h = 34, 46
    rows = 1 + len(variants)
    sheet = Image.new("RGB", (panel_w * 2, header_h + rows * (panel_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black")
    entries = [("ORIGINAL", original_a, original_b, [])] + variants
    y = header_h
    for label, image_a, image_b, meta in entries:
        extra = ""
        if meta:
            extra = " | injected: " + " ; ".join(item["shared_text"] for item in meta)
        draw.text((12, y + 7), f"{label}{extra}", fill="black")
        draw.text((panel_w + 12, y + 7), "line B", fill="black")
        sheet.paste(_fit(image_a, panel_w, panel_h), (0, y + label_h))
        sheet.paste(_fit(image_b, panel_w, panel_h), (panel_w, y + label_h))
        y += panel_h + label_h
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--source-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default="high_match,medium_match")
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    manifest = root / "dataset_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    output = Path(args.output_dir).expanduser().resolve()
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)

    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    raw = [r for r in _read_manifest(manifest) if not labels or str(r.get("label_type", "")) in labels]
    rng = random.Random(args.seed)
    rng.shuffle(raw)
    raw = raw[: max(args.source_pairs, args.num_pairs)]

    pairs = []
    donors = []
    for idx, record in enumerate(raw, start=1):
        pair = _load_pair(root, record, idx)
        if pair is None:
            continue
        pairs.append(pair)
        if pair["shared"]:
            donors.append(pair)

    if not pairs:
        raise RuntimeError("No readable real A/B pairs were found in the manifest.")
    if not donors:
        raise RuntimeError("No donor pairs with an exact shared Arabic word were found in the selected manifest rows.")

    created = 0
    summary_items = []
    rng.shuffle(pairs)
    for target in pairs:
        donor_pool = [d for d in donors if d["pair_id"] != target["pair_id"]]
        if len(donor_pool) < 3:
            break
        with Image.open(target["image_a"]) as opened:
            original_a = opened.convert("RGB").copy()
        with Image.open(target["image_b"]) as opened:
            original_b = opened.convert("RGB").copy()

        chosen_components = []
        used_words = set()
        used_pairs = set()
        attempts = 0
        while len(chosen_components) < 3 and attempts < 200:
            attempts += 1
            donor = rng.choice(donor_pool)
            if donor["pair_id"] in used_pairs:
                continue
            options = [item for item in donor["shared"] if item[0] not in used_words]
            if not options:
                continue
            shared = rng.choice(options)
            word, crop_a, crop_b = _crop_word(donor, shared)
            chosen_components.append((word, crop_a, crop_b, donor["pair_id"]))
            used_words.add(word)
            used_pairs.add(donor["pair_id"])
        if len(chosen_components) < 3:
            continue

        variants = []
        variant_summary = {}
        for count in (1, 2, 3):
            components = [(w, ca, cb) for w, ca, cb, _pid in chosen_components[:count]]
            aug_a, aug_b, meta = _inject(original_a, original_b, components)
            for item, source in zip(meta, chosen_components[:count]):
                item["donor_pair_id"] = source[3]
            variants.append((f"{count} SHARED INJECTED REGION{'S' if count > 1 else ''} | red = injected", aug_a, aug_b, meta))
            variant_summary[str(count)] = meta

        sheet = _sheet(
            original_a, original_b, variants,
            f"target={target['pair_id']} | VISUAL-ONLY estimated transcript-to-image injection preview",
        )
        filename = previews / f"preview_{created + 1:03d}_{target['pair_id']}.png"
        sheet.save(filename)
        summary_items.append({
            "target_pair_id": target["pair_id"],
            "preview": str(filename),
            "components": variant_summary,
        })
        created += 1
        if created >= args.num_pairs:
            break

    if created == 0:
        raise RuntimeError("Could not construct any 1/2/3-component visual injection previews from the selected real pairs.")

    summary = {
        "dataset": str(root),
        "created_previews": created,
        "requested_previews": args.num_pairs,
        "source_pairs_scanned": len(pairs),
        "donor_pairs_with_shared_words": len(donors),
        "method": "visual-only exact-shared-transcript-word injection with estimated RTL horizontal crop location",
        "bbox_required": False,
        "training_safe": False,
        "warning": "Inspection only. Estimated word locations are not ground-truth bbox annotations and must not be used for training.",
        "items": summary_items,
    }
    (output / "preview_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created {created} real multi-injection visual previews")
    print(f"Preview directory: {previews}")
    print(f"Summary: {output / 'preview_summary.json'}")
    print("NOTE: visual inspection only; bbox annotations are not used by this fallback.")


if __name__ == "__main__":
    main()
