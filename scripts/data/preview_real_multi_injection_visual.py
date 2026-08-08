#!/usr/bin/env python3
"""Preview 1/2/3 shared real-handwriting full-height strip injections.

VISUAL-ONLY inspection tool. It does not modify the training dataset.

For every injected component, the donor crop is a vertical strip spanning the
FULL HEIGHT of the donor line. Only x is cropped: from the estimated left edge
of the first selected subword through the estimated right edge of the last
selected subword. The target replacement is also a full-height vertical strip.

The target transcript is edited at exactly the target token span replaced in
the image. The preview and a sidecar text file show the complete text after
injection for lines A and B. A/B use two different real handwriting crops for
the same canonical shared donor span.

The horizontal spans are estimated from transcript position because the current
page-level bbox-to-line mapping is unresolved. Therefore this script is for
visual inspection only and the estimated spans must not be used as training
annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_RUN = re.compile(r"[\u0600-\u06ff]+")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (root / path, root.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_marks(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return _ARABIC_DIACRITICS.sub("", text.replace("ـ", ""))


def _canonical(text: str) -> str:
    return " ".join(_strip_marks(text).split())


def _token_records(text: str) -> list[dict]:
    records = []
    for match in _ARABIC_RUN.finditer(text):
        raw = match.group(0)
        canonical = _canonical(raw)
        if not canonical:
            continue
        records.append(
            {
                "raw": raw,
                "canonical": canonical,
                "start": int(match.start()),
                "end": int(match.end()),
            }
        )
    return records


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
        [
            arr[:band].reshape(-1, 3),
            arr[-band:].reshape(-1, 3),
            arr[:, :band].reshape(-1, 3),
            arr[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(int(round(v)) for v in np.median(samples, axis=0))


def _foreground_x_bounds(image: Image.Image) -> tuple[int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate(
        [gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()]
    )
    bg = float(np.median(border)) if border.size else 255.0
    threshold = max(0.0, bg - 22.0)
    ys, xs = np.where(gray < threshold)
    if xs.size < 8:
        return 0, image.width
    pad = max(2, int(round(image.width * 0.01)))
    return max(0, int(xs.min()) - pad), min(image.width, int(xs.max()) + 1 + pad)


def _span_rect(
    records: list[dict],
    start: int,
    size: int,
    image: Image.Image,
) -> tuple[int, int, int, int]:
    """Estimated RTL token span, always returned as a FULL-HEIGHT strip."""
    if not records or start < 0 or size <= 0 or start + size > len(records):
        raise ValueError("invalid transcript span")
    ink_x0, ink_x1 = _foreground_x_bounds(image)
    ink_width = max(1, ink_x1 - ink_x0)
    weights = [max(1.0, float(len(item["canonical"]))) for item in records]
    gap = 0.45
    total = sum(weights) + gap * max(0, len(weights) - 1)

    before = sum(weights[:start]) + gap * start
    end_index = start + size
    through = sum(weights[:end_index]) + gap * max(0, end_index - 1)

    # Arabic transcript order is RTL: token index 0 starts at the right.
    right = ink_x1 - int(round((before / total) * ink_width))
    left = ink_x1 - int(round((through / total) * ink_width))
    pad = max(2, int(round(ink_width * 0.006)))
    left = max(0, min(image.width - 1, left - pad))
    right = max(left + 1, min(image.width, right + pad))
    return left, 0, right, image.height


def _raw_span(text: str, records: list[dict], start: int, size: int) -> str:
    first = records[start]
    last = records[start + size - 1]
    return text[first["start"] : last["end"]]


def _shared_runs(text_a: str, text_b: str, max_tokens: int = 3, min_chars: int = 4) -> list[dict]:
    a = _token_records(text_a)
    b = _token_records(text_b)
    result = []
    seen = set()
    for size in range(max_tokens, 0, -1):
        b_index: dict[str, list[int]] = {}
        for j in range(0, len(b) - size + 1):
            phrase = " ".join(item["canonical"] for item in b[j : j + size])
            b_index.setdefault(phrase, []).append(j)
        for i in range(0, len(a) - size + 1):
            phrase = " ".join(item["canonical"] for item in a[i : i + size])
            compact = phrase.replace(" ", "")
            if len(compact) < min_chars or phrase in seen or phrase not in b_index:
                continue
            j = b_index[phrase][0]
            result.append(
                {
                    "canonical_text": phrase,
                    "start_a": i,
                    "start_b": j,
                    "size": size,
                    "raw_a": _raw_span(text_a, a, i, size),
                    "raw_b": _raw_span(text_b, b, j, size),
                }
            )
            seen.add(phrase)
    return result


def _load_pair(root: Path, record: dict, fallback: int) -> dict | None:
    side_a = _side_paths(root, record, "A")
    side_b = _side_paths(root, record, "B")
    if side_a is None or side_b is None:
        return None
    text_a = side_a[1].read_text(encoding="utf-8").strip()
    text_b = side_b[1].read_text(encoding="utf-8").strip()
    records_a = _token_records(text_a)
    records_b = _token_records(text_b)
    if not records_a or not records_b:
        return None
    return {
        "pair_id": _pair_id(record, fallback),
        "image_a": side_a[0],
        "image_b": side_b[0],
        "text_a": text_a,
        "text_b": text_b,
        "records_a": records_a,
        "records_b": records_b,
        "shared": _shared_runs(text_a, text_b),
    }


def _crop_component(pair: dict, shared: dict) -> dict:
    with Image.open(pair["image_a"]) as opened:
        image_a = opened.convert("RGB")
    with Image.open(pair["image_b"]) as opened:
        image_b = opened.convert("RGB")

    rect_a = _span_rect(pair["records_a"], shared["start_a"], shared["size"], image_a)
    rect_b = _span_rect(pair["records_b"], shared["start_b"], shared["size"], image_b)
    return {
        "canonical_text": shared["canonical_text"],
        "text_a": shared["raw_a"],
        "text_b": shared["raw_b"],
        "size": int(shared["size"]),
        "crop_a": image_a.crop(rect_a),
        "crop_b": image_b.crop(rect_b),
        "donor_rect_a": rect_a,
        "donor_rect_b": rect_b,
        "donor_pair_id": pair["pair_id"],
    }


def _target_starts(records: list[dict], components: list[dict]) -> list[int] | None:
    n = len(components)
    centers = {1: [0.50], 2: [0.28, 0.72], 3: [0.18, 0.50, 0.82]}[n]
    occupied: set[int] = set()
    starts: list[int] = []
    for center, component in zip(centers, components):
        size = int(component["size"])
        if size > len(records):
            return None
        candidates = list(range(0, len(records) - size + 1))
        desired = center * max(0, len(records) - size)
        candidates.sort(key=lambda value: abs(value - desired))
        chosen = None
        for start in candidates:
            covered = set(range(start, start + size))
            if covered.isdisjoint(occupied):
                chosen = start
                occupied.update(covered)
                break
        if chosen is None:
            return None
        starts.append(chosen)
    return starts


def _replace_text(text: str, records: list[dict], replacements: list[tuple[int, int, str]]) -> str:
    edits = []
    for start, size, replacement in replacements:
        first = records[start]
        last = records[start + size - 1]
        edits.append((int(first["start"]), int(last["end"]), replacement))
    output = text
    for char_start, char_end, replacement in sorted(edits, key=lambda item: item[0], reverse=True):
        output = output[:char_start] + replacement + output[char_end:]
    return output


def _inject_side(
    base: Image.Image,
    text: str,
    records: list[dict],
    components: list[dict],
    starts: list[int],
    side: str,
) -> tuple[Image.Image, str, list[dict]]:
    out = base.convert("RGB").copy()
    replacements = []
    metadata = []
    for component, start in zip(components, starts):
        size = int(component["size"])
        target_rect = _span_rect(records, start, size, out)
        x0, _y0, x1, _y1 = target_rect
        target_width = max(1, x1 - x0)
        donor_crop = component[f"crop_{side}"].convert("RGB")

        # Required injection rule: donor and target are FULL-HEIGHT vertical strips.
        fitted = donor_crop.resize((target_width, out.height), Image.Resampling.LANCZOS)
        ImageDraw.Draw(out).rectangle(target_rect, fill=_background(out))
        out.paste(fitted, (x0, 0))
        ImageDraw.Draw(out).rectangle(target_rect, outline=(220, 30, 30), width=3)

        replacement = component[f"text_{side}"]
        old_text = _raw_span(text, records, start, size)
        replacements.append((start, size, replacement))
        metadata.append(
            {
                "canonical_shared_text": component["canonical_text"],
                "donor_pair_id": component["donor_pair_id"],
                "donor_full_height_strip": list(component[f"donor_rect_{side}"]),
                "target_full_height_strip": list(target_rect),
                "target_text_before": old_text,
                "target_text_after": replacement,
                "target_token_start": int(start),
                "target_token_count": int(size),
            }
        )

    augmented_text = _replace_text(text, records, replacements)
    return out, augmented_text, metadata


def _inject_pair(
    target: dict,
    original_a: Image.Image,
    original_b: Image.Image,
    components: list[dict],
) -> tuple[Image.Image, Image.Image, str, str, list[dict]] | None:
    starts_a = _target_starts(target["records_a"], components)
    starts_b = _target_starts(target["records_b"], components)
    if starts_a is None or starts_b is None:
        return None

    aug_a, text_a, meta_a = _inject_side(
        original_a, target["text_a"], target["records_a"], components, starts_a, "a"
    )
    aug_b, text_b, meta_b = _inject_side(
        original_b, target["text_b"], target["records_b"], components, starts_b, "b"
    )
    combined = []
    for index, component in enumerate(components):
        combined.append(
            {
                "component": index + 1,
                "canonical_shared_text": component["canonical_text"],
                "donor_pair_id": component["donor_pair_id"],
                "A": meta_a[index],
                "B": meta_b[index],
            }
        )
    return aug_a, aug_b, text_a, text_b, combined


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_rtl_text(draw: ImageDraw.ImageDraw, right: int, y: int, text: str, font) -> None:
    rendered = text if len(text) <= 150 else text[:147] + "..."
    try:
        bbox = draw.textbbox((0, 0), rendered, font=font, direction="rtl")
        width = max(1, bbox[2] - bbox[0])
        draw.text((max(8, right - width), y), rendered, fill="black", font=font, direction="rtl")
    except Exception:
        try:
            bbox = draw.textbbox((0, 0), rendered, font=font)
            width = max(1, bbox[2] - bbox[0])
            draw.text((max(8, right - width), y), rendered, fill="black", font=font)
        except Exception:
            pass


def _sheet(
    original_a: Image.Image,
    original_b: Image.Image,
    original_text_a: str,
    original_text_b: str,
    variants: list[tuple[str, Image.Image, Image.Image, str, str, list[dict]]],
    title: str,
) -> Image.Image:
    panel_w, panel_h = 1000, 180
    header_h = 50
    info_h = 86
    rows = 1 + len(variants)
    sheet = Image.new("RGB", (panel_w * 2, header_h + rows * (panel_h + info_h)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = _font(18)
    text_font = _font(18)
    draw.text((12, 12), title, fill="black", font=label_font)

    entries = [
        ("ORIGINAL", original_a, original_b, original_text_a, original_text_b, [])
    ] + variants
    y = header_h
    for label, image_a, image_b, text_a, text_b, meta in entries:
        injected = ""
        if meta:
            injected = " | injected: " + " ; ".join(item["canonical_shared_text"] for item in meta)
        draw.text((12, y + 5), f"{label}{injected}", fill="black", font=label_font)
        draw.text((12, y + 32), "A text after:", fill="black", font=label_font)
        draw.text((panel_w + 12, y + 32), "B text after:", fill="black", font=label_font)
        _draw_rtl_text(draw, panel_w - 12, y + 57, text_a, text_font)
        _draw_rtl_text(draw, panel_w * 2 - 12, y + 57, text_b, text_font)
        sheet.paste(_fit(image_a, panel_w, panel_h), (0, y + info_h))
        sheet.paste(_fit(image_b, panel_w, panel_h), (panel_w, y + info_h))
        y += panel_h + info_h
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
    raw = [
        record
        for record in _read_manifest(manifest)
        if not labels or str(record.get("label_type", "")) in labels
    ]
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
        raise RuntimeError("No donor pairs with an exact shared Arabic span were found.")

    created = 0
    summary_items = []
    rng.shuffle(pairs)
    for target in pairs:
        donor_pool = [donor for donor in donors if donor["pair_id"] != target["pair_id"]]
        if len(donor_pool) < 3:
            break

        with Image.open(target["image_a"]) as opened:
            original_a = opened.convert("RGB").copy()
        with Image.open(target["image_b"]) as opened:
            original_b = opened.convert("RGB").copy()

        chosen_components = []
        used_texts = set()
        used_pairs = set()
        attempts = 0
        max_target_tokens = min(len(target["records_a"]), len(target["records_b"]))
        while len(chosen_components) < 3 and attempts < 300:
            attempts += 1
            donor = rng.choice(donor_pool)
            if donor["pair_id"] in used_pairs:
                continue
            options = [
                item
                for item in donor["shared"]
                if item["canonical_text"] not in used_texts
                and int(item["size"]) <= max_target_tokens
            ]
            if not options:
                continue
            shared = rng.choice(options)
            component = _crop_component(donor, shared)
            chosen_components.append(component)
            used_texts.add(component["canonical_text"])
            used_pairs.add(donor["pair_id"])

        if len(chosen_components) < 3:
            continue

        variants = []
        variant_summary = {}
        text_sidecar = [
            "ORIGINAL",
            f"A: {target['text_a']}",
            f"B: {target['text_b']}",
            "",
        ]
        valid = True
        for count in (1, 2, 3):
            components = chosen_components[:count]
            injected = _inject_pair(target, original_a, original_b, components)
            if injected is None:
                valid = False
                break
            aug_a, aug_b, text_a, text_b, meta = injected
            label = f"{count} SHARED FULL-HEIGHT INJECTED STRIP{'S' if count > 1 else ''} | red = replaced strip"
            variants.append((label, aug_a, aug_b, text_a, text_b, meta))
            variant_summary[str(count)] = {
                "text_A_after": text_a,
                "text_B_after": text_b,
                "components": meta,
            }
            text_sidecar.extend(
                [
                    f"{count} INJECTED REGION{'S' if count > 1 else ''}",
                    f"A: {text_a}",
                    f"B: {text_b}",
                    "",
                ]
            )
        if not valid:
            continue

        sheet = _sheet(
            original_a,
            original_b,
            target["text_a"],
            target["text_b"],
            variants,
            f"target={target['pair_id']} | full-height vertical-strip injection | VISUAL ONLY",
        )
        stem = f"preview_{created + 1:03d}_{target['pair_id']}"
        filename = previews / f"{stem}.png"
        text_filename = previews / f"{stem}.txt"
        sheet.save(filename)
        text_filename.write_text("\n".join(text_sidecar), encoding="utf-8")
        summary_items.append(
            {
                "target_pair_id": target["pair_id"],
                "preview": str(filename),
                "text_sidecar": str(text_filename),
                "original_text_A": target["text_a"],
                "original_text_B": target["text_b"],
                "variants": variant_summary,
            }
        )
        created += 1
        if created >= args.num_pairs:
            break

    if created == 0:
        raise RuntimeError(
            "Could not construct any 1/2/3-component full-height injection previews "
            "from the selected real pairs."
        )

    summary = {
        "dataset": str(root),
        "created_previews": created,
        "requested_previews": args.num_pairs,
        "source_pairs_scanned": len(pairs),
        "donor_pairs_with_shared_spans": len(donors),
        "method": "visual-only full-height vertical-strip injection with text-synchronized target-span replacement",
        "crop_rule": "y=0..full line height; x=first selected subword edge..last selected subword edge",
        "text_rule": "replace exactly the target transcript span whose full-height strip was replaced",
        "bbox_required": False,
        "training_safe": False,
        "warning": "Inspection only. Horizontal subword boundaries are estimated from RTL transcript position until true bbox-to-line mapping is fixed.",
        "items": summary_items,
    }
    (output / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {created} real multi-injection visual previews")
    print(f"Preview directory: {previews}")
    print(f"Summary: {output / 'preview_summary.json'}")
    print("Each preview has a .txt sidecar containing the full text after each injection.")
    print("NOTE: visual inspection only; true bbox mapping is still required before training.")


if __name__ == "__main__":
    main()
