#!/usr/bin/env python3
"""Preview 1, 2, and 3 shared real-handwriting injections on real line pairs.

This is an inspection tool. It first uses the canonical line-level bbox loader.
When the dataset only exposes page-level debug/bboxes.json and that loader cannot
resolve a line, it uses a preview-only fallback: page boxes are grouped into
text lines by vertical position and geometrically remapped onto the foreground
extent of the requested line image. The fallback is deliberately recorded in
metadata and must be visually reviewed before being used for training.

For every selected pair the preview shows original A/B and aligned injections
with 1, 2, and 3 separated shared components. For each component the same
annotated text is injected into A and B using crops from two distinct real donor
line images. Red rectangles identify injected boxes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from Evaluation import real_subword_box_json as boxjson
from Evaluation.real_subword_box_metrics import SubwordBox
from scripts.data import augment_real_bbox_dataset as aug


_LINE_RE = re.compile(r"line[_-]?0*(\d+)", re.IGNORECASE)


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


def _line_number(path: Path) -> int:
    match = _LINE_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot extract line number from {path.name}")
    return int(match.group(1))


def _side_root(image_path: Path) -> Path:
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    raise ValueError(f"Cannot locate A/B side root for {image_path}")


def _foreground_bbox(image: Image.Image) -> tuple[float, float, float, float]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    # Robustly find dark manuscript foreground while ignoring an almost-white page.
    threshold = min(245, int(np.percentile(gray, 80)))
    mask = gray < threshold
    ys, xs = np.where(mask)
    if xs.size < 8 or ys.size < 8:
        return (0.0, 0.0, float(image.width), float(image.height))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int(round(0.01 * image.width)))
    pad_y = max(1, int(round(0.05 * image.height)))
    return (
        float(max(0, x0 - pad_x)),
        float(max(0, y0 - pad_y)),
        float(min(image.width, x1 + pad_x)),
        float(min(image.height, y1 + pad_y)),
    )


def _kmeans_y(boxes: list[SubwordBox], k: int) -> list[list[SubwordBox]]:
    if k <= 0 or len(boxes) < k:
        raise ValueError(f"Cannot cluster {len(boxes)} boxes into {k} lines")
    centers = np.asarray([(box.y0 + box.y1) * 0.5 for box in boxes], dtype=np.float64)
    ordered = np.sort(centers)
    initial_positions = np.linspace(0, len(ordered) - 1, k)
    centroids = np.asarray([ordered[int(round(pos))] for pos in initial_positions], dtype=np.float64)

    assignments = np.zeros(len(boxes), dtype=np.int32)
    for _ in range(80):
        distances = np.abs(centers[:, None] - centroids[None, :])
        new_assignments = distances.argmin(axis=1).astype(np.int32)
        new_centroids = centroids.copy()
        for index in range(k):
            members = centers[new_assignments == index]
            if members.size:
                new_centroids[index] = float(members.mean())
        if np.array_equal(new_assignments, assignments) and np.allclose(new_centroids, centroids):
            assignments = new_assignments
            centroids = new_centroids
            break
        assignments = new_assignments
        centroids = new_centroids

    cluster_pairs: list[tuple[float, list[SubwordBox]]] = []
    for index in range(k):
        members = [box for box, assignment in zip(boxes, assignments) if int(assignment) == index]
        if not members:
            continue
        center = float(np.mean([(box.y0 + box.y1) * 0.5 for box in members]))
        cluster_pairs.append((center, members))
    if len(cluster_pairs) != k:
        raise ValueError(f"Vertical clustering produced {len(cluster_pairs)} non-empty lines, expected {k}")
    return [members for _center, members in sorted(cluster_pairs, key=lambda item: item[0])]


def _raw_page_boxes(json_path: Path) -> list[SubwordBox]:
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    records = payload if isinstance(payload, list) else []
    boxes: list[SubwordBox] = []
    for row, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        box = boxjson._box_from_record(record, row)  # reuse the canonical generic schema parser
        if box is not None and str(box.text or "").strip():
            boxes.append(box)
    if not boxes:
        raise ValueError(f"No text-bearing boxes could be parsed from {json_path}")
    return boxes


def _fallback_page_boxes_to_line(image_path: Path) -> tuple[list[SubwordBox], Path, str]:
    side_root = _side_root(image_path)
    candidates = [
        side_root / "debug" / "bboxes.json",
        side_root / "debug" / "bbox.json",
        side_root / "bboxes.json",
        side_root / "bbox.json",
    ]
    json_path = next((path for path in candidates if path.is_file()), None)
    if json_path is None:
        raise ValueError(f"No page-level bbox JSON found under {side_root}")

    page_boxes = _raw_page_boxes(json_path)
    line_images = sorted(
        (path for path in (side_root / "linesImages").glob("line_*.png") if path.is_file()),
        key=_line_number,
    )
    if not line_images:
        raise ValueError(f"No line images found under {side_root / 'linesImages'}")
    clusters = _kmeans_y(page_boxes, len(line_images))

    requested_number = _line_number(image_path)
    number_to_position = {_line_number(path): index for index, path in enumerate(line_images)}
    if requested_number not in number_to_position:
        raise ValueError(f"{image_path.name} is not present in {side_root / 'linesImages'}")
    source_boxes = clusters[number_to_position[requested_number]]

    src_x0 = min(box.x0 for box in source_boxes)
    src_y0 = min(box.y0 for box in source_boxes)
    src_x1 = max(box.x1 for box in source_boxes)
    src_y1 = max(box.y1 for box in source_boxes)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        raise ValueError("Degenerate page-level line bbox extent")

    with Image.open(image_path) as opened:
        line_image = opened.convert("RGB")
        dst_x0, dst_y0, dst_x1, dst_y1 = _foreground_bbox(line_image)
        width, height = line_image.size

    scale_x = (dst_x1 - dst_x0) / max(1e-6, src_x1 - src_x0)
    scale_y = (dst_y1 - dst_y0) / max(1e-6, src_y1 - src_y0)
    transformed: list[SubwordBox] = []
    for box in source_boxes:
        x0 = dst_x0 + (box.x0 - src_x0) * scale_x
        y0 = dst_y0 + (box.y0 - src_y0) * scale_y
        x1 = dst_x0 + (box.x1 - src_x0) * scale_x
        y1 = dst_y0 + (box.y1 - src_y0) * scale_y
        x0, x1 = sorted((max(0.0, min(float(width), x0)), max(0.0, min(float(width), x1))))
        y0, y1 = sorted((max(0.0, min(float(height), y0)), max(0.0, min(float(height), y1))))
        if x1 > x0 and y1 > y0:
            transformed.append(SubwordBox(box.text, x0, y0, x1, y1, source_row=box.source_row))
    if not transformed:
        raise ValueError(f"No page boxes survived remapping for {image_path}")
    return transformed, json_path, "preview_page_bbox_vertical_cluster+foreground_remap"


def _load_source_line(dataset_root: Path, pair_id: str, side_name: str, side: dict) -> tuple[aug.SourceLine, str]:
    image_path = _resolve(dataset_root, side["line_image_path"])
    text_path = _resolve(dataset_root, side["text_original_path"])
    text = text_path.read_text(encoding="utf-8").strip()

    annotations = boxjson.load_json_annotations(image_path)
    if annotations.status == "ok" and annotations.boxes:
        boxes = list(annotations.boxes)
        annotation_path = Path(annotations.workbook)
        mapping_method = "canonical_line_bbox_loader"
    else:
        boxes, annotation_path, mapping_method = _fallback_page_boxes_to_line(image_path)

    mapped = aug._map_boxes_strict(text, boxes)
    with Image.open(image_path) as opened:
        width, height = opened.size
    for item in mapped:
        box = item.box
        if not (0 <= box.x0 < box.x1 <= width and 0 <= box.y0 < box.y1 <= height):
            raise ValueError(f"Mapped bbox outside {image_path.name}: {box}")
    return (
        aug.SourceLine(
            image_path=image_path,
            text_path=text_path,
            text=text,
            boxes=mapped,
            annotation_path=str(annotation_path),
            side=side_name,
            pair_id=str(pair_id),
            line_idx=int(side.get("line_idx", _line_number(image_path))),
        ),
        mapping_method,
    )


def _current_target_candidates(line: aug.AugmentedLine, donor: aug.DonorRun, width_ratio_min: float, width_ratio_max: float) -> list[int]:
    mapped = aug._map_boxes_strict(line.text, line.boxes)
    donor_width = max(1.0, donor.rect[2] - donor.rect[0])
    blocked = set(int(index) for index in line.injected_box_indices)
    blocked |= {index - 1 for index in tuple(blocked)} | {index + 1 for index in tuple(blocked)}
    candidates: list[int] = []
    for start in range(0, len(mapped) - donor.size + 1):
        indices = set(range(start, start + donor.size))
        if indices & blocked:
            continue
        selected = mapped[start : start + donor.size]
        rect = aug._rect_for_boxes(item.box for item in selected)
        target_width = max(1.0, rect[2] - rect[0])
        ratio = donor_width / target_width
        if not (width_ratio_min <= ratio <= width_ratio_max):
            continue
        exact_text = line.text[selected[0].text_start : selected[-1].text_end].strip()
        if aug._canonical_text(exact_text) == donor.canonical_text:
            continue
        candidates.append(start)
    return candidates


def _inject_into_current(
    rng: random.Random,
    line: aug.AugmentedLine,
    donor: aug.DonorRun,
    width_ratio_min: float,
    width_ratio_max: float,
    padding: int,
) -> bool:
    mapped = aug._map_boxes_strict(line.text, line.boxes)
    candidates = _current_target_candidates(line, donor, width_ratio_min, width_ratio_max)
    if not candidates:
        return False
    target_start = rng.choice(candidates)
    target_items = mapped[target_start : target_start + donor.size]
    target_rect = aug._expand_rect(
        aug._rect_for_boxes(item.box for item in target_items),
        line.image.width,
        line.image.height,
        padding,
    )

    with Image.open(donor.line.image_path) as opened:
        donor_image = opened.convert("RGB")
    donor_crop_rect = aug._expand_rect(donor.rect, donor_image.width, donor_image.height, padding)
    donor_crop = donor_image.crop(donor_crop_rect)
    target_width = max(1, target_rect[2] - target_rect[0])
    target_height = max(1, target_rect[3] - target_rect[1])
    fitted = ImageOps.contain(donor_crop, (target_width, target_height), Image.Resampling.LANCZOS)
    paste_x = target_rect[0] + (target_width - fitted.width) // 2
    paste_y = target_rect[1] + (target_height - fitted.height) // 2

    draw = ImageDraw.Draw(line.image)
    draw.rectangle(target_rect, fill=aug._image_background(line.image))
    line.image.paste(fitted, (paste_x, paste_y))

    scale_x = fitted.width / max(1, donor_crop.width)
    scale_y = fitted.height / max(1, donor_crop.height)
    injected_boxes = aug._transform_donor_boxes(
        donor, donor_crop_rect, paste_x, paste_y, scale_x, scale_y
    )

    text_start = target_items[0].text_start
    text_end = target_items[-1].text_end
    line.text = line.text[:text_start] + donor.text.strip() + line.text[text_end:]
    before = line.boxes[:target_start]
    after = line.boxes[target_start + donor.size :]
    line.boxes = before + injected_boxes + after
    new_indices = set(range(len(before), len(before) + len(injected_boxes)))
    line.injected_box_indices = sorted(set(line.injected_box_indices) | new_indices)
    line.operations.append(
        {
            "type": "bbox_injection_preview",
            "component_number": 1 + sum(op.get("type") == "bbox_injection_preview" for op in line.operations),
            "donor_image": str(donor.line.image_path),
            "donor_text": donor.text,
            "donor_canonical_text": donor.canonical_text,
            "target_rect": [int(value) for value in target_rect],
            "donor_crop_rect": [int(value) for value in donor_crop_rect],
        }
    )
    aug._validate_augmented_line(line)
    return True


def _choose_aligned_donors(
    rng: random.Random,
    donor_index: dict[tuple[int, str], list[aug.DonorRun]],
    source_a: aug.SourceLine,
    source_b: aug.SourceLine,
    used_texts: set[str],
) -> tuple[aug.DonorRun, aug.DonorRun] | None:
    keys = [key for key in donor_index if key[1] not in used_texts]
    rng.shuffle(keys)
    for key in keys:
        eligible = [
            run for run in donor_index[key]
            if run.line.image_path not in {source_a.image_path, source_b.image_path}
        ]
        if len(eligible) < 2:
            continue
        rng.shuffle(eligible)
        for first in eligible:
            for second in eligible:
                if first.line.image_path != second.line.image_path:
                    return first, second
    return None


def _make_variant(
    rng: random.Random,
    source_a: aug.SourceLine,
    source_b: aug.SourceLine,
    donor_index: dict[tuple[int, str], list[aug.DonorRun]],
    components: int,
    args,
) -> tuple[aug.AugmentedLine, aug.AugmentedLine, list[dict]] | None:
    line_a = aug._copy_line(source_a)
    line_b = aug._copy_line(source_b)
    used_texts: set[str] = set()
    metadata: list[dict] = []

    for component_index in range(1, components + 1):
        success = False
        for _attempt in range(args.max_component_attempts):
            donors = _choose_aligned_donors(rng, donor_index, source_a, source_b, used_texts)
            if donors is None:
                break
            # Work on copies so a failed A/B placement cannot partially mutate the variant.
            trial_a = aug.AugmentedLine(
                image=line_a.image.copy(), text=line_a.text, boxes=list(line_a.boxes),
                injected_box_indices=list(line_a.injected_box_indices), operations=list(line_a.operations),
            )
            trial_b = aug.AugmentedLine(
                image=line_b.image.copy(), text=line_b.text, boxes=list(line_b.boxes),
                injected_box_indices=list(line_b.injected_box_indices), operations=list(line_b.operations),
            )
            ok_a = _inject_into_current(
                rng, trial_a, donors[0], args.injection_width_ratio_min,
                args.injection_width_ratio_max, args.injection_padding,
            )
            ok_b = _inject_into_current(
                rng, trial_b, donors[1], args.injection_width_ratio_min,
                args.injection_width_ratio_max, args.injection_padding,
            )
            if not (ok_a and ok_b):
                continue
            line_a, line_b = trial_a, trial_b
            used_texts.add(donors[0].canonical_text)
            metadata.append(
                {
                    "component": component_index,
                    "shared_text": donors[0].canonical_text,
                    "donor_A": str(donors[0].line.image_path),
                    "donor_B": str(donors[1].line.image_path),
                }
            )
            success = True
            break
        if not success:
            return None
    return line_a, line_b, metadata


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def _boxed_image(line: aug.AugmentedLine) -> Image.Image:
    return aug._draw_boxes(line.image, line.boxes, set(line.injected_box_indices))


def _save_sheet(
    output: Path,
    pair_id: str,
    source_a: aug.SourceLine,
    source_b: aug.SourceLine,
    variants: dict[int, tuple[aug.AugmentedLine, aug.AugmentedLine, list[dict]]],
    mapping_a: str,
    mapping_b: str,
) -> None:
    panel_w, panel_h = 950, 210
    label_h, header_h = 42, 72
    rows = 1 + len(variants)
    sheet = Image.new("RGB", (panel_w * 2, header_h + rows * (panel_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{pair_id} | REAL MULTI-INJECTION PREVIEW | red = injected real handwriting", fill="black")
    draw.text((12, 31), f"A bbox mapping: {mapping_a}", fill="black")
    draw.text((12, 49), f"B bbox mapping: {mapping_b}", fill="black")

    original_a = aug._draw_boxes(
        Image.open(source_a.image_path).convert("RGB"), [item.box for item in source_a.boxes], set()
    )
    original_b = aug._draw_boxes(
        Image.open(source_b.image_path).convert("RGB"), [item.box for item in source_b.boxes], set()
    )
    rows_data: list[tuple[str, Image.Image, Image.Image]] = [("ORIGINAL", original_a, original_b)]
    for count in sorted(variants):
        line_a, line_b, metadata = variants[count]
        texts = " | ".join(str(item["shared_text"]) for item in metadata)
        rows_data.append((f"{count} SHARED INJECTED REGION(S) | {texts}", _boxed_image(line_a), _boxed_image(line_b)))

    y = header_h
    for label, image_a, image_b in rows_data:
        draw.text((12, y + 8), label, fill="black")
        draw.text((panel_w + 12, y + 8), "A" if label == "ORIGINAL" else "B", fill="black")
        sheet.paste(_fit_panel(image_a, panel_w, panel_h), (0, y + label_h))
        sheet.paste(_fit_panel(image_b, panel_w, panel_h), (panel_w, y + label_h))
        y += panel_h + label_h
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument("--source-pairs", type=int, default=300)
    parser.add_argument("--max-ngram-boxes", type=int, default=3)
    parser.add_argument("--injection-min-chars", type=int, default=4)
    parser.add_argument("--injection-max-chars", type=int, default=28)
    parser.add_argument("--injection-width-ratio-min", type=float, default=0.55)
    parser.add_argument("--injection-width-ratio-max", type=float, default=1.80)
    parser.add_argument("--injection-padding", type=int, default=3)
    parser.add_argument("--max-component-attempts", type=int, default=120)
    parser.add_argument("--max-pair-attempts", type=int, default=300)
    args = parser.parse_args()

    if args.num_pairs <= 0:
        raise SystemExit("--num-pairs must be positive")
    root = args.data_dir.expanduser().resolve()
    manifest = root / "dataset_manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    output = args.output_dir.expanduser().resolve()
    previews = output / "previews"
    previews.mkdir(parents=True, exist_ok=True)

    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    records = [
        record for record in _read_manifest(manifest)
        if not labels or str(record.get("label_type", "")) in labels
    ]
    rng = random.Random(args.seed)
    rng.shuffle(records)

    line_cache: dict[Path, tuple[aug.SourceLine, str]] = {}
    usable: list[tuple[dict, aug.SourceLine, aug.SourceLine, str, str]] = []
    failures: dict[str, int] = {}

    for record in records:
        if len(usable) >= args.source_pairs:
            break
        pair_id = str(record.get("pair_id", len(usable) + 1))
        try:
            loaded = []
            methods = []
            for side_name in ("A", "B"):
                side = record[side_name]
                image_path = _resolve(root, side["line_image_path"])
                if image_path not in line_cache:
                    line_cache[image_path] = _load_source_line(root, pair_id, side_name, side)
                line, method = line_cache[image_path]
                loaded.append(line)
                methods.append(method)
            usable.append((record, loaded[0], loaded[1], methods[0], methods[1]))
        except Exception as exc:
            key = f"{type(exc).__name__}: {exc}"
            failures[key] = failures.get(key, 0) + 1

    if not usable:
        sample = "\n".join(f"  {count} x {reason}" for reason, count in list(failures.items())[:8])
        raise RuntimeError(f"No usable real pairs for injection preview.\n{sample}")

    donor_index = aug._build_donor_index(
        [line for line, _method in line_cache.values()],
        args.max_ngram_boxes,
        args.injection_min_chars,
        args.injection_max_chars,
    )
    if not donor_index:
        raise RuntimeError("No repeated real bbox text spans are available for aligned injection preview")

    created = 0
    metadata_rows = []
    candidates = list(usable)
    rng.shuffle(candidates)
    for record, source_a, source_b, method_a, method_b in candidates:
        if created >= args.num_pairs:
            break
        variants = {}
        local_rng = random.Random(rng.randrange(2**31))
        failed = False
        for count in (1, 2, 3):
            variant = _make_variant(local_rng, source_a, source_b, donor_index, count, args)
            if variant is None:
                failed = True
                break
            variants[count] = variant
        if failed:
            continue

        created += 1
        pair_id = str(record.get("pair_id", f"pair_{created:03d}"))
        preview_path = previews / f"injection_{created:03d}.png"
        _save_sheet(preview_path, pair_id, source_a, source_b, variants, method_a, method_b)
        item = {
            "pair_id": pair_id,
            "source_A": str(source_a.image_path),
            "source_B": str(source_b.image_path),
            "bbox_mapping_A": method_a,
            "bbox_mapping_B": method_b,
            "preview": str(preview_path),
            "variants": {
                str(count): metadata for count, (_a, _b, metadata) in variants.items()
            },
        }
        metadata_rows.append(item)
        (preview_path.with_suffix(".json")).write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if created == 0:
        raise RuntimeError(
            "Bbox/text lines were found, but no pair could accept all 1/2/3 separated injections. "
            "Inspect the mapping failures or widen injection width ratios for preview only."
        )

    summary = {
        "preview_only": True,
        "training_generator_modified": False,
        "dataset": str(root),
        "requested_previews": args.num_pairs,
        "created_previews": created,
        "usable_source_pairs": len(usable),
        "repeated_donor_spans": len(donor_index),
        "components_shown": [1, 2, 3],
        "policy": "same text in A/B; distinct real donor handwriting images; separated target regions",
        "page_bbox_fallback_warning": (
            "When canonical line bbox lookup fails, page bboxes are vertically clustered and remapped "
            "to line-image foreground for visual inspection only. Do not use this fallback for training "
            "until the previews verify the mapping."
        ),
        "items": metadata_rows,
        "source_failures": failures,
    }
    (output / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {created} multi-injection real-pair previews")
    print(f"Preview directory: {previews}")
    print(f"Summary: {output / 'preview_summary.json'}")


if __name__ == "__main__":
    main()
