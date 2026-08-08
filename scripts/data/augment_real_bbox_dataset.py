#!/usr/bin/env python3
"""Create bbox-aware real Arabic line-pair augmentations with strict text/image checks.

This generator never invents Arabic glyphs. Every text-bearing injected crop
comes from a real annotated bbox.json region. Transcript edits are accepted
only when every bbox label can be mapped back to the source transcript in
reading order. The generated dataset therefore keeps line images, transcripts,
and bbox.json annotations synchronized.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
import unicodedata
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from Evaluation.real_subword_box_json import load_json_annotations
from Evaluation.real_subword_box_metrics import SubwordBox


_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
MODES = (
    "original",
    "photometric",
    "geometry",
    "aligned_injection",
    "unaligned_injection",
    "mixed",
)


@dataclass(frozen=True)
class MappedBox:
    box: SubwordBox
    text_start: int
    text_end: int


@dataclass(frozen=True)
class SourceLine:
    image_path: Path
    text_path: Path
    text: str
    boxes: tuple[MappedBox, ...]
    annotation_path: str
    side: str
    pair_id: str
    line_idx: int


@dataclass(frozen=True)
class DonorRun:
    line: SourceLine
    start: int
    size: int
    text: str
    canonical_text: str
    rect: tuple[float, float, float, float]


@dataclass
class AugmentedLine:
    image: Image.Image
    text: str
    boxes: list[SubwordBox]
    injected_box_indices: list[int]
    operations: list[dict]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (root / path, root.parent / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_marks(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = _ARABIC_DIACRITICS.sub("", value.replace("ـ", ""))
    return value


def _canonical_text(value: str) -> str:
    return " ".join(_strip_marks(value).split())


def _compact(value: str) -> str:
    return "".join(_canonical_text(value).split())


def _normalized_stream_with_source_map(text: str) -> tuple[str, list[int]]:
    stream: list[str] = []
    source_indices: list[int] = []
    for source_index, raw_char in enumerate(text):
        normalized = _strip_marks(raw_char)
        for char in normalized:
            if char.isspace():
                continue
            stream.append(char)
            source_indices.append(source_index)
    return "".join(stream), source_indices


def _try_map_boxes(text: str, boxes: Iterable[SubwordBox]) -> tuple[MappedBox, ...] | None:
    stream, source_map = _normalized_stream_with_source_map(text)
    cursor = 0
    mapped: list[MappedBox] = []
    for box in boxes:
        token = _compact(box.text)
        if not token:
            return None
        position = stream.find(token, cursor)
        if position < 0:
            return None
        end_position = position + len(token) - 1
        if end_position >= len(source_map):
            return None
        start = source_map[position]
        end = source_map[end_position] + 1
        mapped.append(MappedBox(box=box, text_start=int(start), text_end=int(end)))
        cursor = position + len(token)
    return tuple(mapped)


def _map_boxes_strict(text: str, boxes: Iterable[SubwordBox]) -> tuple[MappedBox, ...]:
    boxes = tuple(boxes)
    candidates = [
        boxes,
        tuple(sorted(boxes, key=lambda box: box.center_x, reverse=True)),
        tuple(sorted(boxes, key=lambda box: box.center_x)),
    ]
    seen = set()
    for candidate in candidates:
        signature = tuple((box.text, box.x0, box.y0, box.x1, box.y1) for box in candidate)
        if signature in seen:
            continue
        seen.add(signature)
        mapped = _try_map_boxes(text, candidate)
        if mapped is not None:
            return mapped
    raise ValueError("bbox labels cannot be mapped exactly to the transcript in a consistent order")


def _load_line(
    dataset_root: Path,
    pair_id: str,
    side_name: str,
    side: dict,
) -> SourceLine:
    image_path = _resolve(dataset_root, side["line_image_path"])
    text_path = _resolve(dataset_root, side["text_original_path"])
    text = text_path.read_text(encoding="utf-8").strip()
    annotations = load_json_annotations(image_path)
    if annotations.status != "ok" or not annotations.boxes:
        raise ValueError(
            f"bbox.json unresolved for {image_path}: status={annotations.status} error={annotations.error}"
        )
    mapped = _map_boxes_strict(text, annotations.boxes)
    with Image.open(image_path) as opened:
        width, height = opened.size
    for item in mapped:
        box = item.box
        if not (0 <= box.x0 < box.x1 <= width and 0 <= box.y0 < box.y1 <= height):
            raise ValueError(f"out-of-bounds bbox for {image_path}: {box}")
    return SourceLine(
        image_path=image_path,
        text_path=text_path,
        text=text,
        boxes=mapped,
        annotation_path=annotations.workbook,
        side=side_name,
        pair_id=str(pair_id),
        line_idx=int(side.get("line_idx", -1)),
    )


def _rect_for_boxes(boxes: Iterable[SubwordBox]) -> tuple[float, float, float, float]:
    values = tuple(boxes)
    return (
        min(box.x0 for box in values),
        min(box.y0 for box in values),
        max(box.x1 for box in values),
        max(box.y1 for box in values),
    )


def _expand_rect(
    rect: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (
        max(0, int(math.floor(x0)) - padding),
        max(0, int(math.floor(y0)) - padding),
        min(width, int(math.ceil(x1)) + padding),
        min(height, int(math.ceil(y1)) + padding),
    )


def _run_from_line(line: SourceLine, start: int, size: int) -> DonorRun:
    selected = line.boxes[start : start + size]
    text_start = selected[0].text_start
    text_end = selected[-1].text_end
    exact_text = line.text[text_start:text_end].strip()
    return DonorRun(
        line=line,
        start=start,
        size=size,
        text=exact_text,
        canonical_text=_canonical_text(exact_text),
        rect=_rect_for_boxes(item.box for item in selected),
    )


def _build_donor_index(
    lines: Iterable[SourceLine],
    max_ngram: int,
    min_chars: int,
    max_chars: int,
) -> dict[tuple[int, str], list[DonorRun]]:
    index: dict[tuple[int, str], list[DonorRun]] = defaultdict(list)
    for line in lines:
        for size in range(1, max_ngram + 1):
            if size > len(line.boxes):
                break
            for start in range(len(line.boxes) - size + 1):
                run = _run_from_line(line, start, size)
                compact_length = len(_compact(run.canonical_text))
                if not (min_chars <= compact_length <= max_chars):
                    continue
                index[(size, run.canonical_text)].append(run)

    result: dict[tuple[int, str], list[DonorRun]] = {}
    for key, runs in index.items():
        by_image: dict[Path, DonorRun] = {}
        for run in runs:
            by_image.setdefault(run.line.image_path, run)
        if len(by_image) >= 2:
            result[key] = list(by_image.values())
    return result


def _image_background(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    band = max(1, min(8, height // 8, width // 8))
    samples = np.concatenate(
        (
            array[:band].reshape(-1, 3),
            array[-band:].reshape(-1, 3),
            array[:, :band].reshape(-1, 3),
            array[:, -band:].reshape(-1, 3),
        ),
        axis=0,
    )
    median = np.median(samples, axis=0)
    return tuple(int(round(value)) for value in median)


def _copy_line(line: SourceLine) -> AugmentedLine:
    with Image.open(line.image_path) as opened:
        image = opened.convert("RGB").copy()
    return AugmentedLine(
        image=image,
        text=line.text,
        boxes=[item.box for item in line.boxes],
        injected_box_indices=[],
        operations=[],
    )


def _choose_target_run(
    rng: random.Random,
    line: SourceLine,
    size: int,
    donor_rect: tuple[float, float, float, float],
    donor_text: str,
    width_ratio_min: float,
    width_ratio_max: float,
) -> int | None:
    donor_width = max(1.0, donor_rect[2] - donor_rect[0])
    candidates: list[int] = []
    for start in range(0, len(line.boxes) - size + 1):
        run = _run_from_line(line, start, size)
        target_width = max(1.0, run.rect[2] - run.rect[0])
        ratio = donor_width / target_width
        if not (width_ratio_min <= ratio <= width_ratio_max):
            continue
        if _canonical_text(run.text) == _canonical_text(donor_text):
            continue
        candidates.append(start)
    return rng.choice(candidates) if candidates else None


def _transform_donor_boxes(
    donor: DonorRun,
    crop_rect: tuple[int, int, int, int],
    paste_x: int,
    paste_y: int,
    scale_x: float,
    scale_y: float,
) -> list[SubwordBox]:
    crop_x0, crop_y0, _, _ = crop_rect
    output = []
    for item in donor.line.boxes[donor.start : donor.start + donor.size]:
        box = item.box
        output.append(
            SubwordBox(
                text=box.text,
                x0=paste_x + (box.x0 - crop_x0) * scale_x,
                y0=paste_y + (box.y0 - crop_y0) * scale_y,
                x1=paste_x + (box.x1 - crop_x0) * scale_x,
                y1=paste_y + (box.y1 - crop_y0) * scale_y,
                source_row=box.source_row,
            )
        )
    return output


def _inject_run(
    rng: random.Random,
    target: SourceLine,
    donor: DonorRun,
    width_ratio_min: float,
    width_ratio_max: float,
    padding: int,
) -> AugmentedLine | None:
    target_start = _choose_target_run(
        rng,
        target,
        donor.size,
        donor.rect,
        donor.text,
        width_ratio_min,
        width_ratio_max,
    )
    if target_start is None:
        return None

    result = _copy_line(target)
    target_items = target.boxes[target_start : target_start + donor.size]
    target_rect_float = _rect_for_boxes(item.box for item in target_items)
    target_rect = _expand_rect(
        target_rect_float,
        result.image.width,
        result.image.height,
        padding,
    )

    with Image.open(donor.line.image_path) as opened:
        donor_image = opened.convert("RGB")
    donor_crop_rect = _expand_rect(
        donor.rect,
        donor_image.width,
        donor_image.height,
        padding,
    )
    donor_crop = donor_image.crop(donor_crop_rect)

    target_width = max(1, target_rect[2] - target_rect[0])
    target_height = max(1, target_rect[3] - target_rect[1])
    fitted = ImageOps.contain(donor_crop, (target_width, target_height), Image.Resampling.LANCZOS)
    paste_x = target_rect[0] + (target_width - fitted.width) // 2
    paste_y = target_rect[1] + (target_height - fitted.height) // 2

    draw = ImageDraw.Draw(result.image)
    draw.rectangle(target_rect, fill=_image_background(result.image))
    result.image.paste(fitted, (paste_x, paste_y))

    scale_x = fitted.width / max(1, donor_crop.width)
    scale_y = fitted.height / max(1, donor_crop.height)
    injected_boxes = _transform_donor_boxes(
        donor,
        donor_crop_rect,
        paste_x,
        paste_y,
        scale_x,
        scale_y,
    )

    text_start = target_items[0].text_start
    text_end = target_items[-1].text_end
    replacement = donor.text.strip()
    result.text = target.text[:text_start] + replacement + target.text[text_end:]

    before = result.boxes[:target_start]
    after = result.boxes[target_start + donor.size :]
    result.boxes = before + injected_boxes + after
    result.injected_box_indices = list(
        range(len(before), len(before) + len(injected_boxes))
    )
    result.operations.append(
        {
            "type": "bbox_injection",
            "target_image": str(target.image_path),
            "target_box_start": int(target_start),
            "target_box_count": int(donor.size),
            "donor_image": str(donor.line.image_path),
            "donor_annotation": donor.line.annotation_path,
            "donor_text": donor.text,
            "donor_canonical_text": donor.canonical_text,
            "target_rect": [int(value) for value in target_rect],
            "donor_crop_rect": [int(value) for value in donor_crop_rect],
        }
    )
    _validate_augmented_line(result)
    return result


def _apply_photometric(rng: random.Random, line: AugmentedLine, args) -> None:
    operations = []
    brightness = rng.uniform(args.brightness_min, args.brightness_max)
    contrast = rng.uniform(args.contrast_min, args.contrast_max)
    line.image = ImageEnhance.Brightness(line.image).enhance(brightness)
    line.image = ImageEnhance.Contrast(line.image).enhance(contrast)
    operations.extend(
        [
            {"type": "brightness", "factor": round(brightness, 4)},
            {"type": "contrast", "factor": round(contrast, 4)},
        ]
    )

    if rng.random() < args.blur_probability:
        radius = rng.uniform(args.blur_radius_min, args.blur_radius_max)
        line.image = line.image.filter(ImageFilter.GaussianBlur(radius=radius))
        operations.append({"type": "blur", "radius": round(radius, 4)})

    array = np.asarray(line.image).astype(np.float32)
    local_rng = np.random.default_rng(rng.randrange(2**32))
    if rng.random() < args.gaussian_probability:
        std = rng.uniform(args.gaussian_std_min, args.gaussian_std_max)
        array += local_rng.normal(0.0, std, size=array.shape)
        operations.append({"type": "gaussian", "std": round(std, 4)})

    if rng.random() < args.salt_pepper_probability:
        density = rng.uniform(args.salt_pepper_density_min, args.salt_pepper_density_max)
        random_map = local_rng.random(array.shape[:2])
        pepper = random_map < density / 2.0
        salt = (random_map >= density / 2.0) & (random_map < density)
        array[pepper] = 0
        array[salt] = 255
        operations.append({"type": "salt_pepper", "density": round(density, 6)})

    line.image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    line.operations.extend(operations)


def _apply_geometry(rng: random.Random, line: AugmentedLine, args) -> bool:
    image = line.image
    width, height = image.size
    scale_x = rng.uniform(args.scale_x_min, args.scale_x_max)
    scale_y = rng.uniform(args.scale_y_min, args.scale_y_max)
    new_width = max(1, int(round(width * scale_x)))
    new_height = max(1, int(round(height * scale_y)))
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    max_dx = int(round(width * args.translate_x_fraction))
    max_dy = int(round(height * args.translate_y_fraction))
    offset_x = (width - new_width) // 2 + rng.randint(-max_dx, max_dx)
    offset_y = (height - new_height) // 2 + rng.randint(-max_dy, max_dy)

    background = _image_background(image)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(resized, (offset_x, offset_y))

    transformed: list[SubwordBox] = []
    for box in line.boxes:
        x0 = box.x0 * scale_x + offset_x
        y0 = box.y0 * scale_y + offset_y
        x1 = box.x1 * scale_x + offset_x
        y1 = box.y1 * scale_y + offset_y
        clipped = (
            max(0.0, min(float(width), x0)),
            max(0.0, min(float(height), y0)),
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
        )
        original_area = max(1.0, (x1 - x0) * (y1 - y0))
        clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if clipped_area / original_area < 0.90:
            return False
        transformed.append(
            SubwordBox(box.text, *clipped, source_row=box.source_row)
        )

    line.image = canvas
    line.boxes = transformed
    line.operations.append(
        {
            "type": "geometry",
            "scale_x": round(scale_x, 5),
            "scale_y": round(scale_y, 5),
            "offset_x": int(offset_x),
            "offset_y": int(offset_y),
        }
    )
    _validate_augmented_line(line)
    return True


def _validate_augmented_line(line: AugmentedLine) -> None:
    width, height = line.image.size
    for box in line.boxes:
        if not (0 <= box.x0 < box.x1 <= width and 0 <= box.y0 < box.y1 <= height):
            raise ValueError(f"generated bbox is outside image bounds: {box}")
    _map_boxes_strict(line.text, line.boxes)


def _choose_aligned_donors(
    rng: random.Random,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    source_a: SourceLine,
    source_b: SourceLine,
) -> tuple[DonorRun, DonorRun] | None:
    keys = list(donor_index)
    rng.shuffle(keys)
    for key in keys:
        eligible = [
            run
            for run in donor_index[key]
            if run.line.image_path not in {source_a.image_path, source_b.image_path}
        ]
        if len(eligible) < 2:
            continue
        rng.shuffle(eligible)
        for first in eligible:
            for second in eligible:
                if first.line.image_path == second.line.image_path:
                    continue
                return first, second
    return None


def _choose_unaligned_donors(
    rng: random.Random,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    source_a: SourceLine,
    source_b: SourceLine,
) -> tuple[DonorRun, DonorRun] | None:
    keys = list(donor_index)
    rng.shuffle(keys)
    for first_key in keys:
        first_runs = [
            run
            for run in donor_index[first_key]
            if run.line.image_path not in {source_a.image_path, source_b.image_path}
        ]
        if not first_runs:
            continue
        for second_key in keys:
            if second_key[0] != first_key[0] or second_key[1] == first_key[1]:
                continue
            second_runs = [
                run
                for run in donor_index[second_key]
                if run.line.image_path not in {source_a.image_path, source_b.image_path}
            ]
            if second_runs:
                return rng.choice(first_runs), rng.choice(second_runs)
    return None


def _augment_pair(
    rng: random.Random,
    mode: str,
    source_a: SourceLine,
    source_b: SourceLine,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    args,
) -> tuple[AugmentedLine, AugmentedLine, dict] | None:
    metadata: dict[str, object] = {"mode": mode}
    line_a = _copy_line(source_a)
    line_b = _copy_line(source_b)

    if mode in {"aligned_injection", "mixed"}:
        donors = _choose_aligned_donors(rng, donor_index, source_a, source_b)
        if donors is None:
            return None
        line_a = _inject_run(
            rng,
            source_a,
            donors[0],
            args.injection_width_ratio_min,
            args.injection_width_ratio_max,
            args.injection_padding,
        )
        line_b = _inject_run(
            rng,
            source_b,
            donors[1],
            args.injection_width_ratio_min,
            args.injection_width_ratio_max,
            args.injection_padding,
        )
        if line_a is None or line_b is None:
            return None
        metadata["injected_shared_text"] = donors[0].canonical_text
        metadata["donor_images"] = [str(donors[0].line.image_path), str(donors[1].line.image_path)]

    elif mode == "unaligned_injection":
        donors = _choose_unaligned_donors(rng, donor_index, source_a, source_b)
        if donors is None:
            return None
        line_a = _inject_run(
            rng,
            source_a,
            donors[0],
            args.injection_width_ratio_min,
            args.injection_width_ratio_max,
            args.injection_padding,
        )
        line_b = _inject_run(
            rng,
            source_b,
            donors[1],
            args.injection_width_ratio_min,
            args.injection_width_ratio_max,
            args.injection_padding,
        )
        if line_a is None or line_b is None:
            return None
        metadata["injected_text_A"] = donors[0].canonical_text
        metadata["injected_text_B"] = donors[1].canonical_text

    if mode == "geometry":
        if not _apply_geometry(rng, line_a, args):
            return None
        if not _apply_geometry(rng, line_b, args):
            return None

    if mode in {"photometric", "mixed"}:
        _apply_photometric(rng, line_a, args)
        _apply_photometric(rng, line_b, args)

    _validate_augmented_line(line_a)
    _validate_augmented_line(line_b)
    metadata["A_operations"] = line_a.operations
    metadata["B_operations"] = line_b.operations
    return line_a, line_b, metadata


def _box_payload(image_name: str, boxes: Iterable[SubwordBox]) -> dict:
    return {
        "image": image_name,
        "boxes": [
            {
                "text": box.text,
                "x0": round(float(box.x0), 3),
                "y0": round(float(box.y0), 3),
                "x1": round(float(box.x1), 3),
                "y1": round(float(box.y1), 3),
            }
            for box in boxes
        ],
    }


def _save_side(root: Path, name: str, line: AugmentedLine) -> dict:
    side_root = root / name
    side_root.mkdir(parents=True, exist_ok=True)
    image_path = side_root / "line.png"
    text_path = side_root / "text_original.txt"
    bbox_path = side_root / "bbox.json"
    line.image.save(image_path)
    text_path.write_text(line.text.strip() + "\n", encoding="utf-8")
    bbox_path.write_text(
        json.dumps(_box_payload(image_path.name, line.boxes), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "line_image_path": str(image_path),
        "text_original_path": str(text_path),
        "bbox_path": str(bbox_path),
    }


def _draw_boxes(image: Image.Image, boxes: Iterable[SubwordBox], injected: set[int]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for index, box in enumerate(boxes):
        color = (220, 40, 40) if index in injected else (30, 100, 220)
        draw.rectangle(
            [int(round(box.x0)), int(round(box.y0)), int(round(box.x1)), int(round(box.y1))],
            outline=color,
            width=2,
        )
        draw.text((int(round(box.x0)), max(0, int(round(box.y0)) - 11)), str(index), fill=color)
    return output


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
    return canvas


def _save_preview(
    path: Path,
    source_a: SourceLine,
    source_b: SourceLine,
    augmented_a: AugmentedLine,
    augmented_b: AugmentedLine,
    mode: str,
    metadata: dict,
) -> None:
    with Image.open(source_a.image_path) as opened:
        original_a = _draw_boxes(opened.convert("RGB"), [item.box for item in source_a.boxes], set())
    with Image.open(source_b.image_path) as opened:
        original_b = _draw_boxes(opened.convert("RGB"), [item.box for item in source_b.boxes], set())
    augmented_a_view = _draw_boxes(
        augmented_a.image, augmented_a.boxes, set(augmented_a.injected_box_indices)
    )
    augmented_b_view = _draw_boxes(
        augmented_b.image, augmented_b.boxes, set(augmented_b.injected_box_indices)
    )

    panel_w = max(original_a.width, original_b.width, augmented_a_view.width, augmented_b_view.width)
    panel_h = max(original_a.height, original_b.height, augmented_a_view.height, augmented_b_view.height) + 28
    sheet = Image.new("RGB", (panel_w * 2, panel_h * 2 + 50), (250, 250, 250))
    labels = (
        ("A original", original_a, 0, 0),
        ("A augmented", augmented_a_view, panel_w, 0),
        ("B original", original_b, 0, panel_h),
        ("B augmented", augmented_b_view, panel_w, panel_h),
    )
    draw = ImageDraw.Draw(sheet)
    for label, image, x, y in labels:
        panel = _fit_panel(image, panel_w, panel_h - 28)
        sheet.paste(panel, (x, y + 28))
        draw.text((x + 8, y + 7), label, fill=(0, 0, 0))
    draw.text((8, panel_h * 2 + 12), f"mode={mode} | red=injected bbox | blue=existing bbox", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "mode": mode,
                "source_A": str(source_a.image_path),
                "source_B": str(source_b.image_path),
                "original_text_A": source_a.text,
                "augmented_text_A": augmented_a.text,
                "original_text_B": source_b.text,
                "augmented_text_B": augmented_b.text,
                "augmentation": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        raise ValueError(f"Unknown augmentation mode(s): {invalid}; choices={MODES}")
    if not modes:
        raise ValueError("At least one augmentation mode is required")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DataSet/ArabicDataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated modes: " + ",".join(MODES),
    )
    parser.add_argument("--max-ngram-boxes", type=int, default=3)
    parser.add_argument("--injection-min-chars", type=int, default=4)
    parser.add_argument("--injection-max-chars", type=int, default=28)
    parser.add_argument("--injection-width-ratio-min", type=float, default=0.65)
    parser.add_argument("--injection-width-ratio-max", type=float, default=1.55)
    parser.add_argument("--injection-padding", type=int, default=3)
    parser.add_argument("--max-attempts-per-output", type=int, default=80)
    parser.add_argument("--brightness-min", type=float, default=0.88)
    parser.add_argument("--brightness-max", type=float, default=1.12)
    parser.add_argument("--contrast-min", type=float, default=0.85)
    parser.add_argument("--contrast-max", type=float, default=1.20)
    parser.add_argument("--blur-probability", type=float, default=0.30)
    parser.add_argument("--blur-radius-min", type=float, default=0.15)
    parser.add_argument("--blur-radius-max", type=float, default=0.75)
    parser.add_argument("--gaussian-probability", type=float, default=0.85)
    parser.add_argument("--gaussian-std-min", type=float, default=1.5)
    parser.add_argument("--gaussian-std-max", type=float, default=8.0)
    parser.add_argument("--salt-pepper-probability", type=float, default=0.45)
    parser.add_argument("--salt-pepper-density-min", type=float, default=0.0004)
    parser.add_argument("--salt-pepper-density-max", type=float, default=0.0025)
    parser.add_argument("--scale-x-min", type=float, default=0.94)
    parser.add_argument("--scale-x-max", type=float, default=1.04)
    parser.add_argument("--scale-y-min", type=float, default=0.96)
    parser.add_argument("--scale-y-max", type=float, default=1.04)
    parser.add_argument("--translate-x-fraction", type=float, default=0.015)
    parser.add_argument("--translate-y-fraction", type=float, default=0.035)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if args.max_ngram_boxes <= 0:
        raise ValueError("--max-ngram-boxes must be positive")
    modes = _parse_modes(args.modes)

    dataset_root = args.data_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else dataset_root / "dataset_manifest.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    raw_records = [
        record
        for record in _read_manifest(manifest_path)
        if not labels or str(record.get("label_type", "")) in labels
    ]
    if not raw_records:
        raise ValueError(f"No manifest pairs remain for labels={sorted(labels)}")

    line_cache: dict[Path, SourceLine] = {}
    usable_pairs: list[tuple[dict, SourceLine, SourceLine]] = []
    rejected_reasons: Counter[str] = Counter()

    for record in raw_records:
        try:
            pair_id = str(record.get("pair_id", len(usable_pairs) + 1))
            loaded = []
            for side_name in ("A", "B"):
                side = record[side_name]
                image_path = _resolve(dataset_root, side["line_image_path"])
                if image_path not in line_cache:
                    line_cache[image_path] = _load_line(dataset_root, pair_id, side_name, side)
                loaded.append(line_cache[image_path])
            usable_pairs.append((record, loaded[0], loaded[1]))
        except Exception as exc:
            rejected_reasons[f"{type(exc).__name__}: {exc}"] += 1

    if not usable_pairs:
        details = "\n".join(f"  {count} x {reason}" for reason, count in rejected_reasons.most_common(8))
        raise RuntimeError(f"No bbox/text-valid real pairs are available.\n{details}")

    donor_index = _build_donor_index(
        line_cache.values(),
        args.max_ngram_boxes,
        args.injection_min_chars,
        args.injection_max_chars,
    )
    if any(mode in {"aligned_injection", "unaligned_injection", "mixed"} for mode in modes) and not donor_index:
        raise RuntimeError(
            "No repeated real bbox n-grams were found for injection. "
            "Check bbox labels or lower --injection-min-chars."
        )

    rng = random.Random(args.seed)
    schedule = [modes[index % len(modes)] for index in range(args.num_pairs)]
    rng.shuffle(schedule)
    output_manifest = output_root / "dataset_manifest.jsonl"
    preview_root = output_root / "previews"
    mode_counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()

    with output_manifest.open("w", encoding="utf-8") as manifest_handle:
        for output_index, mode in enumerate(schedule, start=1):
            generated = None
            source_record = source_a = source_b = None
            last_error = None
            for _ in range(args.max_attempts_per_output):
                source_record, source_a, source_b = rng.choice(usable_pairs)
                try:
                    generated = _augment_pair(
                        rng, mode, source_a, source_b, donor_index, args
                    )
                except Exception as exc:
                    last_error = exc
                    failures[f"{mode}: {type(exc).__name__}: {exc}"] += 1
                    generated = None
                if generated is not None:
                    break
            if generated is None:
                raise RuntimeError(
                    f"Could not create augmentation {output_index}/{args.num_pairs} mode={mode} "
                    f"after {args.max_attempts_per_output} attempts"
                ) from last_error

            augmented_a, augmented_b, augmentation_meta = generated
            pair_root = output_root / "pairs" / f"aug_{output_index:06d}"
            saved_a = _save_side(pair_root, "A", augmented_a)
            saved_b = _save_side(pair_root, "B", augmented_b)

            source_label = str(source_record.get("label_type", ""))
            label_type = source_label if mode in {"original", "photometric", "geometry"} else "augmented_positive"
            record = {
                "pair_id": f"{source_record.get('pair_id', output_index)}__aug_{output_index:06d}",
                "label_type": label_type,
                "source_label_type": source_label,
                "source_pair_id": source_record.get("pair_id"),
                "scores": source_record.get("scores", {}),
                "augmentation": augmentation_meta,
                "A": {
                    **saved_a,
                    "line_idx": int(source_a.line_idx),
                    "source_line_image_path": str(source_a.image_path),
                    "source_bbox_json": source_a.annotation_path,
                },
                "B": {
                    **saved_b,
                    "line_idx": int(source_b.line_idx),
                    "source_line_image_path": str(source_b.image_path),
                    "source_bbox_json": source_b.annotation_path,
                },
            }
            manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            mode_counts[mode] += 1

            if args.preview:
                _save_preview(
                    preview_root / f"preview_{output_index:03d}_{mode}.png",
                    source_a,
                    source_b,
                    augmented_a,
                    augmented_b,
                    mode,
                    augmentation_meta,
                )

    summary = {
        "source_dataset": str(dataset_root),
        "source_manifest": str(manifest_path),
        "requested_labels": sorted(labels),
        "source_manifest_pairs": len(raw_records),
        "bbox_text_valid_pairs": len(usable_pairs),
        "bbox_text_valid_unique_lines": len(line_cache),
        "repeated_injection_ngram_keys": len(donor_index),
        "generated_pairs": args.num_pairs,
        "mode_counts": dict(mode_counts),
        "modes": modes,
        "seed": args.seed,
        "strict_text_bbox_validation": True,
        "injection_policy": {
            "aligned": "same annotated transcript span from two distinct real donor images",
            "unaligned": "different annotated transcript spans from real donor images",
            "target_operation": "bbox-span replacement on a real target line",
            "transcript_operation": "replace exactly the character span mapped from target bbox labels",
            "no_generated_arabic_glyphs": True,
        },
        "preview_enabled": bool(args.preview),
        "rejected_source_reasons": dict(rejected_reasons),
        "generation_retry_failures": dict(failures),
        "output_manifest": str(output_manifest),
    }
    (output_root / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
