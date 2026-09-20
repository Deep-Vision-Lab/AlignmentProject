"""Strict zero-shot preprocessing and visual-training helpers.

This module deliberately uses only synthetic images during training.  It makes
synthetic and real line images enter the visual encoder through the same
geometry and binary-image pipeline, while randomizing synthetic appearance so
real manuscript scans are less out-of-distribution.
"""
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Scalar approximation used when the branch is configured for true 1-channel
# grayscale input.  It keeps the normalized range close to ImageNet.
IMAGENET_GRAY_MEAN = (0.449,)
IMAGENET_GRAY_STD = (0.226,)

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(levels, histogram))
    left_weight = 0.0
    left_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        left_weight += histogram[threshold]
        if left_weight <= 0:
            continue
        right_weight = total - left_weight
        if right_weight <= 0:
            break
        left_sum += threshold * histogram[threshold]
        left_mean = left_sum / left_weight
        right_mean = (total_sum - left_sum) / right_weight
        variance = left_weight * right_weight * (left_mean - right_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def _border_mean(values: np.ndarray) -> float:
    height, width = values.shape
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.01)))
    border = np.concatenate(
        [
            values[:border_h, :].reshape(-1),
            values[-border_h:, :].reshape(-1),
            values[:, :border_w].reshape(-1),
            values[:, -border_w:].reshape(-1),
        ]
    )
    return float(border.mean()) if border.size else 255.0


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    threshold = otsu_threshold(gray)
    dark_ink = gray <= threshold
    light_ink = gray > threshold
    # The border normally represents page background.  Select the foreground
    # polarity that disagrees with it.
    return dark_ink if _border_mean(gray) >= 127.5 else light_ink


def _border_values(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.01)))
    return np.concatenate(
        [
            values[:border_h, :].reshape(-1),
            values[-border_h:, :].reshape(-1),
            values[:, :border_w].reshape(-1),
            values[:, -border_w:].reshape(-1),
        ]
    )


def _weighted_mass_bounds(weights: np.ndarray, low=0.001, high=0.999):
    """Ignore tiny isolated foreground mass when finding a 1-D support bbox."""
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(weights.sum())
    if total <= 0:
        return 0, int(weights.size)
    cumulative = np.cumsum(weights)
    lo = int(np.searchsorted(cumulative, total * float(low), side="left"))
    hi = int(np.searchsorted(cumulative, total * float(high), side="left")) + 1
    lo = max(0, min(lo, weights.size - 1))
    hi = max(lo + 1, min(hi, weights.size))
    return lo, hi


def foreground_detection_mask_with_metadata(image: Image.Image):
    """Build a TEMPORARY adaptive foreground mask without changing RGB pixels.

    Manuscript paper is not a flat color: illumination, stains and page texture
    can differ strongly from one side of a line crop to the other. A single
    global background estimate therefore marks large paper regions as
    foreground. Estimate the slowly varying local paper background with a wide
    Gaussian blur and detect only high-frequency dark/light stroke contrast.
    """
    source = image.convert("RGB")
    gray_image = source.convert("L")
    gray = np.asarray(gray_image, dtype=np.uint8)
    height, width = gray.shape

    radius = max(
        5.0,
        float(os.environ.get(
            "ZERO_SHOT_CROP_BACKGROUND_RADIUS",
            str(max(8.0, min(32.0, height * 0.08))),
        )),
    )
    smooth = np.asarray(
        gray_image.filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    )
    gray_f = gray.astype(np.float32)

    dark_contrast = np.clip(smooth - gray_f, 0.0, 255.0)
    light_contrast = np.clip(gray_f - smooth, 0.0, 255.0)

    # Choose the stroke polarity with the stronger sparse high-frequency tail.
    dark_strength = float(np.quantile(dark_contrast, 0.995))
    light_strength = float(np.quantile(light_contrast, 0.995))
    if dark_strength >= light_strength:
        contrast = dark_contrast
        polarity = "dark_ink"
    else:
        contrast = light_contrast
        polarity = "light_ink"

    contrast_u8 = np.clip(contrast, 0, 255).astype(np.uint8)
    otsu = int(otsu_threshold(contrast_u8))
    threshold = max(
        int(os.environ.get("ZERO_SHOT_CROP_MIN_CONTRAST", "10")),
        otsu,
    )
    mask = contrast_u8 >= threshold

    # A real text row/column contains several stroke pixels. Remove isolated
    # speckles before taking the support envelope.
    min_row_pixels = max(
        3,
        int(round(width * float(
            os.environ.get("ZERO_SHOT_CROP_MIN_ROW_FRACTION", "0.006")
        ))),
    )
    min_col_pixels = max(
        2,
        int(round(height * float(
            os.environ.get("ZERO_SHOT_CROP_MIN_COL_FRACTION", "0.015")
        ))),
    )
    row_keep = mask.sum(axis=1) >= min_row_pixels
    col_keep = mask.sum(axis=0) >= min_col_pixels
    filtered = mask & row_keep[:, None] & col_keep[None, :]

    if int(filtered.sum()) < 8:
        filtered = mask

    row_mass = filtered.sum(axis=1)
    col_mass = filtered.sum(axis=0)
    y0, y1 = _weighted_mass_bounds(row_mass, low=0.004, high=0.996)
    x0, x1 = _weighted_mass_bounds(col_mass, low=0.002, high=0.998)

    metadata = {
        "crop_detector": "adaptive-local-background-projection",
        "crop_background_blur_radius": float(radius),
        "crop_stroke_polarity": polarity,
        "crop_dark_strength_q995": dark_strength,
        "crop_light_strength_q995": light_strength,
        "crop_contrast_threshold": int(threshold),
        "crop_otsu_contrast_threshold": int(otsu),
        "crop_min_row_pixels": int(min_row_pixels),
        "crop_min_col_pixels": int(min_col_pixels),
        "crop_mask_pixels": int(mask.sum()),
        "crop_filtered_mask_pixels": int(filtered.sum()),
        "crop_raw_support_left": int(x0),
        "crop_raw_support_top": int(y0),
        "crop_raw_support_right": int(x1),
        "crop_raw_support_bottom": int(y1),
    }
    return filtered, metadata



def _contiguous_true_runs(values: np.ndarray):
    values = np.asarray(values, dtype=bool).reshape(-1)
    runs = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(values.size)))
    return runs


def detect_vertical_side_borders(mask: np.ndarray):
    """Detect near-full-height structural border lines on the two image sides.

    The line crops in ArabicDataset often contain one vertical page/box boundary
    near each side.  In the temporary foreground mask these appear as white
    strokes connecting the upper and lower portions of the line image.  Detect
    those structures before attempting any text bbox estimation.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("vertical border detector expects a 2-D mask")
    height, width = mask.shape
    radius = max(1, int(os.environ.get("ZERO_SHOT_BORDER_NEIGHBORHOOD", "3")))
    min_coverage = float(os.environ.get("ZERO_SHOT_BORDER_MIN_COVERAGE", "0.72"))
    band_fraction = float(os.environ.get("ZERO_SHOT_BORDER_BAND_FRACTION", "0.10"))
    min_band_coverage = float(
        os.environ.get("ZERO_SHOT_BORDER_MIN_BAND_COVERAGE", "0.45")
    )
    side_fraction = float(os.environ.get("ZERO_SHOT_BORDER_SIDE_FRACTION", "0.42"))

    top_h = max(2, int(round(height * band_fraction)))
    bottom_y = max(0, height - top_h)

    coverage = np.zeros(width, dtype=np.float32)
    top_coverage = np.zeros(width, dtype=np.float32)
    bottom_coverage = np.zeros(width, dtype=np.float32)
    for x in range(width):
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        rows = mask[:, x0:x1].any(axis=1)
        coverage[x] = float(rows.mean())
        top_coverage[x] = float(rows[:top_h].mean())
        bottom_coverage[x] = float(rows[bottom_y:].mean())

    candidates = (
        (coverage >= min_coverage)
        & (top_coverage >= min_band_coverage)
        & (bottom_coverage >= min_band_coverage)
    )
    runs = _contiguous_true_runs(candidates)

    left_limit = int(round(width * side_fraction))
    right_limit = int(round(width * (1.0 - side_fraction)))
    left_runs = [run for run in runs if (run[0] + run[1]) // 2 <= left_limit]
    right_runs = [run for run in runs if (run[0] + run[1]) // 2 >= right_limit]

    def _run_score(run):
        start, end = run
        center = max(start, min(width - 1, (start + end - 1) // 2))
        return float(
            coverage[start:end].max()
            + 0.20 * top_coverage[center]
            + 0.20 * bottom_coverage[center]
        )

    # If several nearly full-height strokes exist on one side, prefer the
    # innermost one among similarly strong candidates; it is the structural
    # boundary separating the line content from the outer margin.
    def _choose_left(items):
        if not items:
            return None
        best_score = max(_run_score(run) for run in items)
        strong = [run for run in items if _run_score(run) >= best_score - 0.05]
        return max(strong, key=lambda run: run[1])

    def _choose_right(items):
        if not items:
            return None
        best_score = max(_run_score(run) for run in items)
        strong = [run for run in items if _run_score(run) >= best_score - 0.05]
        return min(strong, key=lambda run: run[0])

    left = _choose_left(left_runs)
    right = _choose_right(right_runs)

    valid_pair = (
        left is not None
        and right is not None
        and int(right[0]) - int(left[1]) >= max(32, int(round(width * 0.25)))
    )

    metadata = {
        "side_border_detector": "near-full-height-mask-connectivity",
        "side_border_min_vertical_coverage": float(min_coverage),
        "side_border_min_top_bottom_coverage": float(min_band_coverage),
        "side_border_neighborhood_radius": int(radius),
        "side_border_left_run": list(left) if left is not None else None,
        "side_border_right_run": list(right) if right is not None else None,
        "side_border_pair_valid": bool(valid_pair),
        "side_border_candidate_runs": [list(run) for run in runs],
        "side_border_left_coverage": (
            float(coverage[left[0]:left[1]].max()) if left is not None else None
        ),
        "side_border_right_coverage": (
            float(coverage[right[0]:right[1]].max()) if right is not None else None
        ),
    }
    return left, right, metadata



def detect_horizontal_frame_borders(mask: np.ndarray):
    """Detect long structural frame lines above and/or below the handwriting.

    Unlike Arabic baselines and connected strokes, a page/frame rule occupies
    most columns of the candidate crop.  Detection is performed on the
    temporary foreground mask only; the returned boundaries are later applied
    to the untouched RGB source.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("horizontal frame detector expects a 2-D mask")
    height, width = mask.shape
    if height <= 1 or width <= 1:
        empty = {
            "horizontal_border_detector": "near-full-width-mask-connectivity",
            "horizontal_border_top_run": None,
            "horizontal_border_bottom_run": None,
            "horizontal_border_pair_valid": False,
            "horizontal_border_candidate_runs": [],
        }
        return None, None, empty

    radius = max(
        1,
        int(os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_NEIGHBORHOOD", "2")),
    )
    min_coverage = float(
        os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_MIN_COVERAGE", "0.58")
    )
    edge_fraction = float(
        os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_EDGE_FRACTION", "0.42")
    )

    coverage = np.zeros(height, dtype=np.float32)
    for y in range(height):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        columns = mask[y0:y1, :].any(axis=0)
        coverage[y] = float(columns.mean())

    candidates = coverage >= min_coverage
    runs = _contiguous_true_runs(candidates)

    # A frame rule is long but thin. Without this constraint a dense Arabic
    # handwriting band could look "wide" in the row projection and be mistaken
    # for a structural rule.
    max_thickness = max(
        5,
        int(round(height * float(
            os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_MAX_THICKNESS_FRACTION", "0.08")
        ))),
    )
    runs = [
        run for run in runs
        if int(run[1]) - int(run[0]) <= int(max_thickness)
    ]

    top_limit = int(round(height * edge_fraction))
    bottom_limit = int(round(height * (1.0 - edge_fraction)))
    top_runs = [run for run in runs if (run[0] + run[1]) // 2 <= top_limit]
    bottom_runs = [run for run in runs if (run[0] + run[1]) // 2 >= bottom_limit]

    def _score(run):
        start, end = run
        return float(coverage[start:end].max())

    def _choose_top(items):
        if not items:
            return None
        best = max(_score(run) for run in items)
        strong = [run for run in items if _score(run) >= best - 0.06]
        # The innermost strong rule is the actual boundary of the text frame.
        return max(strong, key=lambda run: run[1])

    def _choose_bottom(items):
        if not items:
            return None
        best = max(_score(run) for run in items)
        strong = [run for run in items if _score(run) >= best - 0.06]
        return min(strong, key=lambda run: run[0])

    top = _choose_top(top_runs)
    bottom = _choose_bottom(bottom_runs)
    valid_pair = (
        top is not None
        and bottom is not None
        and int(bottom[0]) - int(top[1])
        >= max(12, int(round(height * 0.12)))
    )

    metadata = {
        "horizontal_border_detector": "near-full-width-mask-connectivity",
        "horizontal_border_min_coverage": float(min_coverage),
        "horizontal_border_neighborhood_radius": int(radius),
        "horizontal_border_max_thickness": int(max_thickness),
        "horizontal_border_top_run": list(top) if top is not None else None,
        "horizontal_border_bottom_run": list(bottom) if bottom is not None else None,
        "horizontal_border_pair_valid": bool(valid_pair),
        "horizontal_border_candidate_runs": [list(run) for run in runs],
        "horizontal_border_top_coverage": (
            float(coverage[top[0]:top[1]].max()) if top is not None else None
        ),
        "horizontal_border_bottom_coverage": (
            float(coverage[bottom[0]:bottom[1]].max()) if bottom is not None else None
        ),
    }
    return top, bottom, metadata


def _erase_horizontal_frame_candidates(mask: np.ndarray, horizontal_meta: dict):
    """Remove structural horizontal rules only from the TEMP detection mask."""
    work = np.asarray(mask, dtype=bool).copy()
    height = int(work.shape[0])
    pad = max(
        1,
        int(os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_ERASE_PAD", "2")),
    )
    for run in horizontal_meta.get("horizontal_border_candidate_runs", []) or []:
        start, end = map(int, run)
        work[max(0, start - pad) : min(height, end + pad), :] = False
    return work


def _resolve_vertical_crop_from_frame(
    mask: np.ndarray,
    *,
    fallback_band=None,
):
    """Use top/bottom frame rules as hard bounds and projection elsewhere."""
    top, bottom, meta = detect_horizontal_frame_borders(mask)
    clean = _erase_horizontal_frame_candidates(mask, meta)

    band, _work, band_meta = _dominant_text_row_band(clean)
    if band is None:
        band = fallback_band
    if band is None:
        row_mass = clean.sum(axis=1).astype(np.float64)
        if float(row_mass.sum()) > 0.0:
            y0, y1 = _weighted_mass_bounds(row_mass, low=0.003, high=0.997)
            band = (int(y0), int(y1))
        else:
            band = (0, int(mask.shape[0]))

    proj_y0, proj_y1 = map(int, band)
    inset = max(
        1,
        int(os.environ.get("ZERO_SHOT_HORIZONTAL_BORDER_CROP_INSET", "2")),
    )

    y0 = int(top[1]) + inset if top is not None else proj_y0
    y1 = int(bottom[0]) - inset if bottom is not None else proj_y1

    y0 = max(0, min(int(mask.shape[0]) - 1, y0))
    y1 = max(y0 + 1, min(int(mask.shape[0]), y1))
    meta = {
        **meta,
        **band_meta,
        "horizontal_border_crop_inset": int(inset),
        "vertical_crop_projection_top": int(proj_y0),
        "vertical_crop_projection_bottom": int(proj_y1),
        "vertical_crop_used_top_frame": bool(top is not None),
        "vertical_crop_used_bottom_frame": bool(bottom is not None),
    }
    return (int(y0), int(y1)), clean, meta


def _merge_close_runs(runs, max_gap: int):
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if int(start) - int(merged[-1][1]) <= int(max_gap):
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return [tuple(item) for item in merged]


def _dominant_text_row_band(mask: np.ndarray, excluded_column_runs=()):
    """Find the main handwriting band while rejecting neighboring rows.

    Line crops occasionally contain part of the preceding/following manuscript
    line. Side borders are removed temporarily before the row projection so
    they cannot connect otherwise separate horizontal text bands. Selection
    prefers substantial foreground mass near the vertical center of the crop;
    edge-touching bands are penalized because they are usually leaked neighbor
    lines from the source page.
    """
    work = np.asarray(mask, dtype=bool).copy()
    if work.ndim != 2:
        raise ValueError("dominant text-band detector expects a 2-D mask")
    height, width = work.shape

    erase_pad = max(1, int(round(width * 0.002)))
    for run in excluded_column_runs or ():
        if run is None:
            continue
        start, end = map(int, run)
        work[:, max(0, start - erase_pad) : min(width, end + erase_pad)] = False

    row_mass = work.sum(axis=1).astype(np.float64)
    if float(row_mass.sum()) <= 0.0:
        return None, work, {"dominant_row_candidate_runs": []}

    smooth_width = max(3, int(round(height * 0.025)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    kernel = np.ones(smooth_width, dtype=np.float64) / float(smooth_width)
    smoothed = np.convolve(row_mass, kernel, mode="same")
    positive = smoothed[smoothed > 0.0]
    if positive.size == 0:
        return None, work, {"dominant_row_candidate_runs": []}

    # A deliberately permissive floor keeps dots/diacritics associated with the
    # main band. Nearby fragments are merged before a band is selected.
    floor = max(1.0, float(np.quantile(positive, 0.55)) * 0.20)
    active_runs = _contiguous_true_runs(smoothed >= floor)
    active_runs = _merge_close_runs(
        active_runs,
        max_gap=max(3, int(round(height * 0.055))),
    )
    if not active_runs:
        return None, work, {"dominant_row_candidate_runs": []}

    center_y = 0.5 * float(height)
    half_height = max(1.0, center_y)
    scored = []
    for start, end in active_runs:
        if end <= start:
            continue
        mass = float(row_mass[start:end].sum())
        if mass <= 0.0:
            continue
        band_center = 0.5 * float(start + end)
        center_distance = min(1.0, abs(band_center - center_y) / half_height)
        center_factor = 1.0 - 0.35 * center_distance
        edge_factor = 0.72 if start <= 1 or end >= height - 1 else 1.0
        score = mass * center_factor * edge_factor
        scored.append((score, int(start), int(end), mass))

    if not scored:
        return None, work, {
            "dominant_row_candidate_runs": [list(run) for run in active_runs]
        }

    score, start, end, mass = max(scored, key=lambda item: item[0])
    pad_y = max(3, int(round((end - start) * 0.18)))
    y0 = max(0, start - pad_y)
    y1 = min(height, end + pad_y)
    meta = {
        "dominant_row_candidate_runs": [list(run) for run in active_runs],
        "dominant_row_selected_run": [int(start), int(end)],
        "dominant_row_selected_mass": float(mass),
        "dominant_row_selected_score": float(score),
        "dominant_row_safety_margin": int(pad_y),
    }
    return (int(y0), int(y1)), work, meta


def _crop_with_partial_side_borders(
    source: Image.Image,
    mask: np.ndarray,
    detector_meta: dict,
):
    """Crop an incomplete rectangular/L-shaped frame around one text line.

    Vertical and horizontal structural rules are treated independently. A
    detected frame rule is a hard boundary; missing boundaries are estimated
    from the handwriting support after those rules are removed from the
    temporary mask.
    """
    left, right, border_meta = detect_vertical_side_borders(mask)
    if border_meta["side_border_pair_valid"]:
        return None, {**detector_meta, **border_meta}

    inset_x = max(1, int(os.environ.get("ZERO_SHOT_BORDER_CROP_INSET", "2")))

    # First build a broad interior using any reliable vertical side that exists.
    broad_x0 = (
        min(source.width - 1, int(left[1]) + inset_x)
        if left is not None
        else int(detector_meta["crop_raw_support_left"])
    )
    broad_x1 = (
        max(broad_x0 + 1, int(right[0]) - inset_x)
        if right is not None
        else int(detector_meta["crop_raw_support_right"])
    )
    broad_x0 = max(0, min(source.width - 1, broad_x0))
    broad_x1 = max(broad_x0 + 1, min(source.width, broad_x1))

    broad_mask = np.asarray(mask[:, broad_x0:broad_x1], dtype=bool)
    broad_vertical, broad_clean, horizontal_meta = _resolve_vertical_crop_from_frame(
        broad_mask
    )
    broad_y0, broad_y1 = broad_vertical

    # Convert the clean broad mask back to source-width coordinates and remove
    # candidate side-border columns. This prevents an L/U-shaped frame from
    # merging with the handwriting during the dominant-band estimate.
    work_mask = np.asarray(mask, dtype=bool).copy()
    work_mask[:, broad_x0:broad_x1] = broad_clean
    excluded_runs = [
        tuple(run) for run in border_meta.get("side_border_candidate_runs", [])
    ]
    band, work_mask, band_meta = _dominant_text_row_band(
        work_mask,
        excluded_column_runs=excluded_runs,
    )
    if band is None:
        band = (broad_y0, broad_y1)
    band_y0, band_y1 = map(int, band)

    # Refine the missing left/right boundary from only the target text band, not
    # from a top/bottom frame line.
    band_mask = np.asarray(work_mask[band_y0:band_y1], dtype=bool)
    col_mass = band_mask.sum(axis=0).astype(np.float64)
    if float(col_mass.sum()) <= 0.0:
        return None, {
            **detector_meta,
            **border_meta,
            **horizontal_meta,
            **band_meta,
        }

    proj_x0, proj_x1 = _weighted_mass_bounds(col_mass, low=0.002, high=0.998)
    pad_x = max(2, int(round((proj_x1 - proj_x0) * 0.02)))
    proj_x0 = max(0, int(proj_x0) - pad_x)
    proj_x1 = min(source.width, int(proj_x1) + pad_x)

    one_sided = (left is None) ^ (right is None)
    if one_sided and left is not None:
        x0 = min(source.width - 1, int(left[1]) + inset_x)
        x1 = int(proj_x1)
        crop_mode = "single_left_frame_border"
    elif one_sided and right is not None:
        x0 = int(proj_x0)
        x1 = max(x0 + 1, int(right[0]) - inset_x)
        crop_mode = "single_right_frame_border"
    else:
        x0, x1 = int(proj_x0), int(proj_x1)
        crop_mode = "horizontal_or_text_frame_projection"

    x0 = max(0, min(source.width - 1, int(x0)))
    x1 = max(x0 + 1, min(source.width, int(x1)))

    # Re-detect horizontal rules inside the final horizontal span. This catches
    # a rule connected to only one side whose coverage was too small before the
    # missing x-boundary was refined.
    final_mask = np.asarray(mask[:, x0:x1], dtype=bool)
    (y0, y1), _final_clean, final_horizontal_meta = (
        _resolve_vertical_crop_from_frame(
            final_mask,
            fallback_band=(band_y0, band_y1),
        )
    )

    if x1 - x0 < 32 or y1 <= y0:
        return None, {
            **detector_meta,
            **border_meta,
            **horizontal_meta,
            **final_horizontal_meta,
            **band_meta,
        }

    used_horizontal = (
        final_horizontal_meta.get("vertical_crop_used_top_frame", False)
        or final_horizontal_meta.get("vertical_crop_used_bottom_frame", False)
    )
    if used_horizontal:
        crop_mode = crop_mode + "_with_horizontal_frame"

    metadata = {
        **detector_meta,
        **border_meta,
        **horizontal_meta,
        **final_horizontal_meta,
        **band_meta,
        "crop_mode": crop_mode,
        "crop_detector": "independent-frame-borders-plus-text-projection",
        "crop_raw_support_left": int(x0),
        "crop_raw_support_top": int(y0),
        "crop_raw_support_right": int(x1),
        "crop_raw_support_bottom": int(y1),
        "side_border_crop_inset": int(inset_x),
        "projection_horizontal_margin": int(pad_x),
    }
    return (int(x0), int(y0), int(x1), int(y1)), metadata


def _crop_between_vertical_borders(
    source: Image.Image,
    mask: np.ndarray,
    detector_meta: dict,
):
    """Crop inside a complete/partial rectangular frame around the text."""
    left, right, border_meta = detect_vertical_side_borders(mask)
    if not border_meta["side_border_pair_valid"]:
        return None, {**detector_meta, **border_meta}

    inset_x = max(1, int(os.environ.get("ZERO_SHOT_BORDER_CROP_INSET", "2")))
    x0 = min(source.width - 1, int(left[1]) + inset_x)
    x1 = max(x0 + 1, int(right[0]) - inset_x)

    # Once the vertical frame is removed, inspect only the interior span for
    # long horizontal top/bottom rules. This works for complete rectangles and
    # for U/L shapes where a horizontal rule is connected to one side.
    interior_mask = np.asarray(mask[:, x0:x1], dtype=bool)
    (y0, y1), _clean, horizontal_meta = _resolve_vertical_crop_from_frame(
        interior_mask
    )

    if x1 - x0 < 32 or y1 <= y0:
        return None, {**detector_meta, **border_meta, **horizontal_meta}

    used_top = bool(horizontal_meta.get("vertical_crop_used_top_frame", False))
    used_bottom = bool(
        horizontal_meta.get("vertical_crop_used_bottom_frame", False)
    )
    if used_top and used_bottom:
        crop_mode = "full_frame_borders"
    elif used_top:
        crop_mode = "vertical_borders_with_top_frame"
    elif used_bottom:
        crop_mode = "vertical_borders_with_bottom_frame"
    else:
        crop_mode = "vertical_borders_text_projection"

    metadata = {
        **detector_meta,
        **border_meta,
        **horizontal_meta,
        "crop_mode": crop_mode,
        "crop_detector": "independent-vertical-horizontal-frame-borders",
        "crop_raw_support_left": int(x0),
        "crop_raw_support_top": int(y0),
        "crop_raw_support_right": int(x1),
        "crop_raw_support_bottom": int(y1),
        "side_border_crop_inset": int(inset_x),
    }
    return (int(x0), int(y0), int(x1), int(y1)), metadata


def foreground_crop_with_metadata(
    image: Image.Image,
    margin_x=0.025,
    margin_y=0.15,
):
    """Locate handwriting with a temporary mask, crop untouched ORIGINAL RGB."""
    source = image.convert("RGB")
    mode = os.environ.get("ZERO_SHOT_CROP_MODE", "legacy_otsu").strip().lower()

    if mode in {"vertical_borders", "side_borders", "full_height_borders"}:
        mask, detector_meta = foreground_detection_mask_with_metadata(source)
        border_box, detector_meta = _crop_between_vertical_borders(
            source, mask, detector_meta
        )
        if border_box is not None:
            x0, y0, x1, y1 = border_box
            # The side borders already define horizontal limits; do not add the
            # generic x margin outside them.
            margin_x = 0.0
            margin_y = 0.0
        else:
            partial_box, detector_meta = _crop_with_partial_side_borders(
                source, mask, detector_meta
            )
            if partial_box is not None:
                x0, y0, x1, y1 = partial_box
                # This helper already supplies both the structural/projection
                # horizontal safety and the vertical band safety margin.
                margin_x = 0.0
                margin_y = 0.0
            else:
                # Last-resort fallback only when no reliable dominant band can
                # be isolated. Preserve the previous robust projection behavior.
                x0 = int(detector_meta["crop_raw_support_left"])
                y0 = int(detector_meta["crop_raw_support_top"])
                x1 = int(detector_meta["crop_raw_support_right"])
                y1 = int(detector_meta["crop_raw_support_bottom"])
                detector_meta["crop_mode"] = "vertical_borders_fallback_projection"
    elif mode in {"robust", "robust_projection", "paper_contrast"}:
        mask, detector_meta = foreground_detection_mask_with_metadata(source)
        x0 = int(detector_meta["crop_raw_support_left"])
        y0 = int(detector_meta["crop_raw_support_top"])
        x1 = int(detector_meta["crop_raw_support_right"])
        y1 = int(detector_meta["crop_raw_support_bottom"])
        if int(mask.sum()) < 4 or x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, source.width, source.height
        detector_meta["crop_mode"] = "robust_projection"
    else:
        gray_for_mask = ImageOps.autocontrast(source.convert("L"))
        gray = np.asarray(gray_for_mask, dtype=np.uint8)
        mask = _ink_mask(gray)
        ys, xs = np.nonzero(mask)
        if xs.size < 4 or ys.size < 4:
            x0, y0, x1, y1 = 0, 0, source.width, source.height
        else:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
        detector_meta = {
            "crop_mode": "legacy_otsu",
            "crop_detector": "autocontrast-otsu-minmax",
            "crop_mask_pixels": int(mask.sum()),
            "crop_raw_support_left": int(x0),
            "crop_raw_support_top": int(y0),
            "crop_raw_support_right": int(x1),
            "crop_raw_support_bottom": int(y1),
        }

    pad_x = max(2, int(round((x1 - x0) * float(margin_x))))
    pad_y = max(2, int(round((y1 - y0) * float(margin_y))))
    box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(source.width, x1 + pad_x),
        min(source.height, y1 + pad_y),
    )
    cropped = source.crop(box)
    metadata = {
        "source_width": int(source.width),
        "source_height": int(source.height),
        "crop_left": int(box[0]),
        "crop_top": int(box[1]),
        "crop_right": int(box[2]),
        "crop_bottom": int(box[3]),
        "crop_width": int(box[2] - box[0]),
        "crop_height": int(box[3] - box[1]),
        "crop_margin_x": float(margin_x),
        "crop_margin_y": float(margin_y),
        **detector_meta,
    }
    return cropped, metadata


def foreground_crop(image: Image.Image, margin_x=0.025, margin_y=0.15) -> Image.Image:
    return foreground_crop_with_metadata(image, margin_x, margin_y)[0]


def aspect_preserving_pad_with_metadata(
    image: Image.Image,
    size=(128, 1024),
    target_ink_height_ratio=0.72,
    horizontal_jitter=0.0,
):
    """Resize once with one scale, then pad; never stretch windows independently."""
    target_h, target_w = map(int, size)
    source_mode = "L" if image.mode == "L" else "RGB"
    source = image.convert(source_mode)
    desired_h = max(8, int(round(target_h * float(target_ink_height_ratio))))
    scale = min(
        desired_h / max(1, source.height),
        target_w / max(1, source.width),
    )
    new_w = max(1, min(target_w, int(round(source.width * scale))))
    new_h = max(1, min(target_h, int(round(source.height * scale))))
    resized = source.resize((new_w, new_h), _BILINEAR)
    canvas = Image.new(
        source_mode,
        (target_w, target_h),
        color=255 if source_mode == "L" else (255, 255, 255),
    )
    max_x = max(0, target_w - new_w)
    centered_x = max_x // 2
    jitter = int(round(max_x * max(0.0, float(horizontal_jitter))))
    x = min(max_x, max(0, centered_x + random.randint(-jitter, jitter))) if jitter else centered_x
    y = max(0, (target_h - new_h) // 2)
    canvas.paste(resized, (x, y))
    metadata = {
        "scale_x": float(new_w / max(1, source.width)),
        "scale_y": float(new_h / max(1, source.height)),
        "resize_scale": float(scale),
        "resized_width": int(new_w),
        "resized_height": int(new_h),
        "offset_x": int(x),
        "offset_y": int(y),
        "canvas_width": int(target_w),
        "canvas_height": int(target_h),
    }
    return canvas, metadata


def aspect_preserving_pad(
    image: Image.Image,
    size=(128, 1024),
    target_ink_height_ratio=0.72,
    horizontal_jitter=0.0,
) -> Image.Image:
    return aspect_preserving_pad_with_metadata(
        image, size, target_ink_height_ratio, horizontal_jitter
    )[0]


def _random_resize(image: Image.Image) -> Image.Image:
    width_scale = random.uniform(0.84, 1.16)
    height_scale = random.uniform(0.90, 1.10)
    width = max(8, int(round(image.width * width_scale)))
    height = max(8, int(round(image.height * height_scale)))
    return image.resize((width, height), _BILINEAR)


def _add_gray_noise(image: Image.Image, std: float) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    array += np.random.normal(0.0, float(std), size=array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def _add_scan_artifacts(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    height, width = array.shape

    # Broken/faded strokes: short white interruptions.
    for _ in range(random.randint(0, 4)):
        block_w = random.randint(1, max(2, width // 100))
        block_h = random.randint(1, max(2, height // 12))
        x0 = random.randint(0, max(0, width - block_w))
        y0 = random.randint(0, max(0, height - block_h))
        array[y0 : y0 + block_h, x0 : x0 + block_w] = 255

    # Dust and bleed-through-like speckles.
    count = int(array.size * random.uniform(0.0, 0.0025))
    if count > 0:
        ys = np.random.randint(0, height, size=count)
        xs = np.random.randint(0, width, size=count)
        array[ys, xs] = np.random.choice([0, 40, 210, 255], size=count)
    return Image.fromarray(array, mode="L")


class ManuscriptLinePreprocessor:
    """Crop, geometrically normalize, degrade, and binarize a line image."""

    def __init__(
        self,
        size=(128, 1024),
        *,
        training=False,
        augment=False,
        binarize=True,
        method="otsu",
        fixed_threshold=180,
        threshold_jitter=0,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        auto_invert=True,
        autocontrast=True,
        augment_probability=0.85,
        clean_probability=0.20,
        white_ink_on_black=False,
        grayscale=False,
    ):
        self.size = tuple(map(int, size))
        self.training = bool(training)
        self.augment = bool(augment)
        self.binarize = bool(binarize)
        self.method = str(method).lower()
        self.fixed_threshold = int(fixed_threshold)
        self.threshold_jitter = max(0, int(threshold_jitter))
        self.preserve_aspect = bool(preserve_aspect)
        self.crop_foreground = bool(crop_foreground)
        self.target_ink_height_ratio = float(target_ink_height_ratio)
        self.auto_invert = bool(auto_invert)
        self.autocontrast = bool(autocontrast)
        self.augment_probability = float(augment_probability)
        self.clean_probability = float(clean_probability)
        self.white_ink_on_black = bool(white_ink_on_black)
        self.grayscale = bool(grayscale)
        if self.method not in {"otsu", "fixed", "random"}:
            raise ValueError("binarization method must be otsu, fixed, or random")

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.training or not self.augment:
            return image
        if random.random() < self.clean_probability:
            return image
        if random.random() > self.augment_probability:
            return image

        image = _random_resize(image)
        angle = random.uniform(-2.5, 2.5)
        image = image.rotate(angle, resample=_BILINEAR, expand=False, fillcolor=255)

        if random.random() < 0.45:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.25, 1.15)))
        if random.random() < 0.35:
            # Black ink: MinFilter expands strokes, MaxFilter erodes them.
            image = image.filter(
                ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3)
            )
        if random.random() < 0.70:
            image = _add_gray_noise(image, random.uniform(2.0, 14.0))
        if random.random() < 0.65:
            image = _add_scan_artifacts(image)
        return image

    def _threshold(self, gray: np.ndarray) -> int:
        if self.method == "fixed":
            return max(0, min(255, self.fixed_threshold))
        base = otsu_threshold(gray)
        if self.method == "random" and self.training and self.threshold_jitter > 0:
            base += random.randint(-self.threshold_jitter, self.threshold_jitter)
        return max(0, min(255, int(base)))

    def preprocess_with_metadata(self, image: Image.Image):
        """Return processed RGB plus crop/resize offsets for inverse mapping."""
        work = image.convert("L" if self.grayscale else "RGB")
        metadata = {
            "source_width": int(work.width),
            "source_height": int(work.height),
            "crop_left": 0,
            "crop_top": 0,
            "crop_width": int(work.width),
            "crop_height": int(work.height),
        }
        if self.crop_foreground:
            work, crop_meta = foreground_crop_with_metadata(work)
            metadata.update(crop_meta)

        # Augmentation is deliberately disabled on this restoration branch, but
        # keep the generic profile behavior for other callers.
        work = self._augment(work)

        if self.preserve_aspect:
            work, resize_meta = aspect_preserving_pad_with_metadata(
                work,
                self.size,
                self.target_ink_height_ratio,
                horizontal_jitter=0.08 if self.training and self.augment else 0.0,
            )
            metadata.update(resize_meta)
        else:
            source_w, source_h = work.size
            target_h, target_w = self.size
            work = work.resize((target_w, target_h), _BILINEAR)
            metadata.update(
                {
                    "scale_x": float(target_w / max(1, source_w)),
                    "scale_y": float(target_h / max(1, source_h)),
                    "resize_scale": None,
                    "resized_width": int(target_w),
                    "resized_height": int(target_h),
                    "offset_x": 0,
                    "offset_y": 0,
                    "canvas_width": int(target_w),
                    "canvas_height": int(target_h),
                }
            )

        metadata.update(
            {
                "binarize": bool(self.binarize),
                "crop_foreground": bool(self.crop_foreground),
                "preserve_aspect": bool(self.preserve_aspect),
            }
        )

        if not self.binarize:
            metadata["grayscale"] = bool(self.grayscale)
            return work.convert("L" if self.grayscale else "RGB"), metadata

        gray_image = work.convert("L")
        if self.autocontrast:
            gray_image = ImageOps.autocontrast(gray_image)
        gray = np.asarray(gray_image, dtype=np.uint8)
        threshold = self._threshold(gray)
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        if self.auto_invert and _border_mean(binary) < 127.5:
            binary = 255 - binary
        # Synthetic-style real-data mode: normalize polarity only AFTER the
        # foreground crop/geometry step. The ordinary canonical polarity above
        # is black ink on white; invert once more so the network receives white
        # handwriting on a black background, matching the requested synthetic
        # appearance.
        if self.white_ink_on_black:
            binary = 255 - binary
        metadata["white_ink_on_black"] = bool(self.white_ink_on_black)
        metadata["background_value"] = 0 if self.white_ink_on_black else 255
        metadata["ink_value"] = 255 if self.white_ink_on_black else 0
        metadata["grayscale"] = bool(self.grayscale)
        result = Image.fromarray(binary, mode="L")
        return (
            result if self.grayscale else result.convert("RGB")
        ), metadata

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.preprocess_with_metadata(image)[0]


def build_preprocessor(dataset_type: str, training: bool) -> ManuscriptLinePreprocessor:
    synthetic = str(dataset_type).lower() == "synthetic"
    real_synthetic_style = (
        not synthetic and env_flag("REAL_SYNTHETIC_STYLE", False)
    )
    enabled = env_flag("ZERO_SHOT_PREPROCESS", True)
    if not enabled:
        return ManuscriptLinePreprocessor(
            training=False,
            augment=False,
            binarize=False,
            preserve_aspect=False,
            crop_foreground=False,
            grayscale=env_flag("VISUAL_GRAYSCALE", False),
        )
    return ManuscriptLinePreprocessor(
        training=bool(training),
        augment=synthetic and env_flag("SYNTHETIC_MANUSCRIPT_AUGMENT", True),
        binarize=(
            env_flag("SYNTHETIC_BINARIZE", True)
            if synthetic
            else (
                True
                if real_synthetic_style
                else env_flag("REAL_BINARIZE", True)
            )
        ),
        method=(
            os.environ.get("SYNTHETIC_BINARIZE_METHOD", "random")
            if synthetic
            else os.environ.get("REAL_BINARIZE_METHOD", "otsu")
        ),
        fixed_threshold=env_int(
            "SYNTHETIC_BINARIZE_THRESHOLD" if synthetic else "REAL_BINARIZE_THRESHOLD",
            180,
        ),
        threshold_jitter=env_int("SYNTHETIC_THRESHOLD_JITTER", 24) if synthetic else 0,
        preserve_aspect=env_flag("ZERO_SHOT_PRESERVE_ASPECT", True),
        crop_foreground=env_flag("ZERO_SHOT_FOREGROUND_CROP", True),
        target_ink_height_ratio=env_float("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72),
        auto_invert=(
            True
            if real_synthetic_style
            else env_flag("REAL_BINARIZE_AUTO_INVERT", True)
        ),
        autocontrast=(
            True
            if real_synthetic_style
            else env_flag("REAL_BINARIZE_AUTOCONTRAST", True)
        ),
        augment_probability=env_float("SYNTHETIC_AUGMENT_PROBABILITY", 0.85),
        clean_probability=env_float("SYNTHETIC_CLEAN_PROBABILITY", 0.20),
        white_ink_on_black=real_synthetic_style,
        grayscale=env_flag("VISUAL_GRAYSCALE", False),
    )


def build_tensor_transform(dataset_type: str, training: bool):
    grayscale = env_flag("VISUAL_GRAYSCALE", False)
    mean = IMAGENET_GRAY_MEAN if grayscale else IMAGENET_MEAN
    std = IMAGENET_GRAY_STD if grayscale else IMAGENET_STD
    return transforms.Compose(
        [
            build_preprocessor(dataset_type, training),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class TransformViewDataset(Dataset):
    """Apply a split-specific PIL transform without changing the base records."""

    def __init__(self, dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _transform_image(self, image):
        if torch.is_tensor(image):
            return image
        return self.transform(image)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, dict):
            result = dict(item)
            if "image1" in result:
                result["image1"] = self._transform_image(result["image1"])
            if "image2" in result:
                result["image2"] = self._transform_image(result["image2"])
            return result
        if isinstance(item, tuple) and len(item) == 2:
            text, image = item
            return text, self._transform_image(image)
        return item


def install_dataloader_profile() -> None:
    """Install augmented train and clean validation transforms for synthetic data."""
    import DataLoader as data_loader

    if getattr(data_loader, "_zero_shot_profile_installed", False):
        return
    original_build = data_loader.build_dataloaders

    def build_dataloaders(data_dir=None):
        resolved = data_dir or data_loader._default_data_dir
        if data_loader._detect_dataset_type(resolved) != "synthetic":
            return original_build(resolved)

        # Build the base dataset with decoded PIL images.  Augmentation is applied
        # only by the training wrapper; validation and test remain deterministic.
        previous = data_loader.synthetic_transform
        data_loader.synthetic_transform = None
        try:
            full_dataset = data_loader._build_synthetic_dataset(resolved)
        finally:
            data_loader.synthetic_transform = previous
        train_subset, valid_subset, test_subset = data_loader._random_split_seeded(
            full_dataset
        )
        train_dataset = TransformViewDataset(
            train_subset, build_tensor_transform("synthetic", training=True)
        )
        clean_transform = build_tensor_transform("synthetic", training=False)
        valid_dataset = TransformViewDataset(valid_subset, clean_transform)
        test_dataset = TransformViewDataset(test_subset, clean_transform)
        return (
            data_loader._make_loader(train_dataset, shuffle=True),
            data_loader._make_loader(valid_dataset, shuffle=False),
            data_loader._make_loader(test_dataset, shuffle=False),
        )

    data_loader.build_dataloaders = build_dataloaders
    data_loader._zero_shot_profile_installed = True


def install_embedding_profile(train_module) -> None:
    """Use grouped+local features in the existing local hard-negative objective."""
    if getattr(train_module, "_zero_shot_embedding_profile_installed", False):
        return

    def compute_embeddings(image_embedder, images):
        with train_module.autocast(
            dtype=train_module.AMP_DTYPE, enabled=train_module.USE_AMP
        ):
            contextual, local, grouped, ink = image_embedder(
                images,
                return_local=True,
                return_grouped=True,
                return_ink=True,
            )
        grouped_weight = max(0.0, min(1.0, env_float("ZERO_SHOT_GROUPED_BLEND", 0.50)))
        local_grouped = (1.0 - grouped_weight) * local + grouped_weight * grouped
        return (
            F.normalize(contextual.float(), p=2, dim=-1),
            F.normalize(local_grouped.float(), p=2, dim=-1),
            ink,
            local_grouped,
        )

    train_module.compute_embeddings = compute_embeddings
    train_module._zero_shot_embedding_profile_installed = True


def _force_batch_norm_eval(module, _inputs) -> None:
    module.training = False


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _replace_batch_norm(parent: nn.Module) -> int:
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            group_norm = nn.GroupNorm(_group_count(child.num_features), child.num_features)
            if child.affine:
                with torch.no_grad():
                    group_norm.weight.copy_(child.weight)
                    group_norm.bias.copy_(child.bias)
            setattr(parent, name, group_norm)
            replaced += 1
        else:
            replaced += _replace_batch_norm(child)
    return replaced


def configure_domain_robust_normalization(model: nn.Module) -> dict:
    """Prevent BatchNorm running statistics from specializing to synthetic scans."""
    mode = os.environ.get("ZERO_SHOT_NORM_MODE", "frozen-bn").strip().lower()
    if mode in {"none", "train-bn", "batchnorm"}:
        return {"zero_shot_norm_mode": "train-bn", "zero_shot_norm_layers": 0}
    if mode == "groupnorm":
        count = _replace_batch_norm(model)
        return {"zero_shot_norm_mode": "groupnorm", "zero_shot_norm_layers": count}
    if mode not in {"frozen", "frozen-bn", "frozen_batchnorm"}:
        raise ValueError("ZERO_SHOT_NORM_MODE must be frozen-bn, groupnorm, or train-bn")

    count = 0
    handles = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            if module.affine:
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
            handles.append(module.register_forward_pre_hook(_force_batch_norm_eval))
            count += 1
    model._zero_shot_batch_norm_handles = handles
    return {"zero_shot_norm_mode": "frozen-bn", "zero_shot_norm_layers": count}


def zero_shot_config() -> dict:
    return {
        "zero_shot_profile": True,
        "zero_shot_preserve_aspect": env_flag("ZERO_SHOT_PRESERVE_ASPECT", True),
        "zero_shot_foreground_crop": env_flag("ZERO_SHOT_FOREGROUND_CROP", True),
        "zero_shot_target_ink_height_ratio": env_float(
            "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72
        ),
        "synthetic_manuscript_augment": env_flag(
            "SYNTHETIC_MANUSCRIPT_AUGMENT", True
        ),
        "synthetic_binarize": env_flag("SYNTHETIC_BINARIZE", True),
        "synthetic_binarize_method": os.environ.get(
            "SYNTHETIC_BINARIZE_METHOD", "random"
        ),
        "synthetic_threshold_jitter": env_int("SYNTHETIC_THRESHOLD_JITTER", 24),
        "zero_shot_grouped_blend": env_float("ZERO_SHOT_GROUPED_BLEND", 0.50),
        "zero_shot_norm_mode": os.environ.get("ZERO_SHOT_NORM_MODE", "frozen-bn"),
    }
