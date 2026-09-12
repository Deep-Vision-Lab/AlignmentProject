"""Image-only Arabic word-region pooling for word-level image alignment.

The visual encoder still produces overlapping window features. This module
segments complete word regions directly from image ink/whitespace, pools every
window that overlaps each word, and exposes one visual feature vector per word.

No transcript, OCR, or text encoder is used to define the word boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F

from Evaluation._eval_utils import ImageFeatures


@dataclass(frozen=True)
class VisualWordRegion:
    index: int
    x0: int
    x1: int
    window_indices: tuple[int, ...]

    @property
    def width(self) -> int:
        return max(0, int(self.x1) - int(self.x0))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _gray(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        return value.astype(np.float32)
    rgb = value[..., :3].astype(np.float32)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _otsu(values: np.ndarray) -> int:
    clipped = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    hist = np.bincount(clipped.reshape(-1), minlength=256).astype(np.float64)
    total = float(clipped.size)
    if total <= 0:
        return 0
    levels = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(levels, hist))
    left_weight = 0.0
    left_sum = 0.0
    best_value = -1.0
    best_threshold = 0
    for threshold in range(256):
        left_weight += hist[threshold]
        if left_weight <= 0:
            continue
        right_weight = total - left_weight
        if right_weight <= 0:
            break
        left_sum += threshold * hist[threshold]
        left_mean = left_sum / left_weight
        right_mean = (total_sum - left_sum) / right_weight
        value = left_weight * right_weight * (left_mean - right_mean) ** 2
        if value > best_value:
            best_value = value
            best_threshold = threshold
    return int(best_threshold)


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    result = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            result.append((start, previous + 1))
            start = value
        previous = value
    result.append((start, previous + 1))
    return result


def detect_visual_word_boxes(
    image: np.ndarray,
    *,
    use_flip: bool,
    min_gap_px: int | None = None,
) -> list[tuple[int, int]]:
    """Detect complete words from horizontal whitespace in a line image.

    Arabic words are generally internally connected or separated by much
    smaller gaps than inter-word spaces. We estimate foreground relative to the
    line border background, project it onto x, then bridge only short blank
    gaps. The surviving runs are word candidates.
    """
    gray = _gray(image)
    height, width = gray.shape
    border = np.concatenate(
        [gray[0], gray[-1], gray[:, 0], gray[:, -1]], axis=0
    )
    background = float(np.median(border))
    contrast = np.abs(gray - background)

    # Otsu on background contrast is stable for both white synthetic pages and
    # mildly textured manuscript backgrounds. Keep a small lower bound so
    # antialiasing/background compression noise does not become foreground.
    threshold = max(
        _env_int("WORD_ALIGNMENT_MIN_CONTRAST", 14),
        _otsu(contrast),
    )
    foreground = contrast > float(threshold)
    min_column_ink = max(
        2,
        _env_int("WORD_ALIGNMENT_MIN_COLUMN_INK", max(2, int(round(height * 0.015)))),
    )
    active = foreground.sum(axis=0) >= min_column_ink
    raw = _runs(active)
    if not raw:
        return [(0, width)]

    gap_limit = (
        max(1, int(min_gap_px))
        if min_gap_px is not None
        else max(
            1,
            _env_int(
                "WORD_ALIGNMENT_MAX_INTRAWORD_GAP_PX",
                max(8, int(round(height * 0.075))),
            ),
        )
    )

    merged: list[list[int]] = []
    for start, end in raw:
        if not merged:
            merged.append([start, end])
            continue
        gap = start - merged[-1][1]
        if gap <= gap_limit:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Expand to the midpoint of adjacent whitespace. This makes each region own
    # its complete word pixels while never crossing into its neighbor.
    boxes: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(merged):
        left = int(start)
        right = int(end)
        if index > 0:
            left = (merged[index - 1][1] + start) // 2
        if index + 1 < len(merged):
            right = (end + merged[index + 1][0]) // 2
        left = max(0, left)
        right = min(width, max(left + 1, right))
        boxes.append((left, right))

    # Feature sequences for Arabic are flipped into logical reading order, so
    # word regions must follow that same right-to-left ordering.
    boxes.sort(key=lambda item: item[0], reverse=bool(use_flip))
    return boxes


def _window_geometry(
    logical_index: int,
    *,
    n_windows: int,
    image_width: int,
    use_flip: bool,
    window_size: int,
    stride: int,
) -> tuple[float, float]:
    physical = n_windows - 1 - logical_index if use_flip else logical_index
    left = float(physical * stride)
    right = min(float(image_width), left + float(window_size))
    return left, right


def _region_weights(
    box: tuple[int, int],
    ink: torch.Tensor,
    *,
    n_windows: int,
    image_width: int,
    use_flip: bool,
    window_size: int,
    stride: int,
) -> tuple[tuple[int, ...], torch.Tensor]:
    x0, x1 = map(float, box)
    candidates = []
    values = []
    for logical in range(n_windows):
        left, right = _window_geometry(
            logical,
            n_windows=n_windows,
            image_width=image_width,
            use_flip=use_flip,
            window_size=window_size,
            stride=stride,
        )
        overlap = max(0.0, min(right, x1) - max(left, x0))
        center = 0.5 * (left + right)
        inside = x0 <= center <= x1
        fraction = overlap / max(1.0, right - left)
        if inside or fraction >= 0.25:
            candidates.append(logical)
            ink_value = float(max(0.0, ink[logical].item()))
            values.append(max(0.05, ink_value) * max(0.10, fraction))

    if not candidates:
        target = 0.5 * (x0 + x1)
        distances = []
        for logical in range(n_windows):
            left, right = _window_geometry(
                logical,
                n_windows=n_windows,
                image_width=image_width,
                use_flip=use_flip,
                window_size=window_size,
                stride=stride,
            )
            distances.append(abs(0.5 * (left + right) - target))
        nearest = int(np.argmin(distances))
        candidates = [nearest]
        values = [1.0]

    weights = ink.new_tensor(values, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    return tuple(candidates), weights


def pool_visual_words(
    features: ImageFeatures,
    image: np.ndarray,
    *,
    use_flip: bool,
    window_size: int,
    stride: int,
) -> tuple[ImageFeatures, list[VisualWordRegion]]:
    boxes = detect_visual_word_boxes(image, use_flip=use_flip)
    n_windows = int(features.contextual.shape[0])
    image_width = int(np.asarray(image).shape[1])

    local_words = []
    contextual_words = []
    grouped_words = []
    ink_words = []
    regions = []

    for word_index, box in enumerate(boxes):
        indices, weights = _region_weights(
            box,
            features.ink,
            n_windows=n_windows,
            image_width=image_width,
            use_flip=use_flip,
            window_size=int(window_size),
            stride=int(stride),
        )
        index_tensor = torch.tensor(indices, device=features.local.device, dtype=torch.long)
        local_weight = weights.to(features.local.device).unsqueeze(-1)
        contextual_weight = weights.to(features.contextual.device).unsqueeze(-1)
        grouped_weight = weights.to(features.grouped.device).unsqueeze(-1)

        local = (features.local.index_select(0, index_tensor) * local_weight).sum(dim=0)
        contextual = (
            features.contextual.index_select(0, index_tensor) * contextual_weight
        ).sum(dim=0)
        grouped = (
            features.grouped.index_select(0, index_tensor) * grouped_weight
        ).sum(dim=0)

        local_words.append(F.normalize(local.float(), p=2, dim=-1))
        contextual_words.append(F.normalize(contextual.float(), p=2, dim=-1))
        grouped_words.append(F.normalize(grouped.float(), p=2, dim=-1))
        ink_words.append(
            float(features.ink.index_select(0, index_tensor.to(features.ink.device)).mean().item())
        )
        regions.append(
            VisualWordRegion(
                index=word_index,
                x0=int(box[0]),
                x1=int(box[1]),
                window_indices=indices,
            )
        )

    if not regions:
        raise ValueError("No visual word regions detected")

    pooled = ImageFeatures(
        contextual=torch.stack(contextual_words),
        local=torch.stack(local_words),
        grouped=torch.stack(grouped_words),
        ink=features.ink.new_tensor(ink_words),
        image_size=features.image_size,
    )
    return pooled, regions


def save_word_regions_csv(path: Path, regions: list[VisualWordRegion]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["word_index", "x0", "x1", "width", "window_indices"])
        for region in regions:
            writer.writerow(
                [
                    region.index,
                    region.x0,
                    region.x1,
                    region.width,
                    " ".join(map(str, region.window_indices)),
                ]
            )


def matched_word_pairs(result, match_scores: np.ndarray, support_floor: float = 0.0):
    matrix = np.asarray(match_scores, dtype=np.float32)
    pairs = []
    for step in result.steps:
        if step.index1 is None or step.index2 is None:
            continue
        i, j = int(step.index1), int(step.index2)
        if float(matrix[i, j]) > float(support_floor):
            pairs.append((i, j))
    return pairs


def region_intervals(regions: list[VisualWordRegion], indices) -> list[list[float]]:
    unique = []
    seen = set()
    for value in indices:
        index = int(value)
        if index in seen or index < 0 or index >= len(regions):
            continue
        seen.add(index)
        region = regions[index]
        unique.append([float(region.x0), float(region.x1)])
    return unique


def save_word_alignment_visualization(
    *,
    arr1: np.ndarray,
    arr2: np.ndarray,
    regions1: list[VisualWordRegion],
    regions2: list[VisualWordRegion],
    matrix: np.ndarray,
    result,
    supported_pairs,
    output: Path,
    matrix_label: str,
) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    n1, n2 = matrix.shape
    figure_width = max(14.0, min(28.0, 1.1 * max(n1, n2)))
    heatmap_height = max(5.0, min(18.0, 0.8 * max(n1, n2)))
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(figure_width, 5.0 + heatmap_height),
        gridspec_kw={"height_ratios": [1.8, 1.8, heatmap_height]},
        constrained_layout=True,
    )

    supported1 = {i for i, _ in supported_pairs}
    supported2 = {j for _, j in supported_pairs}
    for axis, array, regions, supported, title in (
        (axes[0], arr1, regions1, supported1, "line 1: complete visual words"),
        (axes[1], arr2, regions2, supported2, "line 2: complete visual words"),
    ):
        axis.imshow(array)
        for region in regions:
            selected = region.index in supported
            axis.add_patch(
                Rectangle(
                    (region.x0, 1),
                    max(1, region.width),
                    max(1, array.shape[0] - 2),
                    fill=selected,
                    alpha=0.20 if selected else 0.0,
                    linewidth=2.0 if selected else 0.8,
                    edgecolor="red" if selected else "black",
                    facecolor="red" if selected else "none",
                )
            )
            axis.text(
                0.5 * (region.x0 + region.x1),
                3,
                f"W{region.index}",
                ha="center",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
        axis.set_title(title)
        axis.axis("off")

    image = axes[2].imshow(matrix, aspect="auto", interpolation="nearest")
    axes[2].set_title(matrix_label)
    axes[2].set_xlabel("line 2 words (reading order)")
    axes[2].set_ylabel("line 1 words (reading order)")
    axes[2].set_xticks(range(n2))
    axes[2].set_yticks(range(n1))
    axes[2].set_xticklabels([f"W{i}" for i in range(n2)])
    axes[2].set_yticklabels([f"W{i}" for i in range(n1)])

    for row in range(n1):
        for col in range(n2):
            value = float(matrix[row, col])
            if not math.isfinite(value):
                continue
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axes[2].text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if luminance > 0.58 else "white",
            )

    # Full global NW correspondence is shown with small black centers; supported
    # word matches that become masks are outlined in red.
    for i, j in result.pairs:
        axes[2].plot(j, i, "ko", markersize=2.5)
    for i, j in supported_pairs:
        axes[2].add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )
    figure.colorbar(image, ax=axes[2], fraction=0.025, pad=0.02)
    figure.savefig(output, dpi=170)
    plt.close(figure)
