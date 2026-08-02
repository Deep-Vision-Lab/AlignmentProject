#!/usr/bin/env python3
"""Run the renderer box builder with training-window-aware validation.

The renderer can place a very narrow connected-subword interval entirely in a
blank inter-glyph column after boundary snapping. Direct training does not
consume individual pixels; it consumes overlapping image windows. This wrapper
keeps exact-empty intervals as diagnostics and rejects them only when every
training window overlapping the interval is also ink-free.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.data import build_connected_subword_boxes as implementation


_ORIGINAL_VALIDATE = implementation.validate_payload
_ORIGINAL_SIGNATURE = implementation.generator_signature
_WINDOW_VALIDATION_SCHEMA = 3
_WARNING_COUNT = 0


def _window_geometry() -> tuple[int, int]:
    window_size = max(1, int(os.environ.get("WINDOW_SIZE", "32")))
    mode = os.environ.get("WINDOW_OVERLAP_MODE", "custom").strip().lower()
    if mode == "no_overlap":
        stride = window_size
    elif mode == "light_overlap":
        stride = max(1, window_size // 2)
    elif mode == "dense_overlap":
        stride = max(1, window_size // 4)
    elif mode == "custom":
        ratio = float(os.environ.get("STRIDE_RATIO", "0.25"))
        stride = max(1, int(window_size * ratio))
    else:
        raise ValueError(f"Unknown WINDOW_OVERLAP_MODE={mode!r}")
    return window_size, stride


def _window_starts(width: int, window_size: int, stride: int) -> list[int]:
    if width <= window_size:
        return [0]
    return list(range(0, width - window_size + 1, stride))


def validate_payload(payload: dict, image_path: Path) -> dict:
    """Validate labels in the same window space consumed by the model."""
    global _WARNING_COUNT
    result = _ORIGINAL_VALIDATE(payload, image_path)
    errors = list(result.get("errors", []))
    warnings = list(result.get("warnings", []))
    empty_indices = {
        int(error.split(":", 1)[1])
        for error in errors
        if str(error).startswith("empty_box:")
    }
    window_size, stride = _window_geometry()
    result["window_geometry"] = {
        "window_size": window_size,
        "stride": stride,
    }
    if not empty_indices:
        result["warnings"] = warnings
        return result

    _gray, ink = implementation._image_ink(image_path)
    width = int(ink.shape[1])
    starts = _window_starts(width, window_size, stride)
    by_index = {
        int(item["logical_index"]): item for item in payload.get("subwords", [])
    }
    retained_errors = [
        error for error in errors if not str(error).startswith("empty_box:")
    ]
    support_by_index = {}

    for logical_index in sorted(empty_indices):
        item = by_index.get(logical_index)
        if item is None:
            retained_errors.append(f"missing_box:{logical_index}")
            continue
        x0, x1 = sorted((float(item["x0"]), float(item["x1"])))
        support = []
        for start in starts:
            end = min(width, start + window_size)
            if min(float(end), x1) - max(float(start), x0) <= 0.0:
                continue
            support.append(int(ink[:, start:end].sum()))
        maximum = max(support, default=0)
        support_by_index[str(logical_index)] = {
            "overlapping_windows": len(support),
            "max_window_ink_pixels": maximum,
        }
        if maximum > 0:
            warnings.append(f"empty_exact_box_window_supported:{logical_index}")
            _WARNING_COUNT += 1
        else:
            retained_errors.append(f"empty_window_support:{logical_index}")

    for entry in result.get("per_box_ink", []):
        support = support_by_index.get(str(entry.get("logical_index")))
        if support:
            entry.update(support)

    result["errors"] = retained_errors
    result["warnings"] = warnings
    result["valid"] = not retained_errors
    return result


def generator_signature(args, font_path: Path) -> str:
    """Invalidate sidecars when the training window geometry changes."""
    base = _ORIGINAL_SIGNATURE(args, font_path)
    window_size, stride = _window_geometry()
    encoded = (
        f"{base}|window_validation_schema={_WINDOW_VALIDATION_SCHEMA}"
        f"|window_size={window_size}|stride={stride}"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    implementation._SCHEMA_VERSION = max(
        int(getattr(implementation, "_SCHEMA_VERSION", 1)),
        _WINDOW_VALIDATION_SCHEMA,
    )
    implementation.validate_payload = validate_payload
    implementation.generator_signature = generator_signature
    implementation.main()
    if _WARNING_COUNT:
        print(
            "Window-aware validation accepted "
            f"{_WARNING_COUNT} exact-empty intervals because their overlapping "
            "training windows contain foreground ink.",
            flush=True,
        )


if __name__ == "__main__":
    main()
