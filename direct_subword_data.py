"""Synthetic connected-subword interval sidecars and DataLoader integration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import torch


def flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def enabled() -> bool:
    return flag("DIRECT_SUBWORD_SUPERVISION", False)


def sidecar_path(image_path: str | os.PathLike[str]) -> Path:
    image_path = Path(image_path)
    match = re.fullmatch(r"img([12])_(\d+)", image_path.stem)
    if match is None:
        raise ValueError(
            "Expected synthetic image name img1_N.png or img2_N.png, got "
            f"{image_path.name!r}."
        )
    explicit = os.environ.get("DIRECT_SUBWORD_BOX_DIR", "").strip()
    root = (
        Path(explicit).expanduser()
        if explicit
        else image_path.parent.parent / "subword_boxes"
    )
    return root / f"subwords{match.group(1)}_{match.group(2)}.json"


def load_sidecar(path: str | os.PathLike[str]) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("subwords") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Invalid direct-subword sidecar: {path}")
    minimum = max(0.0, number("DIRECT_SUBWORD_MIN_BOX_WIDTH", 1.0))
    result = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            text = str(item["text"]).strip()
            x0, x1 = float(item["x0"]), float(item["x1"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and x1 - x0 >= minimum:
            result.append(
                {
                    "text": text,
                    "x0": x0,
                    "x1": x1,
                    "logical_index": int(item.get("logical_index", index)),
                    "word_index": int(item.get("word_index", -1)),
                }
            )
    if not result:
        raise ValueError(f"No usable connected-subword intervals in {path}")
    return result


def window_overlap_weights(
    x0: float,
    x1: float,
    *,
    num_windows: int,
    window_size: int,
    stride: int,
    use_flip: bool,
    device=None,
) -> torch.Tensor:
    if num_windows <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("Window count, size, and stride must be positive")
    x0, x1 = sorted((float(x0), float(x1)))
    starts = torch.arange(num_windows, dtype=torch.float32, device=device) * stride
    ends = starts + window_size
    overlap = (
        torch.minimum(ends, ends.new_tensor(x1))
        - torch.maximum(starts, starts.new_tensor(x0))
    ).clamp_min(0.0)
    return torch.flip(overlap, dims=[0]) if use_flip else overlap


def install_dataset_patch() -> None:
    """Attach sidecar regions to synthetic samples and preserve them in collate."""
    import DataLoader as loader_module
    import DataSet as dataset_module

    dataset_cls = dataset_module.TextLineModern
    if getattr(dataset_cls, "_direct_subword_patched", False):
        return
    original_getitem = dataset_cls.__getitem__

    def patched_getitem(self, index):
        item = original_getitem(self, index)
        if not enabled() or not isinstance(item, dict):
            return item
        record = self._sample_records[int(index)]
        cache = getattr(self, "_direct_subword_cache", None)
        if cache is None:
            cache = self._direct_subword_cache = {}

        def regions(key: str) -> list[dict]:
            image = record[key]
            if image not in cache:
                path = sidecar_path(image)
                if not path.is_file():
                    if flag("DIRECT_SUBWORD_STRICT_BOXES", True):
                        raise FileNotFoundError(
                            f"Missing {path}. Run "
                            "scripts/data/build_connected_subword_boxes.py first."
                        )
                    cache[image] = []
                else:
                    cache[image] = load_sidecar(path)
            return cache[image]

        item = dict(item)
        item["subwords1"] = regions("image1")
        item["subwords2"] = regions("image2")
        item["sample_index"] = int(index)
        return item

    dataset_cls.__getitem__ = patched_getitem
    dataset_cls._direct_subword_patched = True
    original_collate = loader_module.custom_collate_fn

    def patched_collate(batch):
        result = original_collate(batch)
        if enabled() and isinstance(result, dict):
            result["subwords1"] = [item.get("subwords1", []) for item in batch]
            result["subwords2"] = [item.get("subwords2", []) for item in batch]
            result["sample_indices"] = [item.get("sample_index", -1) for item in batch]
        return result

    loader_module.custom_collate_fn = patched_collate
