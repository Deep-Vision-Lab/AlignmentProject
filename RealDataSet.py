"""Manifest-based loader for the real Arabic Quran line-pair dataset.

The dataset layout is documented in ``DATASET_README.md``. This class reads a
``dataset_manifest.jsonl`` and exposes the paired image/text contract used by
training. Optional bridge alignment masks are loaded only when a manifest side
provides ``alignment_mask_path``; ordinary real datasets remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ArabicManifestLinePairDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        transform=None,
        text_key: str = "text_original_path",
        allowed_labels: Optional[Sequence[str]] = ("high_match", "medium_match"),
        max_samples: Optional[int] = None,
        paired: bool = True,
        min_text_score: float = 0.0,
        validate_paths: bool = False,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Real-dataset manifest not found: {self.manifest_path}")
        self.root = self.manifest_path.parent
        self.transform = transform
        self.text_key = str(text_key)
        self.paired = bool(paired)
        self.min_text_score = float(min_text_score)
        self.allowed_labels = (
            None
            if allowed_labels is None
            else {str(label).strip() for label in allowed_labels if str(label).strip()}
        )

        samples = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    sample = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.manifest_path} at line {line_number}: {exc}"
                    ) from exc
                label = str(sample.get("label_type", ""))
                if self.allowed_labels is not None and label not in self.allowed_labels:
                    continue
                text_score = float((sample.get("scores") or {}).get("text_score", 0.0))
                if text_score < self.min_text_score:
                    continue
                self._validate_manifest_row(sample, line_number)
                samples.append(sample)

        if max_samples is not None and int(max_samples) > 0:
            samples = samples[: min(len(samples), int(max_samples))]
        if not samples:
            labels = "all" if self.allowed_labels is None else sorted(self.allowed_labels)
            raise ValueError(
                "No real Arabic line pairs remain after filtering. "
                f"manifest={self.manifest_path}, labels={labels}, "
                f"min_text_score={self.min_text_score}"
            )
        self.samples = samples
        if validate_paths:
            self._validate_all_paths()

    def __len__(self):
        return len(self.samples)

    def _validate_manifest_row(self, sample: dict, line_number: int) -> None:
        for side_name in ("A", "B"):
            side = sample.get(side_name)
            if not isinstance(side, dict):
                raise KeyError(
                    f"Manifest line {line_number} is missing dictionary side {side_name!r}."
                )
            for key in ("line_image_path", self.text_key):
                if not side.get(key):
                    raise KeyError(
                        f"Manifest line {line_number}, side {side_name}, is missing {key!r}."
                    )

    def _candidate_paths(self, path_value) -> Iterable[Path]:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            yield path
            return
        yield self.root / path
        yield Path.cwd() / path
        yield self.root.parent / path

    def _resolve(self, path_value) -> Path:
        candidates = []
        for candidate in self._candidate_paths(path_value):
            candidate = candidate.resolve()
            candidates.append(candidate)
            if candidate.exists():
                return candidate
        rendered = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not resolve manifest path {path_value!r}. Tried:\n  - {rendered}"
        )

    def _read_text(self, path_value) -> str:
        path = self._resolve(path_value)
        with path.open("r", encoding="utf-8") as handle:
            return " " + handle.read().strip() + " "

    def _read_image(self, path_value):
        path = self._resolve(path_value)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                return self.transform(image)
            return image.copy()

    def _read_alignment_mask(self, path_value, image2):
        """Load a binary 0/1 mask at exactly the post-transform image geometry."""
        path = self._resolve(path_value)
        if torch.is_tensor(image2):
            height, width = int(image2.shape[-2]), int(image2.shape[-1])
        else:
            width, height = image2.size
        with Image.open(path) as image:
            image = image.convert("L").resize((width, height), Image.Resampling.NEAREST)
            values = np.asarray(image, dtype=np.uint8).copy()
        binary = torch.from_numpy((values >= 128).astype(np.float32)).unsqueeze(0)
        return binary

    def _validate_all_paths(self) -> None:
        for sample_idx, sample in enumerate(self.samples):
            for side_name in ("A", "B"):
                side = sample[side_name]
                try:
                    self._resolve(side["line_image_path"])
                    self._resolve(side[self.text_key])
                    if side.get("alignment_mask_path"):
                        self._resolve(side["alignment_mask_path"])
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Invalid paths in real dataset sample index {sample_idx}, "
                        f"pair_id={sample.get('pair_id')}, side={side_name}: {exc}"
                    ) from exc

    def __getitem__(self, idx):
        sample = self.samples[int(idx)]
        side_a, side_b = sample["A"], sample["B"]
        image1 = self._read_image(side_a["line_image_path"])
        text1 = self._read_text(side_a[self.text_key])
        if not self.paired:
            return text1, image1

        image2 = self._read_image(side_b["line_image_path"])
        scores = sample.get("scores") or {}
        result = {
            "text1": text1,
            "image1": image1,
            "text2": self._read_text(side_b[self.text_key]),
            "image2": image2,
            "pair_id": str(sample.get("pair_id", idx)),
            "label_type": str(sample.get("label_type", "")),
            "text_score": float(scores.get("text_score", 0.0)),
            "avg_sim": float(scores.get("avg_sim", 0.0)),
            "coverage_A": float(scores.get("coverage_A", 0.0)),
            "coverage_B": float(scores.get("coverage_B", 0.0)),
            "line1_index": int(side_a.get("line_idx", -1)),
            "line2_index": int(side_b.get("line_idx", -1)),
        }
        mask_path = side_b.get("alignment_mask_path")
        if mask_path:
            result["alignment_mask2"] = self._read_alignment_mask(mask_path, image2)
            result["alignment_mask2_path"] = str(mask_path)
        bridge = sample.get("bridge") or {}
        if bridge:
            result["bridge_shared_island_count"] = int(
                bridge.get("shared_island_count", 0) or 0
            )
            result["bridge_shared_texts"] = list(bridge.get("shared_texts") or [])
            result["bridge_shared_boxes_px"] = list(bridge.get("shared_boxes_px") or [])
        return result
