"""Manifest-based loader for the real Arabic Quran line-pair dataset.

The dataset layout is documented in ``DATASET_README.md``. This class reads a
``dataset_manifest.jsonl`` and exposes the paired image/text contract used by
training. Optional bridge alignment masks are loaded only when a manifest side
provides ``alignment_mask_path``; ordinary real datasets remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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


class ArabicManifestIndependentLineDataset(Dataset):
    """Flatten a full pair manifest into unique independent line/text samples.

    Pair labels and pair compatibility are intentionally ignored. Every A/B side
    that has its own line image and transcript becomes one training example.
    Repeated appearances of the same line across pair combinations are
    deduplicated by (line_image_path, text_path).

    This is the correct view for the current positive image->transcript DTW
    objective, which does not require an aligned partner line.
    """

    def __init__(
        self,
        manifest_path,
        transform=None,
        text_key: str = "text_original_path",
        max_samples: Optional[int] = None,
        validate_paths: bool = False,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Real-dataset manifest not found: {self.manifest_path}"
            )
        self.root = self.manifest_path.parent
        self.transform = transform
        self.text_key = str(text_key)

        seen = set()
        samples = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.manifest_path} at line "
                        f"{line_number}: {exc}"
                    ) from exc

                for side_name in ("A", "B"):
                    side = row.get(side_name)
                    if not isinstance(side, dict):
                        continue
                    image_path = side.get("line_image_path")
                    text_path = side.get(self.text_key)
                    if not image_path or not text_path:
                        continue

                    key = (str(image_path), str(text_path))
                    if key in seen:
                        continue
                    seen.add(key)

                    # Keep all lines from the same source page in the same split.
                    page_group = (
                        side.get("original_image")
                        or side.get("page_dir")
                        or f"{row.get('pair_id', 'row')}:{side_name}"
                    )
                    samples.append(
                        {
                            "line_image_path": image_path,
                            "text_path": text_path,
                            "text_key": self.text_key,
                            "pair_id": str(page_group),
                            "source_pair_id": str(row.get("pair_id", "")),
                            "side": side_name,
                            "line_idx": int(side.get("line_idx", -1)),
                            "page_dir": str(side.get("page_dir", "")),
                            "original_image": str(side.get("original_image", "")),
                            "surah_number": row.get("surah_number"),
                            "surah_name": row.get("surah_name"),
                        }
                    )

        if max_samples is not None and int(max_samples) > 0:
            samples = samples[: min(len(samples), int(max_samples))]
        if not samples:
            raise ValueError(
                "No independent line/text examples found in "
                f"{self.manifest_path}"
            )
        self.samples = samples

        if validate_paths:
            self._validate_all_paths()

    def __len__(self):
        return len(self.samples)

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
            f"Could not resolve manifest path {path_value!r}. Tried:\n  - "
            f"{rendered}"
        )

    def _read_text(self, path_value) -> str:
        path = self._resolve(path_value)
        with path.open("r", encoding="utf-8") as handle:
            return " " + handle.read().strip() + " "

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _prepare_line_image(self, sample, image: Image.Image) -> Image.Image:
        """Deterministic real-line preparation before split-specific transforms.

        Order is intentional:
          1) crop to the source-XML text envelope on all four sides;
          2) convert to true one-channel grayscale when requested.
        Online scan augmentation is NOT applied here because validation/test must
        remain clean.  The training subset wrapper applies it afterwards.
        """
        work = image.convert("RGB")

        if self._env_flag("REAL_BBOX_CROP", False):
            from real_line_bbox_crop import bbox_crop_line

            line_idx = int(sample.get("line_idx", -1))
            side_dir = self._resolve(sample.get("page_dir", ""))
            if line_idx <= 0:
                raise ValueError(
                    "REAL_BBOX_CROP requires a positive line_idx, got "
                    f"{line_idx} for {sample.get('line_image_path')}"
                )
            try:
                work, _bbox_meta = bbox_crop_line(
                    work,
                    side_dir,
                    line_idx,
                    margin_ratio=float(
                        os.environ.get("REAL_BBOX_MARGIN_RATIO", "0.05")
                    ),
                    minimum_margin_px=int(
                        os.environ.get("REAL_BBOX_MIN_MARGIN_PX", "2")
                    ),
                )
            except Exception:
                if self._env_flag("REAL_BBOX_CROP_STRICT", True):
                    raise

        if self._env_flag("VISUAL_GRAYSCALE", False) or self._env_flag(
            "REAL_GRAYSCALE", False
        ):
            work = work.convert("L")

        return work

    def read_prepared_pil(self, idx):
        sample = self.samples[int(idx)]
        path = self._resolve(sample["line_image_path"])
        with Image.open(path) as image:
            prepared = self._prepare_line_image(sample, image)
        return self._read_text(sample["text_path"]), prepared.copy()

    def _read_image(self, path_value):
        # Backward-compatible raw reader for callers that only have a path.
        path = self._resolve(path_value)
        with Image.open(path) as image:
            image = image.convert("L" if self._env_flag("VISUAL_GRAYSCALE", False) else "RGB")
            if self.transform is not None:
                return self.transform(image)
            return image.copy()

    def _validate_all_paths(self) -> None:
        for sample_idx, sample in enumerate(self.samples):
            try:
                self._resolve(sample["line_image_path"])
                self._resolve(sample["text_path"])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Invalid independent-line sample "
                    f"index={sample_idx}: {exc}"
                ) from exc

    def __getitem__(self, idx):
        sample = self.samples[int(idx)]
        image = self._read_image(sample["line_image_path"])
        text = self._read_text(sample["text_path"])
        return text, image



class ArabicAllPageLinesDataset(Dataset):
    """Use every available real line image with its own transcript.

    This view does NOT depend on line-pair alignment labels. It scans
    ``DatasetPairs/page_pairs/pair_*/A|B`` directly, so a line is eligible even
    when it has no positive/aligned partner in any line-pair manifest.

    The same source page may be copied into several candidate page-pair
    directories. We identify a page by the SHA1 of its copied
    ``original_image.png`` and deduplicate line numbers within that page. This
    prevents page-pair construction from artificially multiplying the training
    set while still retaining every distinct manuscript line.
    """

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        dataset_root,
        transform=None,
        text_key: str = "text_original_path",
        max_samples: Optional[int] = None,
        validate_paths: bool = False,
    ):
        self.root = Path(dataset_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"ArabicDataset root not found: {self.root}")
        self.transform = transform
        self.text_key = str(text_key)

        if self.text_key == "text_original_path":
            text_rel = Path("text/final/original")
        elif self.text_key == "text_tashkeel_path":
            text_rel = Path("text/final/tashkeel")
        else:
            raise ValueError(
                "ArabicAllPageLinesDataset currently supports "
                "text_original_path or text_tashkeel_path, got "
                f"{self.text_key!r}"
            )

        page_pairs_root = self.root / "DatasetPairs" / "page_pairs"
        if not page_pairs_root.is_dir():
            raise FileNotFoundError(
                "Expected real page-pair directory at "
                f"{page_pairs_root}"
            )

        page_fingerprint_cache = {}
        seen_page_lines = set()
        samples = []
        scanned_side_copies = 0
        duplicate_page_line_copies = 0
        missing_transcripts = 0

        for pair_dir in sorted(page_pairs_root.glob("pair_*")):
            if not pair_dir.is_dir():
                continue
            for side_name in ("A", "B"):
                side_dir = pair_dir / side_name
                lines_dir = side_dir / "linesImages"
                text_dir = side_dir / text_rel
                if not lines_dir.is_dir() or not text_dir.is_dir():
                    continue
                scanned_side_copies += 1

                page_key, original_image = self._page_fingerprint(
                    side_dir, lines_dir, page_fingerprint_cache
                )

                image_paths = sorted(
                    path
                    for path in lines_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in self._IMAGE_SUFFIXES
                )
                for image_path in image_paths:
                    transcript_path = text_dir / f"{image_path.stem}.txt"
                    if not transcript_path.is_file():
                        missing_transcripts += 1
                        continue

                    line_identity = (page_key, image_path.stem)
                    if line_identity in seen_page_lines:
                        duplicate_page_line_copies += 1
                        continue
                    seen_page_lines.add(line_identity)

                    match = re.search(r"(\d+)$", image_path.stem)
                    line_idx = int(match.group(1)) if match else -1
                    samples.append(
                        {
                            "line_image_path": self._relative(image_path),
                            "text_path": self._relative(transcript_path),
                            "text_key": self.text_key,
                            "pair_id": page_key,
                            "source_pair_id": str(pair_dir.name),
                            "side": side_name,
                            "line_idx": line_idx,
                            "page_dir": self._relative(side_dir),
                            "original_image": (
                                self._relative(original_image)
                                if original_image is not None
                                else ""
                            ),
                            "line_source": "all_page_lines_scan",
                        }
                    )

        if max_samples is not None and int(max_samples) > 0:
            samples = samples[: min(len(samples), int(max_samples))]
        if not samples:
            raise ValueError(
                "No line image/transcript examples found by scanning "
                f"{page_pairs_root}"
            )

        self.samples = samples
        self.scan_stats = {
            "unique_lines": len(samples),
            "unique_pages": len({sample["pair_id"] for sample in samples}),
            "scanned_side_copies": int(scanned_side_copies),
            "duplicate_page_line_copies_removed": int(
                duplicate_page_line_copies
            ),
            "missing_transcripts": int(missing_transcripts),
        }

        if validate_paths:
            self._validate_all_paths()

    def __len__(self):
        return len(self.samples)

    def _relative(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    @staticmethod
    def _sha1_file(path: Path) -> str:
        digest = hashlib.sha1()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _page_fingerprint(self, side_dir, lines_dir, cache):
        original_candidates = sorted(
            path
            for path in side_dir.glob("original_image.*")
            if path.is_file()
        )
        original = original_candidates[0] if original_candidates else None
        cache_key = str(original or lines_dir)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached, original

        if original is not None:
            fingerprint = "page_sha1:" + self._sha1_file(original)
        else:
            digest = hashlib.sha1()
            image_paths = sorted(
                path
                for path in lines_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in self._IMAGE_SUFFIXES
            )
            for image_path in image_paths:
                digest.update(self._sha1_file(image_path).encode("ascii"))
            fingerprint = "page_lines_sha1:" + digest.hexdigest()

        cache[cache_key] = fingerprint
        return fingerprint, original

    def _resolve(self, path_value) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.root / path).resolve()

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

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _prepare_line_image(self, sample, image: Image.Image) -> Image.Image:
        """Prepare one real line without changing transcript/window order.

        Deterministic order:
          1) use source XML boxes to crop LEFT/RIGHT/TOP/BOTTOM padding;
          2) convert to true one-channel grayscale when enabled.

        Scan augmentation is deliberately excluded here so validation/test stay
        clean. The training-only wrapper applies it after this step.
        """
        work = image.convert("RGB")
        if self._env_flag("REAL_BBOX_CROP", False):
            from real_line_bbox_crop import bbox_crop_line

            line_idx = int(sample.get("line_idx", -1))
            if line_idx <= 0:
                raise ValueError(
                    "REAL_BBOX_CROP requires a positive line_idx, got "
                    f"{line_idx} for {sample.get('line_image_path')}"
                )
            side_dir = self._resolve(sample.get("page_dir", ""))
            try:
                work, _bbox_meta = bbox_crop_line(
                    work,
                    side_dir,
                    line_idx,
                    margin_ratio=float(
                        os.environ.get("REAL_BBOX_MARGIN_RATIO", "0.05")
                    ),
                    minimum_margin_px=int(
                        os.environ.get("REAL_BBOX_MIN_MARGIN_PX", "2")
                    ),
                )
            except Exception:
                if self._env_flag("REAL_BBOX_CROP_STRICT", True):
                    raise

        if self._env_flag("VISUAL_GRAYSCALE", False) or self._env_flag(
            "REAL_GRAYSCALE", False
        ):
            work = work.convert("L")
        return work

    def read_prepared_pil(self, idx):
        sample = self.samples[int(idx)]
        path = self._resolve(sample["line_image_path"])
        with Image.open(path) as image:
            prepared = self._prepare_line_image(sample, image)
        return self._read_text(sample["text_path"]), prepared.copy()

    def _validate_all_paths(self) -> None:
        for sample_idx, sample in enumerate(self.samples):
            image = self._resolve(sample["line_image_path"])
            text = self._resolve(sample["text_path"])
            if not image.is_file() or not text.is_file():
                raise FileNotFoundError(
                    "Invalid all-page-line sample "
                    f"index={sample_idx} image={image} text={text}"
                )

    def __getitem__(self, idx):
        text, image = self.read_prepared_pil(idx)
        if self.transform is not None:
            image = self.transform(image)
        return text, image
