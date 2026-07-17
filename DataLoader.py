import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # kept for pad_matrices helper
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from DataSet import TextLineModern
from RealDataSet import ArabicManifestLinePairDataset
from Parameters import *


_default_data_dir = f"DataSet/Synthetic_{lang}"

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _RESAMPLE_BILINEAR = Image.BILINEAR


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _otsu_threshold(gray_array: np.ndarray) -> int:
    """Return an Otsu threshold for an uint8 grayscale image."""
    values = np.asarray(gray_array, dtype=np.uint8)
    histogram = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127

    levels = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(levels, histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break

        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (total_sum - background_sum) / foreground_weight
        between_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = threshold

    return int(best_threshold)


class ResizeAndBinarize:
    """Resize a real line image and optionally binarize it.

    Binarization is performed after resizing so the final network input remains
    strictly black/white.  ``auto_invert`` checks the image border and ensures
    that the page background is white rather than black.
    """

    def __init__(
        self,
        size=(128, 1024),
        enabled=True,
        method="otsu",
        fixed_threshold=180,
        auto_invert=True,
        autocontrast=True,
    ):
        self.height = int(size[0])
        self.width = int(size[1])
        self.enabled = bool(enabled)
        self.method = str(method).lower()
        self.fixed_threshold = int(fixed_threshold)
        self.auto_invert = bool(auto_invert)
        self.autocontrast = bool(autocontrast)

        if self.method not in {"otsu", "fixed"}:
            raise ValueError(
                f"REAL_BINARIZE_METHOD must be 'otsu' or 'fixed', got {self.method!r}."
            )

    @staticmethod
    def _border_mean(binary: np.ndarray) -> float:
        height, width = binary.shape
        border_h = max(1, int(round(height * 0.05)))
        border_w = max(1, int(round(width * 0.01)))
        border = np.concatenate(
            [
                binary[:border_h, :].reshape(-1),
                binary[-border_h:, :].reshape(-1),
                binary[:, :border_w].reshape(-1),
                binary[:, -border_w:].reshape(-1),
            ]
        )
        return float(border.mean()) if border.size else 255.0

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L").resize(
            (self.width, self.height),
            _RESAMPLE_BILINEAR,
        )

        if not self.enabled:
            return image.convert("RGB")

        if self.autocontrast:
            image = ImageOps.autocontrast(image)

        gray = np.asarray(image, dtype=np.uint8)
        threshold = (
            _otsu_threshold(gray)
            if self.method == "otsu"
            else max(0, min(255, self.fixed_threshold))
        )
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)

        # Real datasets may contain scans with inverted polarity.  A cropped
        # handwriting line normally has page background along most of its border.
        if self.auto_invert and self._border_mean(binary) < 127.5:
            binary = 255 - binary

        return Image.fromarray(binary, mode="L").convert("RGB")


# Synthetic data keeps its original preprocessing.
synthetic_transform = transforms.Compose([
    transforms.Resize((128, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_real_binarize = os.environ.get("REAL_BINARIZE", "1").lower() in {
    "1", "true", "yes", "on"
}
_real_binarize_auto_invert = os.environ.get(
    "REAL_BINARIZE_AUTO_INVERT", "1"
).lower() in {"1", "true", "yes", "on"}
_real_binarize_autocontrast = os.environ.get(
    "REAL_BINARIZE_AUTOCONTRAST", "1"
).lower() in {"1", "true", "yes", "on"}
_real_binarize_method = os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
_real_binarize_threshold = int(os.environ.get("REAL_BINARIZE_THRESHOLD", 180))

real_transform = transforms.Compose([
    ResizeAndBinarize(
        size=(128, 1024),
        enabled=_real_binarize,
        method=_real_binarize_method,
        fixed_threshold=_real_binarize_threshold,
        auto_invert=_real_binarize_auto_invert,
        autocontrast=_real_binarize_autocontrast,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Backward-compatible name used by older callers.
transform = synthetic_transform


# ---------- Tunables for the DataLoader (override via env vars) ----------
_requested_num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", 4))
_allow_jax_workers = os.environ.get("ALLOW_JAX_DATALOADER_WORKERS", "0").lower() in {
    "1", "true", "yes", "on"
}
if span_dtw_backend == "jax" and not _allow_jax_workers:
    # JAX is multithreaded; forking DataLoader workers after JAX starts can
    # deadlock or duplicate large process state. Keep JAX span-DTW jobs single
    # process unless explicitly overridden.
    _num_workers = 0
else:
    _num_workers = _requested_num_workers
_pin_memory = torch.cuda.is_available()
_persistent_workers = _num_workers > 0
_prefetch_factor = int(os.environ.get("DATALOADER_PREFETCH", 4)) if _num_workers > 0 else None
_split_seed = int(os.environ.get("DATASET_SPLIT_SEED", 42))


def _make_loader(ds, shuffle):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_persistent_workers,
        drop_last=False,
    )
    if _prefetch_factor is not None:
        kwargs["prefetch_factor"] = _prefetch_factor
    return DataLoader(ds, **kwargs)


def _detect_dataset_type(data_dir) -> str:
    requested = os.environ.get("DATASET_TYPE", "auto").strip().lower()
    if requested not in {"auto", "synthetic", "real"}:
        raise ValueError(
            f"DATASET_TYPE must be auto, synthetic, or real; got {requested!r}."
        )
    if requested != "auto":
        return requested

    path = Path(data_dir)
    if path.suffix.lower() == ".jsonl" or (path / "dataset_manifest.jsonl").is_file():
        return "real"
    return "synthetic"


def _real_manifest_path(data_dir) -> Path:
    path = Path(data_dir)
    if path.suffix.lower() == ".jsonl":
        return path
    manifest_name = os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")
    return path / manifest_name


def _parse_real_labels():
    raw = os.environ.get("REAL_DATASET_LABELS", "high_match,medium_match").strip()
    if raw.lower() in {"all", "*", "any"}:
        return None
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    if not labels:
        raise ValueError(
            "REAL_DATASET_LABELS is empty. Use high_match,medium_match or 'all'."
        )
    return labels


def _build_synthetic_dataset(data_dir):
    images_dir = os.path.join(data_dir, "images")
    texts_dir = os.path.join(data_dir, "texts")
    if not os.path.isdir(images_dir) or not os.path.isdir(texts_dir):
        raise FileNotFoundError(
            f"Synthetic dataset expects images/ and texts/ under {data_dir}."
        )

    detected = len([
        name
        for name in os.listdir(images_dir)
        if name.startswith("img1_") and name.endswith(".png")
    ])
    sample_cap = min(int(num_samples), detected) if detected > 0 else int(num_samples)
    dataset_paths = {
        "images": images_dir,
        "matrices": os.path.join(data_dir, "matrices"),
        "diffNWmatrices": os.path.join(data_dir, "diffNWmatrices"),
        "similarity_matrices": os.path.join(data_dir, "similarity_matrices"),
        "texts": texts_dir,
    }
    return TextLineModern(
        new_dataset=dataset_paths,
        transform=synthetic_transform,
        num_samples_override=sample_cap,
    )


def _build_real_dataset(data_dir):
    labels = _parse_real_labels()
    text_key = os.environ.get("REAL_TEXT_KEY", "text_original_path")
    min_text_score = float(os.environ.get("REAL_MIN_TEXT_SCORE", 0.0))
    validate_paths = os.environ.get("REAL_VALIDATE_PATHS", "0").lower() in {
        "1", "true", "yes", "on"
    }
    max_samples = int(num_samples) if int(num_samples) > 0 else None

    return ArabicManifestLinePairDataset(
        manifest_path=_real_manifest_path(data_dir),
        transform=real_transform,
        text_key=text_key,
        allowed_labels=labels,
        max_samples=max_samples,
        paired=bool(use_image_pair_contrastive),
        min_text_score=min_text_score,
        validate_paths=validate_paths,
    )


def _split_lengths(dataset_length):
    train_size = int(0.6 * dataset_length)
    valid_size = int(0.2 * dataset_length)
    test_size = dataset_length - train_size - valid_size
    return train_size, valid_size, test_size


def _random_split_seeded(full_dataset):
    lengths = _split_lengths(len(full_dataset))
    return random_split(
        full_dataset,
        lengths,
        generator=torch.Generator().manual_seed(_split_seed),
    )


def _group_split_real_dataset(full_dataset):
    """Split by page-pair id so related lines cannot leak across splits."""
    groups = {}
    for sample_index, sample in enumerate(full_dataset.samples):
        pair_id = str(sample.get("pair_id", f"sample_{sample_index}"))
        groups.setdefault(pair_id, []).append(sample_index)

    if len(groups) < 3:
        return _random_split_seeded(full_dataset)

    group_ids = list(groups)
    random.Random(_split_seed).shuffle(group_ids)

    # Allocate complete groups while approximately matching the 60/20/20 sample
    # targets.  Group boundaries are more important than exact split counts.
    total = len(full_dataset)
    train_target = int(0.6 * total)
    valid_target = int(0.2 * total)
    train_indices = []
    valid_indices = []
    test_indices = []

    for group_id in group_ids:
        indices = groups[group_id]
        if len(train_indices) < train_target:
            train_indices.extend(indices)
        elif len(valid_indices) < valid_target:
            valid_indices.extend(indices)
        else:
            test_indices.extend(indices)

    # Defensive fallback for very uneven manifests.
    if not train_indices or not valid_indices or not test_indices:
        return _random_split_seeded(full_dataset)

    return (
        Subset(full_dataset, train_indices),
        Subset(full_dataset, valid_indices),
        Subset(full_dataset, test_indices),
    )


def build_dataloaders(data_dir=None):
    """Build train/valid/test loaders for synthetic or real Arabic lines.

    Auto detection:
      - a directory containing ``dataset_manifest.jsonl`` is treated as real;
      - a direct ``.jsonl`` path is treated as real;
      - otherwise the old synthetic ``images/`` + ``texts/`` layout is used.

    Override with ``DATASET_TYPE=synthetic`` or ``DATASET_TYPE=real``.
    """
    if data_dir is None:
        data_dir = _default_data_dir

    dataset_type = _detect_dataset_type(data_dir)
    if dataset_type == "real":
        full_dataset = _build_real_dataset(data_dir)
        split_by_pair = os.environ.get("REAL_SPLIT_BY_PAIR_ID", "1").lower() in {
            "1", "true", "yes", "on"
        }
        splits = (
            _group_split_real_dataset(full_dataset)
            if split_by_pair
            else _random_split_seeded(full_dataset)
        )
        print(
            "Loaded real Arabic manifest dataset: "
            f"samples={len(full_dataset)} labels={_parse_real_labels() or 'all'} "
            f"text_key={full_dataset.text_key} binarize={_real_binarize} "
            f"method={_real_binarize_method} split_by_pair_id={split_by_pair}",
            flush=True,
        )
    else:
        full_dataset = _build_synthetic_dataset(data_dir)
        splits = _random_split_seeded(full_dataset)
        print(
            f"Loaded synthetic dataset: samples={len(full_dataset)} data_dir={data_dir}",
            flush=True,
        )

    train_ds, valid_ds, test_ds = splits
    print(
        f"Dataset split sizes: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}",
        flush=True,
    )
    return (
        _make_loader(train_ds, shuffle=True),
        _make_loader(valid_ds, shuffle=False),
        _make_loader(test_ds, shuffle=False),
    )


train_dataloader = None
valid_dataloader = None
test_dataloader = None


# ---- Hard Negative Generators (applied to the POSITIVE text) ----

_DOT_CONFUSIONS = {
    "ب": ["ت", "ث", "ن", "ي"], "ت": ["ب", "ث", "ن", "ي"],
    "ث": ["ب", "ت", "ن", "ي"], "ن": ["ب", "ت", "ث", "ي"],
    "ي": ["ب", "ت", "ث", "ن"], "ج": ["ح", "خ"], "ح": ["ج", "خ"],
    "خ": ["ج", "ح"], "د": ["ذ"], "ذ": ["د"], "ر": ["ز"], "ز": ["ر"],
    "س": ["ش"], "ش": ["س"], "ص": ["ض"], "ض": ["ص"],
    "ط": ["ظ"], "ظ": ["ط"], "ع": ["غ"], "غ": ["ع"],
    "ف": ["ق"], "ق": ["ف"],
}
_ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


def _hard_neg_crop(text):
    words = text.split()
    if len(words) < 2:
        return text
    cutoff = max(1, len(words) // 2)
    return " ".join(words[:cutoff])


def _hard_neg_drop(text):
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words) - 1)
    return " ".join(words[:idx] + words[idx + 1:])


def _hard_neg_shuffle(text):
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def _hard_neg_dot_confusion(text, p=0.25):
    chars = list(text)
    changed = False
    for i, ch in enumerate(chars):
        if ch in _DOT_CONFUSIONS and random.random() < p:
            chars[i] = random.choice(_DOT_CONFUSIONS[ch])
            changed = True
    if not changed:
        confusable = [(i, c) for i, c in enumerate(chars) if c in _DOT_CONFUSIONS]
        if confusable:
            idx, ch = random.choice(confusable)
            chars[idx] = random.choice(_DOT_CONFUSIONS[ch])
    return "".join(chars)


def _hard_neg_word_shuffle(text):
    words = text.split()
    if len(words) < 2:
        return text
    random.shuffle(words)
    return " ".join(words)


def _hard_neg_same_length_random(text):
    return "".join(random.choice(_ARABIC_LETTERS) if ch != " " else " " for ch in text)


_hard_neg_fns = [_hard_neg_crop, _hard_neg_drop, _hard_neg_shuffle]
_lc_neg_fns = [_hard_neg_word_shuffle, _hard_neg_same_length_random]


def _ensure_different(neg, pos):
    if neg.strip() != pos.strip():
        return neg
    chars = list(pos.strip())
    if len(chars) > 1:
        random.shuffle(chars)
        candidate = "".join(chars)
        if candidate.strip() != pos.strip():
            return candidate
    if len(pos.strip()) > 1:
        return pos.strip()[:-1]
    return pos + "‌"


def _maybe_crop(text):
    if random.random() < 0.3 and len(text) > 3:
        crop_ratio = random.uniform(0.5, 0.9)
        crop_len = max(2, int(len(text) * crop_ratio))
        start = random.randint(0, len(text) - crop_len)
        return text[start:start + crop_len]
    return text


def _build_negatives_for_sample(pos_text, all_pos_texts, sample_idx, mode):
    mode = mode.lower()
    sample_negs = []

    if mode == "mixed":
        num_hard = min(2, num_negatives)
        num_random = num_negatives - num_hard
        for _ in range(num_hard):
            fn = random.choice(_hard_neg_fns)
            sample_negs.append(_ensure_different(fn(pos_text), pos_text))
        available = [j for j, _t in enumerate(all_pos_texts) if j != sample_idx] or [sample_idx]
        sampled = random.sample(available, num_random) if len(available) >= num_random else [random.choice(available) for _ in range(num_random)]
        for j in sampled:
            sample_negs.append(_ensure_different(_maybe_crop(all_pos_texts[j]), pos_text))
    elif mode == "length_controlled":
        fns = _lc_neg_fns + [_hard_neg_word_shuffle]
        for fn in fns[:min(len(fns), num_negatives)]:
            sample_negs.append(_ensure_different(fn(pos_text), pos_text))
        while len(sample_negs) < num_negatives:
            sample_negs.append(_ensure_different(_hard_neg_word_shuffle(pos_text), pos_text))
    elif mode == "dot_confusion":
        for _ in range(num_negatives):
            sample_negs.append(_ensure_different(_hard_neg_dot_confusion(pos_text), pos_text))
    elif mode == "same_length_random":
        for _ in range(num_negatives):
            sample_negs.append(_ensure_different(_hard_neg_same_length_random(pos_text), pos_text))
    elif mode == "shuffle_only":
        for _ in range(num_negatives):
            sample_negs.append(_ensure_different(_hard_neg_word_shuffle(pos_text), pos_text))
    else:
        return _build_negatives_for_sample(pos_text, all_pos_texts, sample_idx, "mixed")

    return sample_negs


def custom_collate_fn(batch):
    """Collate synthetic or manifest samples for contrastive learning."""
    try:
        from Parameters import negative_mode as _neg_mode
    except ImportError:
        _neg_mode = "mixed"

    if isinstance(batch[0], dict):
        texts1 = [b["text1"] for b in batch]
        texts2 = [b["text2"] for b in batch]
        images1 = torch.stack([b["image1"] for b in batch], dim=0)
        images2 = torch.stack([b["image2"] for b in batch], dim=0)
        neg_texts1 = [
            _build_negatives_for_sample(texts1[i], texts1, i, _neg_mode)
            for i in range(len(texts1))
        ]
        neg_texts2 = [
            _build_negatives_for_sample(texts2[i], texts2, i, _neg_mode)
            for i in range(len(texts2))
        ]
        return {
            "images1": images1,
            "texts1": texts1,
            "neg_texts1": neg_texts1,
            "images2": images2,
            "texts2": texts2,
            "neg_texts2": neg_texts2,
        }

    texts1, images1 = zip(*batch)
    images = torch.stack(images1, dim=0)
    pos_texts = list(texts1)
    neg_texts = [
        _build_negatives_for_sample(pos_texts[i], pos_texts, i, _neg_mode)
        for i in range(len(pos_texts))
    ]
    return images, pos_texts, neg_texts


# Optional helper kept for callers that need padded matrices
def pad_matrices(matrices, smooth=False, kernel_size=5, sigma=1.0):
    if not matrices:
        return torch.empty(0)

    device = matrices[0].device
    max_dim = max(max(mat.shape) for mat in matrices)

    gaussian_kernel_2d = None
    if smooth:
        _x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device, dtype=torch.float32)
        _gauss1d = torch.exp(-_x.pow(2) / (2 * sigma**2))
        _gauss1d /= _gauss1d.sum()
        gaussian_kernel_2d = torch.outer(_gauss1d, _gauss1d).unsqueeze(0).unsqueeze(0)

    processed_matrices = []
    for mat in matrices:
        mat = mat.to(device)
        mat_unsqueezed = mat.unsqueeze(0).unsqueeze(0)
        processed_mat = F.interpolate(mat_unsqueezed, size=(max_dim, max_dim), mode="nearest")
        if smooth and gaussian_kernel_2d is not None:
            current_kernel = gaussian_kernel_2d.to(processed_mat.device)
            padding = kernel_size // 2
            processed_mat = F.conv2d(processed_mat, current_kernel, padding=padding)
        processed_matrices.append(processed_mat.squeeze(0).squeeze(0))

    return torch.stack(processed_matrices, dim=0).to(device)


if __name__ == "__main__":
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders()
    for batch in train_dataloader:
        if isinstance(batch, dict):
            print(f"Batch paired images1 shape: {batch['images1'].shape}")
            print(f"Batch paired images2 shape: {batch['images2'].shape}")
            print(f"Texts1: {len(batch['texts1'])}, Texts2: {len(batch['texts2'])}")
            print(f"Neg texts per sample: {len(batch['neg_texts1'][0])} (num_negatives={num_negatives})")
        else:
            images, pos_texts, neg_texts = batch
            print(f"Batch images shape: {images.shape}")
            print(f"Pos texts: {len(pos_texts)}")
            print(f"Neg texts per sample: {len(neg_texts[0])} (num_negatives={num_negatives})")
        break
