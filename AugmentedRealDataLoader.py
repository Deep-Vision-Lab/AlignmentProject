"""Real-dataset DataLoader with optional online or pre-materialized augmentation.

The canonical real training path lives here.  Normal real datasets keep the
existing deterministic split behavior and may opt into online augmentation.
Standalone real+injection datasets can set ``REAL_USE_EXPLICIT_SPLIT_MANIFESTS=1``
to load train/valid/test manifests exactly as written by the offline builder.
"""
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import transforms

import DataLoader as base_loader
from RealDataAugmentation import (
    AugmentedRealSubset,
    BinaryInkAugment,
    RealLinePairAugmentor,
)
from RealDataSet import ArabicManifestLinePairDataset
from real_span_feasibility import filter_subset_by_span_feasibility


class RepeatToLengthDataset(Dataset):
    """Repeat a dataset to a requested number of samples per training epoch."""

    def __init__(self, dataset: Dataset, target_length: int):
        if len(dataset) <= 0:
            raise ValueError("Cannot repeat an empty training dataset.")
        self.dataset = dataset
        self.target_length = max(len(dataset), int(target_length))

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        return self.dataset[int(index) % len(self.dataset)]


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _train_real_transform():
    """Return binarization plus post-binary training perturbations."""
    return transforms.Compose(
        [
            base_loader.ResizeAndBinarize(
                size=(128, 1024),
                enabled=base_loader._real_binarize,
                method=base_loader._real_binarize_method,
                fixed_threshold=base_loader._real_binarize_threshold,
                auto_invert=base_loader._real_binarize_auto_invert,
                autocontrast=base_loader._real_binarize_autocontrast,
            ),
            BinaryInkAugment.from_env(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _filter_positive_subsets(full_dataset, train_subset, valid_subset, test_subset):
    """Filter only impossible positive pairs while preserving split assignment."""
    enabled = _env_flag("REAL_FILTER_INFEASIBLE_SPAN_DTW", True)
    if not enabled:
        return train_subset, valid_subset, test_subset, None

    max_image_windows = int(os.environ.get("REAL_MAX_ALIGNMENT_WINDOWS", "63"))
    max_span_chars = int(
        os.environ.get(
            "MAX_TEXT_SPAN_CHARS",
            getattr(base_loader, "max_text_span_chars", 2),
        )
    )

    filtered = []
    stats = []
    for split_name, subset in (
        ("train", train_subset),
        ("valid", valid_subset),
        ("test", test_subset),
    ):
        filtered_subset, split_stats = filter_subset_by_span_feasibility(
            full_dataset,
            subset,
            split_name=split_name,
            max_image_windows=max_image_windows,
            max_span_chars=max_span_chars,
        )
        filtered.append(filtered_subset)
        stats.append(split_stats)

    summary = " ".join(
        f"{item.split_name}_kept={item.kept} "
        f"{item.split_name}_removed={item.removed}"
        for item in stats
    )
    max_required = max(item.max_required_spans for item in stats)
    examples = [example for item in stats for example in item.examples]
    print(
        "Filtered infeasible real Span-DTW positives: "
        f"max_image_windows={max_image_windows} "
        f"max_span_chars={max_span_chars} "
        f"max_required_spans_seen={max_required} "
        f"{summary}",
        flush=True,
    )
    if examples:
        print("Filtered examples: " + " | ".join(examples[:5]), flush=True)

    return (*filtered, stats)


def _manifest_dataset(path: Path) -> ArabicManifestLinePairDataset:
    if not path.is_file():
        raise FileNotFoundError(f"Explicit real split manifest not found: {path}")
    return ArabicManifestLinePairDataset(
        manifest_path=path,
        transform=base_loader.real_transform,
        text_key=os.environ.get("REAL_TEXT_KEY", "text_original_path"),
        allowed_labels=base_loader._parse_real_labels(),
        max_samples=None,
        paired=bool(base_loader.use_image_pair_contrastive),
        min_text_score=0.0,
        validate_paths=_env_flag("REAL_VALIDATE_PATHS", False),
    )


def _filter_explicit_split(dataset, split_name):
    if not _env_flag("REAL_FILTER_INFEASIBLE_SPAN_DTW", True):
        return dataset, None
    complete_subset = Subset(dataset, list(range(len(dataset))))
    return filter_subset_by_span_feasibility(
        dataset,
        complete_subset,
        split_name=split_name,
        max_image_windows=int(os.environ.get("REAL_MAX_ALIGNMENT_WINDOWS", "63")),
        max_span_chars=int(
            os.environ.get(
                "MAX_TEXT_SPAN_CHARS",
                getattr(base_loader, "max_text_span_chars", 2),
            )
        ),
    )


def _build_explicit_split_dataloaders(data_dir):
    """Load the offline-built train/valid/test manifests without re-splitting."""
    if _env_flag("REAL_AUGMENT", False):
        raise RuntimeError(
            "REAL_AUGMENT must be disabled when REAL_USE_EXPLICIT_SPLIT_MANIFESTS=1; "
            "the bbox injection augmentation is already materialized offline."
        )

    root = Path(data_dir).expanduser().resolve()
    manifests = {
        "train": root / "train_manifest.jsonl",
        "valid": root / "valid_manifest.jsonl",
        "test": root / "test_manifest.jsonl",
    }
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pre-augmented real dataset is missing explicit split manifest(s): "
            + ", ".join(missing)
        )

    train_base = _manifest_dataset(manifests["train"])
    valid_base = _manifest_dataset(manifests["valid"])
    test_base = _manifest_dataset(manifests["test"])

    train_dataset, train_stats = _filter_explicit_split(train_base, "train")
    valid_dataset, valid_stats = _filter_explicit_split(valid_base, "valid")
    test_dataset, test_stats = _filter_explicit_split(test_base, "test")

    filtered_train_samples = len(train_dataset)
    target_train_samples = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if target_train_samples > filtered_train_samples:
        train_dataset = RepeatToLengthDataset(train_dataset, target_train_samples)

    print(
        "Loaded pre-augmented real Arabic dataset from explicit manifests: "
        f"root={root} "
        f"train_manifest={len(train_base)} "
        f"train_after_feasibility={filtered_train_samples} "
        f"train_per_epoch={len(train_dataset)} "
        f"valid={len(valid_dataset)}/{len(valid_base)} "
        f"test={len(test_dataset)}/{len(test_base)} "
        "online_augment=False "
        f"binarize={base_loader._real_binarize}",
        flush=True,
    )
    if train_stats is not None:
        print(
            "Explicit split feasibility filter enabled: "
            f"train_removed={train_stats.removed} "
            f"valid_removed={valid_stats.removed if valid_stats is not None else 0} "
            f"test_removed={test_stats.removed if test_stats is not None else 0}",
            flush=True,
        )

    return (
        base_loader._make_loader(train_dataset, shuffle=True),
        base_loader._make_loader(valid_dataset, shuffle=False),
        base_loader._make_loader(test_dataset, shuffle=False),
    )


def build_dataloaders(data_dir=None):
    """Build real loaders using either explicit manifests or the legacy split."""
    if data_dir is None:
        data_dir = base_loader._default_data_dir

    if base_loader._detect_dataset_type(data_dir) != "real":
        return base_loader.build_dataloaders(data_dir)

    if _env_flag("REAL_USE_EXPLICIT_SPLIT_MANIFESTS", False):
        return _build_explicit_split_dataloaders(data_dir)

    full_dataset = base_loader._build_real_dataset(data_dir)
    split_by_pair = os.environ.get("REAL_SPLIT_BY_PAIR_ID", "1").lower() in {
        "1", "true", "yes", "on"
    }
    train_subset, valid_subset, test_subset = (
        base_loader._group_split_real_dataset(full_dataset)
        if split_by_pair
        else base_loader._random_split_seeded(full_dataset)
    )
    train_subset, valid_subset, test_subset, feasibility_stats = (
        _filter_positive_subsets(
            full_dataset,
            train_subset,
            valid_subset,
            test_subset,
        )
    )

    augmentor = RealLinePairAugmentor.from_env()
    if augmentor.enabled:
        train_dataset = AugmentedRealSubset(
            base_dataset=full_dataset,
            indices=train_subset.indices,
            transform=_train_real_transform(),
            augmentor=augmentor,
        )
    else:
        train_dataset = train_subset

    base_train_samples = len(train_dataset)
    target_train_samples = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if target_train_samples > base_train_samples:
        train_dataset = RepeatToLengthDataset(train_dataset, target_train_samples)

    print(
        "Loaded augmented real Arabic dataset: "
        f"samples={len(full_dataset)} base_train={base_train_samples} "
        f"train_per_epoch={len(train_dataset)} "
        f"valid={len(valid_subset)} test={len(test_subset)} "
        f"augment={augmentor.enabled} "
        f"stitch_prob={augmentor.stitch_probability:.3f} "
        f"appearance_prob={augmentor.appearance_probability:.3f} "
        f"stitch_max_chars={augmentor.stitch_max_text_chars} "
        f"feasibility_filter={feasibility_stats is not None} "
        f"binarize={base_loader._real_binarize} "
        f"split_by_pair_id={split_by_pair}",
        flush=True,
    )

    return (
        base_loader._make_loader(train_dataset, shuffle=True),
        base_loader._make_loader(valid_subset, shuffle=False),
        base_loader._make_loader(test_subset, shuffle=False),
    )
