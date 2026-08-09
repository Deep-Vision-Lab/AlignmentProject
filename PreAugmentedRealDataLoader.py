"""Load the standalone real+injection dataset from explicit split manifests.

This loader is used only when ``REAL_USE_EXPLICIT_SPLIT_MANIFESTS=1``.  It
preserves the train/valid/test assignment created by
``build_real_augmented_same_skeleton.py`` instead of splitting the combined
manifest again.

The dataset is already augmented offline, so online real augmentation is
intentionally rejected.  Validation and test therefore remain the untouched
original-real samples written by the dataset builder.
"""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

from torch.utils.data import Dataset, Subset

import DataLoader as base_loader
from RealDataSet import ArabicManifestLinePairDataset
from real_span_feasibility import filter_subset_by_span_feasibility


class RepeatToLengthDataset(Dataset):
    """Repeat a filtered training set to a requested per-epoch length."""

    def __init__(self, dataset: Dataset, target_length: int):
        if len(dataset) <= 0:
            raise ValueError("Cannot repeat an empty real training dataset.")
        self.dataset = dataset
        self.target_length = max(len(dataset), int(target_length))

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        return self.dataset[int(index) % len(self.dataset)]


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        # Offline injected pairs do not inherit a meaningful old text_score.
        # The standalone builder validates them structurally instead.
        min_text_score=0.0,
        validate_paths=_flag("REAL_VALIDATE_PATHS", False),
    )


def _filter_feasible(dataset: Dataset, split_name: str):
    if not _flag("REAL_FILTER_INFEASIBLE_SPAN_DTW", True):
        return dataset, None

    max_image_windows = int(os.environ.get("REAL_MAX_ALIGNMENT_WINDOWS", "63"))
    max_span_chars = int(
        os.environ.get(
            "MAX_TEXT_SPAN_CHARS",
            getattr(base_loader, "max_text_span_chars", 2),
        )
    )
    complete_subset = Subset(dataset, list(range(len(dataset))))
    return filter_subset_by_span_feasibility(
        dataset,
        complete_subset,
        split_name=split_name,
        max_image_windows=max_image_windows,
        max_span_chars=max_span_chars,
    )


def _origin_counts(dataset: ArabicManifestLinePairDataset) -> dict[str, int]:
    counts = Counter(
        str(sample.get("sample_origin", "unspecified")) for sample in dataset.samples
    )
    return dict(sorted(counts.items()))


def build_dataloaders(data_dir=None):
    """Build loaders from train/valid/test manifests without re-splitting."""
    if data_dir is None:
        raise ValueError("Explicit real split loading requires a dataset directory.")

    root = Path(data_dir).expanduser().resolve()
    required = {
        "train": root / "train_manifest.jsonl",
        "valid": root / "valid_manifest.jsonl",
        "test": root / "test_manifest.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pre-augmented real dataset is missing explicit split manifest(s): "
            + ", ".join(missing)
        )

    if _flag("REAL_AUGMENT", False):
        raise RuntimeError(
            "REAL_AUGMENT must be disabled for ArabicDatasetRealAug10K because "
            "its injection augmentation is already materialized offline."
        )

    train_base = _manifest_dataset(required["train"])
    valid_base = _manifest_dataset(required["valid"])
    test_base = _manifest_dataset(required["test"])

    train_ds, train_stats = _filter_feasible(train_base, "train")
    valid_ds, valid_stats = _filter_feasible(valid_base, "valid")
    test_ds, test_stats = _filter_feasible(test_base, "test")

    filtered_train_length = len(train_ds)
    target_train_samples = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if target_train_samples > filtered_train_length:
        train_ds = RepeatToLengthDataset(train_ds, target_train_samples)

    print(
        "Loaded pre-augmented real Arabic dataset from explicit manifests: "
        f"root={root} "
        f"train_manifest={len(train_base)} "
        f"train_after_feasibility={filtered_train_length} "
        f"train_per_epoch={len(train_ds)} "
        f"valid={len(valid_ds)}/{len(valid_base)} "
        f"test={len(test_ds)}/{len(test_base)} "
        f"train_origins={_origin_counts(train_base)} "
        f"online_augment=False "
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
        base_loader._make_loader(train_ds, shuffle=True),
        base_loader._make_loader(valid_ds, shuffle=False),
        base_loader._make_loader(test_ds, shuffle=False),
    )
