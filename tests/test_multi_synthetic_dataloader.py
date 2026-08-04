from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from multi_synthetic_dataloader import (
    BoxSafeSyntheticAugment,
    build_balanced_synthetic_dataset,
    resolve_synthetic_data_dirs,
)


class FakeTextLineModern:
    def __init__(self, new_dataset, transform, num_samples_override):
        self.new_dataset = new_dataset
        self.transform = transform
        self.num_samples = int(num_samples_override)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.new_dataset["images"], int(index)


def _source(root: Path, name: str, count: int) -> Path:
    source = root / name
    images = source / "images"
    texts = source / "texts"
    images.mkdir(parents=True)
    texts.mkdir(parents=True)
    for index in range(1, count + 1):
        (images / f"img1_{index}.png").touch()
    return source


def test_four_sources_contribute_exactly_three_each(tmp_path, monkeypatch):
    sources = [
        _source(tmp_path, f"Synthetic_Arabic_{index}", 5)
        for index in range(1, 5)
    ]
    monkeypatch.setenv("SYNTHETIC_DATA_DIRS", ",".join(map(str, sources)))
    monkeypatch.setenv("SYNTHETIC_SAMPLES_PER_DIR", "3")
    monkeypatch.setenv("SYNTHETIC_REQUIRE_FULL_PER_DIR", "1")
    loader = SimpleNamespace(
        TextLineModern=FakeTextLineModern,
        synthetic_transform="transform",
        _multi_synthetic_original_builder=lambda _path: None,
    )

    dataset = build_balanced_synthetic_dataset(loader, tmp_path)

    assert len(dataset) == 12
    assert [item["selected"] for item in dataset.synthetic_source_summaries] == [
        3,
        3,
        3,
        3,
    ]


def test_strict_source_count_fails_instead_of_silently_undersampling(
    tmp_path, monkeypatch
):
    source = _source(tmp_path, "Synthetic_Arabic_1", 2)
    monkeypatch.setenv("SYNTHETIC_DATA_DIRS", str(source))
    monkeypatch.setenv("SYNTHETIC_SAMPLES_PER_DIR", "3")
    monkeypatch.setenv("SYNTHETIC_REQUIRE_FULL_PER_DIR", "1")
    loader = SimpleNamespace(
        TextLineModern=FakeTextLineModern,
        synthetic_transform=None,
        _multi_synthetic_original_builder=lambda _path: None,
    )

    with pytest.raises(ValueError, match="only 2 contiguous samples"):
        build_balanced_synthetic_dataset(loader, tmp_path)


def test_dataset_root_discovers_arabic_one_through_four(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHETIC_DATA_DIRS", raising=False)
    expected = [
        _source(tmp_path, f"Synthetic_Arabic_{index}", 1)
        for index in range(1, 5)
    ]

    assert resolve_synthetic_data_dirs(tmp_path) == [item.resolve() for item in expected]


def test_direct_augmentation_never_changes_box_geometry(monkeypatch):
    monkeypatch.setenv("DIRECT_SUBWORD_AUGMENT_PROBABILITY", "1")
    monkeypatch.setenv("DIRECT_SUBWORD_CLEAN_PROBABILITY", "0")
    monkeypatch.setenv("DIRECT_SUBWORD_NOISE_STD_MAX", "0")
    image = Image.new("RGB", (1024, 128), color="white")

    augmented = BoxSafeSyntheticAugment()(image)

    assert augmented.size == image.size
