from pathlib import Path

from generateDataArabicThreeFontsCompatible import (
    full_height_shared_mask,
    output_dirs,
)


def test_full_height_shared_mask_uses_complete_height():
    mask = full_height_shared_mask(
        [{"role": "shared", "box": [12, 31, 44, 89]}],
        (100, 128),
    )
    assert mask.getpixel((20, 0)) == 255
    assert mask.getpixel((20, 127)) == 255
    assert mask.getpixel((5, 64)) == 0


def test_output_dirs_exclude_unwanted_artifacts(tmp_path: Path):
    paths = output_dirs(tmp_path)
    assert set(paths) == {"images", "masks", "texts"}
    assert not (tmp_path / "matrices").exists()
    assert not (tmp_path / "similarity_matrices").exists()
    assert not (tmp_path / "subword_boxes").exists()
