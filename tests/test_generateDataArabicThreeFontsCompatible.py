from pathlib import Path

import pytest

from generateDataArabicThreeFontsCompatible import (
    _remove_render_controls,
    clean_text,
    full_height_shared_mask,
    normalize,
    output_dirs,
    visual_text,
)


def test_arabic_normalization_preserves_connected_words():
    text = "الكتاب الجديد"
    assert clean_text(text) == ["الكتاب", "الجديد"]
    assert normalize(text) == text
    assert normalize(text).count(" ") == 1
    assert "ا ل ك ت ا ب" not in normalize(text)


def test_generate_data_arabic_control_cleanup_removes_box_characters():
    dirty = "ال\u200cكتاب\u200d \ufeffالجديد"
    assert _remove_render_controls(dirty) == "الكتاب الجديد"
    assert normalize(dirty) == "الكتاب الجديد"


def test_arabic_display_shaping_does_not_insert_character_spaces():
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi.algorithm")
    displayed = visual_text("الكتاب الجديد")
    assert displayed.count(" ") == 1
    assert "ا ل" not in displayed


def test_generate_data_arabic_reshaper_deletes_at_sign():
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi.algorithm")
    assert "@" not in visual_text("الكتاب@الجديد")


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
