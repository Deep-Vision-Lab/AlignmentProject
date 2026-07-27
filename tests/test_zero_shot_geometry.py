import numpy as np
from PIL import Image

import zero_shot_preprocessing as preprocessing
from zero_shot_geometry import (
    install_source_compatible_geometry,
    source_compatible_pad,
)


def _ink_bbox(image):
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ys, xs = np.nonzero(gray < 128)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _long_line(inverted=False):
    background, ink = (0, 255) if inverted else (255, 0)
    array = np.full((180, 2400), background, dtype=np.uint8)
    array[62:126, 90:2320] = ink
    array[42:64, 280:340] = ink
    array[45:66, 1850:1910] = ink
    return Image.fromarray(array, mode="L")


def test_source_geometry_keeps_long_line_at_target_ink_height():
    output = source_compatible_pad(
        _long_line(),
        size=(128, 1024),
        target_ink_height_ratio=0.72,
    )

    assert output.size == (1024, 128)
    x0, y0, x1, y1 = _ink_bbox(output)
    ink_height = y1 - y0
    assert 86 <= ink_height <= 96
    assert x1 - x0 >= 950


def test_source_geometry_normalizes_dark_page_background_before_padding():
    output = source_compatible_pad(
        _long_line(inverted=True),
        size=(128, 1024),
        target_ink_height_ratio=0.72,
    )
    gray = np.asarray(output.convert("L"), dtype=np.uint8)

    border = np.concatenate(
        [gray[0], gray[-1], gray[:, 0], gray[:, -1]]
    )
    assert float(border.mean()) > 245.0
    assert int(gray.min()) < 32


def test_runtime_install_replaces_width_limited_geometry(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_SOURCE_GEOMETRY", "1")
    original = preprocessing.aspect_preserving_pad
    try:
        preprocessing._source_compatible_geometry_installed = False
        assert install_source_compatible_geometry()
        assert preprocessing.aspect_preserving_pad is source_compatible_pad
        config = preprocessing.zero_shot_config()
        assert config["zero_shot_geometry_mode"] == "source-compatible-height"
        assert config["zero_shot_source_geometry"] is True
    finally:
        preprocessing.aspect_preserving_pad = original
        preprocessing._source_compatible_geometry_installed = False
