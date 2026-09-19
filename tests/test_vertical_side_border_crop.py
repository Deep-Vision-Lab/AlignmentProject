import numpy as np
from PIL import Image

from zero_shot_preprocessing import (
    _crop_between_vertical_borders,
    detect_vertical_side_borders,
)


def test_detects_full_height_left_and_right_side_borders():
    height, width = 120, 500
    mask = np.zeros((height, width), dtype=bool)

    # Structural side lines, slightly thick and nearly full height.
    mask[2:118, 35:39] = True
    mask[1:119, 458:462] = True

    # Interior handwriting-like strokes must not be mistaken for side borders.
    mask[45:78, 100:150] = True
    mask[38:82, 190:255] = True
    mask[50:88, 300:390] = True

    left, right, meta = detect_vertical_side_borders(mask)

    assert meta["side_border_pair_valid"] is True
    assert left is not None
    assert right is not None
    assert left[0] <= 38 and left[1] >= 36
    assert right[0] <= 461 and right[1] >= 459
    assert meta["side_border_left_coverage"] >= 0.72
    assert meta["side_border_right_coverage"] >= 0.72


def test_border_crop_uses_inside_of_side_lines_and_keeps_original_rgb(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_BORDER_VERTICAL_MARGIN", "0.10")

    height, width = 120, 500
    mask = np.zeros((height, width), dtype=bool)
    mask[1:119, 30:34] = True
    mask[1:119, 470:474] = True

    # Interior text band.
    mask[42:78, 90:420] = True

    source_array = np.zeros((height, width, 3), dtype=np.uint8)
    source_array[..., 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
    source_array[..., 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
    source_array[..., 2] = 137
    source = Image.fromarray(source_array, mode="RGB")

    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_between_vertical_borders(source, mask, detector_meta)

    assert box is not None
    x0, y0, x1, y1 = box

    # Crop begins after left structural line and ends before right structural line.
    assert x0 >= 34
    assert x1 <= 470
    assert 0 < y0 < 42
    assert 78 < y1 < height
    assert meta["crop_mode"] == "vertical_borders"

    cropped = np.asarray(source.crop(box))
    expected = source_array[y0:y1, x0:x1]
    assert np.array_equal(cropped, expected)


def test_missing_second_border_is_not_accepted():
    height, width = 100, 400
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 25:29] = True
    mask[40:70, 150:300] = True

    left, right, meta = detect_vertical_side_borders(mask)

    assert meta["side_border_pair_valid"] is False
