import numpy as np
from PIL import Image

from zero_shot_preprocessing import (
    _crop_between_vertical_borders,
    _crop_with_partial_side_borders,
    detect_horizontal_frame_borders,
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
    assert meta["crop_mode"] == "vertical_borders_text_projection"

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


def test_single_side_border_rejects_neighboring_line_from_above(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_BORDER_CROP_INSET", "2")

    height, width = 160, 640
    mask = np.zeros((height, width), dtype=bool)

    # One real structural side border.
    mask[2:158, 36:40] = True
    # Leaked preceding manuscript line touching the top edge.
    mask[0:24, 70:570] = True
    # Intended line, centered in the crop.
    mask[70:108, 95:535] = True
    # A few diacritic-like fragments around the intended line.
    mask[62:67, 180:205] = True
    mask[111:115, 360:382] = True

    source = Image.new("RGB", (width, height), (238, 231, 214))
    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_with_partial_side_borders(source, mask, detector_meta)

    assert box is not None
    x0, y0, x1, y1 = box
    assert meta["crop_mode"] == "single_left_frame_border"
    assert x0 >= 40
    assert x1 > 535
    # The leaked line above is removed while the intended line and safety
    # margin remain.
    assert y0 > 24
    assert y0 < 70
    assert y1 > 108
    assert y1 <= height


def test_no_side_borders_rejects_neighboring_line_from_above():
    height, width = 150, 620
    mask = np.zeros((height, width), dtype=bool)

    # Leaked line from above, deliberately wide.
    mask[0:20, 30:590] = True
    # Intended target line near the vertical center.
    mask[64:102, 110:510] = True
    mask[57:61, 210:230] = True
    mask[105:109, 390:410] = True

    source = Image.new("RGB", (width, height), (240, 233, 218))
    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_with_partial_side_borders(source, mask, detector_meta)

    assert box is not None
    x0, y0, x1, y1 = box
    assert meta["crop_mode"] == "horizontal_or_text_frame_projection"
    assert y0 > 20
    assert y0 < 64
    assert y1 > 102
    assert x0 < 110
    assert x1 > 510


def test_detects_top_and_bottom_horizontal_frame_inside_side_borders(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_MIN_COVERAGE", "0.55")

    height, width = 150, 600
    mask = np.zeros((height, width), dtype=bool)

    # Complete rectangular manuscript frame.
    mask[2:148, 35:39] = True
    mask[2:148, 560:564] = True
    mask[22:26, 36:563] = True
    mask[126:130, 36:563] = True

    # Target handwriting stays well inside the frame.
    mask[58:92, 95:520] = True
    mask[50:55, 180:200] = True
    mask[96:101, 360:382] = True

    left, right, _ = detect_vertical_side_borders(mask)
    assert left is not None and right is not None
    interior = mask[:, left[1] + 2 : right[0] - 2]
    top, bottom, meta = detect_horizontal_frame_borders(interior)

    assert top is not None
    assert bottom is not None
    assert meta["horizontal_border_pair_valid"] is True
    assert top[0] <= 25 and top[1] >= 23
    assert bottom[0] <= 129 and bottom[1] >= 127


def test_full_rectangular_frame_is_cropped_on_all_four_sides(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_MIN_COVERAGE", "0.55")

    height, width = 160, 640
    mask = np.zeros((height, width), dtype=bool)
    mask[1:159, 30:35] = True
    mask[1:159, 605:610] = True
    mask[18:23, 31:609] = True
    mask[137:142, 31:609] = True
    mask[61:103, 90:560] = True

    source = Image.new("RGB", (width, height), (240, 232, 214))
    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_between_vertical_borders(source, mask, detector_meta)

    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 > 35
    assert x1 < 605
    assert y0 > 23
    assert y1 < 137
    assert meta["crop_mode"] == "full_frame_borders"
    assert meta["vertical_crop_used_top_frame"] is True
    assert meta["vertical_crop_used_bottom_frame"] is True


def test_l_shaped_top_frame_connected_to_one_side_is_cropped(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_MIN_COVERAGE", "0.50")

    height, width = 170, 680
    mask = np.zeros((height, width), dtype=bool)

    # L-shaped frame: one vertical side connected to a long top rule.
    mask[2:168, 34:39] = True
    mask[20:25, 35:590] = True

    # Target text below the rule.
    mask[76:116, 100:555] = True
    mask[68:73, 190:215] = True
    mask[120:125, 385:405] = True

    source = Image.new("RGB", (width, height), (239, 231, 215))
    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_with_partial_side_borders(source, mask, detector_meta)

    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 > 39
    assert x1 > 555
    assert y0 > 25
    assert y0 < 76
    assert y1 > 116
    assert meta["vertical_crop_used_top_frame"] is True
    assert meta["crop_mode"].endswith("_with_horizontal_frame")


def test_bottom_horizontal_frame_without_vertical_sides_is_removed(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_CROP_INSET", "2")
    monkeypatch.setenv("ZERO_SHOT_HORIZONTAL_BORDER_MIN_COVERAGE", "0.55")

    height, width = 155, 620
    mask = np.zeros((height, width), dtype=bool)

    mask[48:88, 105:515] = True
    mask[40:45, 215:235] = True
    mask[92:97, 390:410] = True
    # Long structural rule below the text.
    mask[124:129, 55:575] = True

    source = Image.new("RGB", (width, height), (241, 234, 219))
    detector_meta = {
        "crop_raw_support_left": 0,
        "crop_raw_support_top": 0,
        "crop_raw_support_right": width,
        "crop_raw_support_bottom": height,
    }
    box, meta = _crop_with_partial_side_borders(source, mask, detector_meta)

    assert box is not None
    _x0, y0, _x1, y1 = box
    assert y0 < 48
    assert y1 < 124
    assert y1 > 88
    assert meta["vertical_crop_used_bottom_frame"] is True
