from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from Evaluation.yelda_geometry import prepare_line


def test_cropped_1024_removes_outer_padding_then_resizes_exactly(tmp_path: Path):
    source = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((40, 20, 199, 59), fill="black")
    path = tmp_path / "line.png"
    source.save(path)

    prepared, geometry = prepare_line(path, "synthetic", "cropped_1024")

    assert prepared.size == (1024, 128)
    assert geometry["image_preprocessing"] == "cropped_1024"
    assert geometry["canvas_width"] == 1024
    assert geometry["canvas_height"] == 128
    assert geometry["offset_x"] == 0
    assert geometry["offset_y"] == 0
    assert geometry["artificial_padding"] is False
    assert geometry["crop_foreground"] is True
    assert geometry["preserve_aspect"] is False

    # The crop itself contains only the foreground rectangle, so after direct
    # resizing there must be no re-added white side/top/bottom canvas.
    array = np.asarray(prepared)
    assert float(array.mean()) < 5.0


def test_cropped_1024_records_source_to_canvas_scale(tmp_path: Path):
    source = Image.new("RGB", (300, 100), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((50, 25, 249, 74), fill="black")
    path = tmp_path / "line.png"
    source.save(path)

    prepared, geometry = prepare_line(path, "real", "cropped_1024")

    assert prepared.size == (1024, 128)
    assert geometry["crop_width"] > 0
    assert geometry["crop_height"] > 0
    assert geometry["scale_x"] == 1024 / geometry["crop_width"]
    assert geometry["scale_y"] == 128 / geometry["crop_height"]
