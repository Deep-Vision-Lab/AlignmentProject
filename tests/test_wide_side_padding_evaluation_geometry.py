from pathlib import Path

from PIL import Image, ImageDraw

from Evaluation.yelda_geometry import prepare_line


def test_wide_side_padding_guarantees_144px_minimum_each_side(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("EVAL_SIDE_PADDING_PX", "144")
    source = Image.new("RGB", (1024, 128), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((5, 35, 1018, 92), fill="black")
    path = tmp_path / "wide_line.png"
    source.save(path)

    prepared, geometry = prepare_line(path, "synthetic", "wide_side_padding")

    assert prepared.size == (1024, 128)
    assert geometry["image_preprocessing"] == "wide_side_padding"
    assert geometry["requested_min_side_padding_px"] == 144
    assert geometry["actual_left_padding_px"] >= 144
    assert geometry["actual_right_padding_px"] >= 144
    assert geometry["resized_width"] <= 736
    assert geometry["canvas_width"] == 1024
    assert geometry["canvas_height"] == 128
    assert geometry["preserve_aspect"] is True
    assert geometry["artificial_padding"] is True
    assert abs(geometry["scale_x"] - geometry["scale_y"]) < 0.01


def test_wide_side_padding_is_configurable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EVAL_SIDE_PADDING_PX", "192")
    source = Image.new("RGB", (900, 100), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((10, 20, 889, 79), fill="black")
    path = tmp_path / "line.png"
    source.save(path)

    prepared, geometry = prepare_line(path, "real", "wide_side_padding")

    assert prepared.size == (1024, 128)
    assert geometry["requested_min_side_padding_px"] == 192
    assert geometry["actual_left_padding_px"] >= 192
    assert geometry["actual_right_padding_px"] >= 192
    assert geometry["resized_width"] <= 640
