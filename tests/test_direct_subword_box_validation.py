from PIL import Image, ImageDraw

from scripts.data import build_connected_subword_boxes_window_validated as windowed


def _payload(x0, x1):
    return {
        "subwords": [
            {
                "text": "ب",
                "x0": float(x0),
                "x1": float(x1),
                "logical_index": 0,
                "word_index": 0,
            }
        ]
    }


def test_exact_empty_box_is_valid_when_training_window_contains_ink(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "line.png"
    image = Image.new("L", (64, 16), color=0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((29, 3, 31, 12), fill=255)
    image.save(image_path)

    monkeypatch.setenv("WINDOW_SIZE", "16")
    monkeypatch.setenv("WINDOW_OVERLAP_MODE", "custom")
    monkeypatch.setenv("STRIDE_RATIO", "0.5")

    result = windowed.validate_payload(_payload(20, 24), image_path)

    assert result["valid"]
    assert result["errors"] == []
    assert "empty_exact_box_window_supported:0" in result["warnings"]
    assert result["window_geometry"] == {"window_size": 16, "stride": 8}
    assert result["per_box_ink"][0]["max_window_ink_pixels"] > 0


def test_box_stays_invalid_when_no_overlapping_window_contains_ink(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "line.png"
    Image.new("L", (64, 16), color=0).save(image_path)

    monkeypatch.setenv("WINDOW_SIZE", "16")
    monkeypatch.setenv("WINDOW_OVERLAP_MODE", "custom")
    monkeypatch.setenv("STRIDE_RATIO", "0.5")

    result = windowed.validate_payload(_payload(20, 24), image_path)

    assert not result["valid"]
    assert "empty_window_support:0" in result["errors"]
