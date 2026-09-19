import numpy as np
from PIL import Image, ImageDraw

from zero_shot_preprocessing import build_preprocessor


def test_real_synthetic_style_is_cropped_white_ink_on_black(monkeypatch):
    monkeypatch.setenv("REAL_SYNTHETIC_STYLE", "1")
    monkeypatch.setenv("ZERO_SHOT_PREPROCESS", "1")
    monkeypatch.setenv("ZERO_SHOT_FOREGROUND_CROP", "1")
    monkeypatch.setenv("ZERO_SHOT_PRESERVE_ASPECT", "1")
    monkeypatch.setenv("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")

    source = Image.new("RGB", (600, 120), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((120, 42, 480, 78), fill="black")

    preprocessor = build_preprocessor("real", training=False)
    processed, metadata = preprocessor.preprocess_with_metadata(source)

    assert processed.size == (1024, 128)
    assert metadata["crop_foreground"] is True
    assert metadata["preserve_aspect"] is True
    assert metadata["white_ink_on_black"] is True
    assert metadata["background_value"] == 0
    assert metadata["ink_value"] == 255

    gray = np.asarray(processed.convert("L"), dtype=np.uint8)
    border = np.concatenate(
        [
            gray[:4, :].reshape(-1),
            gray[-4:, :].reshape(-1),
            gray[:, :4].reshape(-1),
            gray[:, -4:].reshape(-1),
        ]
    )
    assert float(border.mean()) < 5.0
    assert int(gray.max()) == 255
    assert int(gray.min()) == 0


def test_real_synthetic_style_forces_otsu_binarization(monkeypatch):
    monkeypatch.setenv("REAL_SYNTHETIC_STYLE", "1")
    monkeypatch.setenv("REAL_BINARIZE", "0")

    source = Image.new("RGB", (300, 80), "white")
    draw = ImageDraw.Draw(source)
    draw.rectangle((40, 25, 260, 55), fill=(70, 70, 70))

    preprocessor = build_preprocessor("real", training=False)
    processed = preprocessor(source)
    unique = set(np.unique(np.asarray(processed.convert("L"))).tolist())

    assert unique.issubset({0, 255})
    assert 0 in unique
    assert 255 in unique
