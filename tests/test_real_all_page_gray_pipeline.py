import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from RealDataSet import ArabicAllPageLinesDataset
from zero_shot_preprocessing import build_preprocessor, build_tensor_transform


XML = """<?xml version="1.0"?>
<ArrayOfDocumentElement>
  <DocumentElement><X>100</X><Y>100</Y><Width>80</Width><Height>40</Height><Transcript>a</Transcript></DocumentElement>
  <DocumentElement><X>220</X><Y>105</Y><Width>60</Width><Height>36</Height><Transcript>b</Transcript></DocumentElement>
</ArrayOfDocumentElement>
"""


def _make_dataset(root: Path):
    side = root / "DatasetPairs" / "page_pairs" / "pair_000001" / "A"
    lines = side / "linesImages"
    text = side / "text" / "final" / "original"
    lines.mkdir(parents=True)
    text.mkdir(parents=True)

    page = Image.new("RGB", (400, 600), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 100, 180, 140), fill="black")
    draw.rectangle((220, 105, 280, 141), fill="black")
    page.save(side / "original_image.png")
    (side / "original.xml").write_text(XML, encoding="utf-8")

    # Reproduce the source builder: raw y=[100,145), 25% vertical padding.
    # raw_h=45 => pad=11, saved y=[89,156), full page width retained.
    line = page.crop((0, 89, 400, 156))
    line.save(lines / "line_01.png")
    (text / "line_01.txt").write_text("اب", encoding="utf-8")


def test_all_page_line_pipeline_is_four_side_crop_then_true_grayscale(monkeypatch, tmp_path):
    _make_dataset(tmp_path)

    monkeypatch.setenv("REAL_BBOX_CROP", "1")
    monkeypatch.setenv("REAL_BBOX_CROP_STRICT", "1")
    monkeypatch.setenv("REAL_BBOX_MARGIN_RATIO", "0.05")
    monkeypatch.setenv("REAL_BBOX_MIN_MARGIN_PX", "2")
    monkeypatch.setenv("VISUAL_GRAYSCALE", "1")
    monkeypatch.setenv("REAL_GRAYSCALE", "1")
    monkeypatch.setenv("REAL_BINARIZE", "0")
    monkeypatch.setenv("ZERO_SHOT_PREPROCESS", "1")
    monkeypatch.setenv("ZERO_SHOT_FOREGROUND_CROP", "0")
    monkeypatch.setenv("ZERO_SHOT_PRESERVE_ASPECT", "0")
    monkeypatch.setenv("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")

    dataset = ArabicAllPageLinesDataset(
        tmp_path,
        transform=None,
        text_key="text_original_path",
        validate_paths=True,
    )
    _text, prepared = dataset.read_prepared_pil(0)

    assert prepared.mode == "L"
    assert prepared.width < 400
    assert prepared.height < 67

    preprocessor = build_preprocessor("real", training=False)
    direct, meta = preprocessor.preprocess_with_metadata(prepared)
    assert direct.mode == "L"
    assert direct.size == (1024, 128)
    assert preprocessor.preserve_aspect is False
    assert meta["preserve_aspect"] is False
    assert meta["offset_x"] == 0
    assert meta["offset_y"] == 0
    assert meta["resized_width"] == 1024
    assert meta["resized_height"] == 128
    assert meta["scale_x"] != meta["scale_y"]

    dataset.transform = build_tensor_transform("real", training=False)
    _text, tensor = dataset[0]
    assert torch.is_tensor(tensor)
    assert tuple(tensor.shape) == (1, 128, 1024)
    assert torch.isfinite(tensor).all()
