from pathlib import Path

from PIL import Image

from real_line_bbox_crop import bbox_crop_line, line_horizontal_bounds


XML = """<?xml version="1.0"?>
<ArrayOfDocumentElement>
  <DocumentElement><X>100</X><Y>100</Y><Width>80</Width><Height>40</Height><Transcript>a</Transcript></DocumentElement>
  <DocumentElement><X>220</X><Y>105</Y><Width>60</Width><Height>36</Height><Transcript>b</Transcript></DocumentElement>
  <DocumentElement><X>50</X><Y>300</Y><Width>90</Width><Height>44</Height><Transcript>c</Transcript></DocumentElement>
  <DocumentElement><X>180</X><Y>305</Y><Width>100</Width><Height>42</Height><Transcript>d</Transcript></DocumentElement>
</ArrayOfDocumentElement>
"""


def _side(tmp_path: Path):
    side = tmp_path / "A"
    side.mkdir()
    (side / "original.xml").write_text(XML, encoding="utf-8")
    Image.new("RGB", (400, 600), "white").save(side / "original_image.png")
    return side


def test_xml_line_bounds_map_directly_to_full_width_line_image(tmp_path):
    side = _side(tmp_path)
    bounds, metadata = line_horizontal_bounds(
        side,
        line_index=1,
        line_image_width=400,
        margin_px=0,
        line_image_height=80,
    )
    assert bounds == (100, 280)
    assert metadata["xml_line_count"] == 2
    assert metadata["scale_x"] == 1.0


def test_bbox_crop_preserves_full_line_height_and_adds_safe_margin(tmp_path):
    side = _side(tmp_path)
    line = Image.new("RGB", (400, 80), "white")
    cropped, metadata = bbox_crop_line(
        line,
        side,
        line_index=1,
        margin_ratio_of_line_height=0.20,
        minimum_margin_px=8,
    )
    # 20% of 80px = 16px: [100-16, 280+16].
    assert cropped.size == (212, 80)
    assert metadata["crop_left"] == 84
    assert metadata["crop_right"] == 296
    assert metadata["preserved_full_line_height"] is True
