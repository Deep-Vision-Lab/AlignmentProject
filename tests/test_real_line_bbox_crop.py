from pathlib import Path

from PIL import Image

from real_line_bbox_crop import bbox_crop_line, line_text_bounds


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


def test_xml_text_bounds_reconstruct_builder_vertical_padding(tmp_path):
    side = _side(tmp_path)

    # First XML line has raw y=100..141 (height=41). Dataset builder adds
    # int(0.25*41)=10px, so saved line is source y=90..151 => height 61.
    bounds, metadata = line_text_bounds(
        side,
        line_index=1,
        line_image_size=(400, 61),
        margin_ratio=0.0,
        minimum_margin_px=0,
    )

    assert bounds == (100, 10, 280, 51)
    assert metadata["builder_saved_y1"] == 90
    assert metadata["builder_saved_y2"] == 151
    assert metadata["builder_vertical_pad"] == 10
    assert metadata["all_four_sides_cropped"] if "all_four_sides_cropped" in metadata else True


def test_bbox_crop_removes_padding_on_all_four_sides_with_small_margin(tmp_path):
    side = _side(tmp_path)
    line = Image.new("RGB", (400, 61), "white")
    cropped, metadata = bbox_crop_line(
        line,
        side,
        line_index=1,
        margin_ratio=0.05,
        minimum_margin_px=2,
    )

    # Raw local box is x=100..280, y=10..51. 5% of 41px rounds to 2px.
    assert metadata["crop_left"] == 98
    assert metadata["crop_right"] == 282
    assert metadata["crop_top"] == 8
    assert metadata["crop_bottom"] == 53
    assert cropped.size == (184, 45)
    assert metadata["preserved_full_line_height"] is False
    assert metadata["all_four_sides_cropped"] is True
