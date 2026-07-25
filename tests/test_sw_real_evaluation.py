from types import SimpleNamespace

import numpy as np
from PIL import Image

from Evaluation.eval_img_align_sw import (
    _display_image,
    _load_pair_manifest,
    _real_pair_paths,
)


def _write_test_line(path, invert=False):
    image = np.full((40, 160), 230, dtype=np.uint8)
    image[12:28, 20:60] = 35
    image[8:32, 90:135] = 70
    if invert:
        image = 255 - image
    Image.fromarray(image, mode="L").save(path)


def test_real_pair_discovery_supports_line_patterns_and_extension_fallback(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _write_test_line(images / "line1_7.jpg")
    _write_test_line(images / "line2_7.jpeg")

    args = SimpleNamespace(
        data_dir=str(tmp_path),
        image1_pattern="line1_{index}.png",
        image2_pattern="line2_{index}.png",
    )
    pair = _real_pair_paths(args, 7)

    assert pair.index == 7
    assert pair.image1.name == "line1_7.jpg"
    assert pair.image2.name == "line2_7.jpeg"


def test_real_display_uses_binary_model_input_shape(tmp_path):
    path = tmp_path / "line.png"
    _write_test_line(path, invert=True)

    array = _display_image(path, "real")

    assert array.shape == (128, 1024, 3)
    assert set(np.unique(array).tolist()).issubset({0, 255})
    assert float(array[[0, -1], :, :].mean()) > 127.5


def test_real_csv_manifest_resolves_paths_relative_to_manifest(tmp_path):
    images = tmp_path / "lines"
    images.mkdir()
    _write_test_line(images / "left.png")
    _write_test_line(images / "right.png")
    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "index,image1,image2\n12,lines/left.png,lines/right.png\n",
        encoding="utf-8",
    )

    pairs = _load_pair_manifest(manifest, tmp_path)

    assert len(pairs) == 1
    assert pairs[0].index == 12
    assert pairs[0].image1 == images / "left.png"
    assert pairs[0].image2 == images / "right.png"
