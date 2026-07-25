import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from Evaluation.eval_img_align_sw import (
    ImagePair,
    _display_image,
    _group_split_pairs,
    _load_arabic_dataset_pairs,
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


def test_real_display_uses_binary_model_input_shape(tmp_path, monkeypatch):
    path = tmp_path / "line.png"
    _write_test_line(path, invert=True)
    monkeypatch.setenv("REAL_BINARIZE_METHOD", "otsu")
    monkeypatch.setenv("REAL_BINARIZE_AUTO_INVERT", "1")
    monkeypatch.setenv("REAL_BINARIZE_AUTOCONTRAST", "1")

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


def _arabic_record(pair_id, label, image_a, image_b, text_a, text_b, score=0.9):
    return {
        "pair_id": pair_id,
        "label_type": label,
        "scores": {"text_score": score, "avg_sim": 0.8},
        "A": {
            "line_image_path": image_a,
            "text_original_path": text_a,
            "line_idx": 3,
        },
        "B": {
            "line_image_path": image_b,
            "text_original_path": text_b,
            "line_idx": 7,
        },
    }


def test_arabic_dataset_manifest_uses_nested_sides_and_training_filters(tmp_path):
    lines = tmp_path / "lines"
    texts = tmp_path / "texts"
    lines.mkdir()
    texts.mkdir()
    for name in ("a1.png", "b1.png", "a2.png", "b2.png"):
        _write_test_line(lines / name)
    for name in ("a1.txt", "b1.txt", "a2.txt", "b2.txt"):
        (texts / name).write_text("نص عربي", encoding="utf-8")

    records = [
        _arabic_record(
            "page_pair_1",
            "high_match",
            "lines/a1.png",
            "lines/b1.png",
            "texts/a1.txt",
            "texts/b1.txt",
            score=0.95,
        ),
        _arabic_record(
            "page_pair_2",
            "no_shared_content",
            "lines/a2.png",
            "lines/b2.png",
            "texts/a2.txt",
            "texts/b2.txt",
            score=0.99,
        ),
    ]
    manifest = tmp_path / "dataset_manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        data_dir=str(tmp_path),
        arabic_manifest=None,
        real_labels="high_match,medium_match",
        real_text_key="text_original_path",
        real_min_text_score=0.0,
        real_validate_paths=True,
        real_split="all",
        split_seed=42,
    )

    pairs = _load_arabic_dataset_pairs(args)

    assert len(pairs) == 1
    assert pairs[0].index == 1
    assert pairs[0].manifest_position == 1
    assert pairs[0].pair_id == "page_pair_1"
    assert pairs[0].label_type == "high_match"
    assert pairs[0].image1 == (lines / "a1.png").resolve()
    assert pairs[0].image2 == (lines / "b1.png").resolve()


def test_arabic_dataset_group_split_keeps_pair_ids_together(tmp_path):
    pairs = []
    for group in range(10):
        for member in range(2):
            pairs.append(
                ImagePair(
                    index=len(pairs) + 1,
                    image1=Path(tmp_path / f"a_{group}_{member}.png"),
                    image2=Path(tmp_path / f"b_{group}_{member}.png"),
                    pair_id=f"pair_{group}",
                    label_type="high_match",
                )
            )

    train, valid, test = _group_split_pairs(pairs, seed=42)
    train_ids = {pair.pair_id for pair in train}
    valid_ids = {pair.pair_id for pair in valid}
    test_ids = {pair.pair_id for pair in test}

    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert len(train) + len(valid) + len(test) == len(pairs)
