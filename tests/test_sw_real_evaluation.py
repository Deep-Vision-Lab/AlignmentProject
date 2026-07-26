import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from Evaluation.eval_img_align_sw import (
    ImagePair,
    _build_match_scores,
    _contiguous_runs,
    _display_image,
    _format_heatmap_value,
    _group_split_pairs,
    _heatmap_matrix,
    _load_arabic_dataset_pairs,
    _load_pair_manifest,
    _real_pair_paths,
    _resolve_score_mode,
    smith_waterman,
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


def test_sw_match_score_heatmap_uses_actual_diagonal_rewards():
    similarity = np.asarray([[0.80, 0.20], [0.55, 0.45]], dtype=np.float32)
    match_scores = similarity - 0.45
    dp_score = np.zeros((3, 3), dtype=np.float32)

    matrix, label = _heatmap_matrix(
        similarity,
        threshold=0.45,
        dp_score=dp_score,
        source="match-score",
        match_scores=match_scores,
        score_mode="raw",
    )

    np.testing.assert_allclose(
        matrix,
        np.asarray([[0.35, -0.25], [0.10, 0.00]], dtype=np.float32),
        atol=1e-6,
    )
    assert "raw score - threshold" in label
    assert _format_heatmap_value(matrix[0, 0], 2) == "0.35"


def test_sw_traceback_starts_at_interior_dp_maximum_not_terminal_cell():
    similarity = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.10, 0.95, 0.10, 0.10],
            [0.10, 0.10, 0.90, 0.10],
            [0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )

    path, score, dp_score, traceback = smith_waterman(
        similarity,
        threshold=0.40,
        gap_penalty=-0.20,
        return_traceback=True,
    )

    max_row, max_col = np.unravel_index(np.argmax(dp_score), dp_score.shape)
    assert score > 0.0
    assert path == [(1, 1), (2, 2)]
    assert (max_row, max_col) == (3, 3)
    np.testing.assert_array_equal(traceback[0], np.asarray([3.0, 3.0]))
    assert not np.array_equal(traceback[0], np.asarray([4.0, 4.0]))
    assert dp_score[int(traceback[-1, 1]), int(traceback[-1, 0])] == 0.0


def test_sw_complete_traceback_contains_gap_steps_in_backwards_direction():
    similarity = np.asarray(
        [
            [0.95, 0.10, 0.10],
            [0.10, 0.10, 0.95],
        ],
        dtype=np.float32,
    )

    path, score, _dp, traceback = smith_waterman(
        similarity,
        threshold=0.40,
        gap_penalty=-0.10,
        return_traceback=True,
    )

    assert score > 0.0
    assert path == [(0, 0), (1, 2)]
    assert len(traceback) - 1 == 3
    steps = np.diff(traceback, axis=0)
    assert any(np.array_equal(step, np.asarray([-1.0, 0.0])) for step in steps)


def test_real_auto_score_mode_is_mutual_z_and_suppresses_broad_positive_bias():
    similarity = np.full((8, 8), 0.70, dtype=np.float32)
    similarity[2:4, 4:6] = 0.90

    raw_scores = _build_match_scores(
        similarity, score_mode="raw", score_clip=4.0, threshold=0.45
    )
    mutual_z_scores = _build_match_scores(
        similarity, score_mode="mutual-z", score_clip=4.0, threshold=0.45
    )

    assert _resolve_score_mode("auto", "real") == "mutual-z"
    assert _resolve_score_mode("auto", "synthetic") == "raw"
    assert float((mutual_z_scores > 0).mean()) < float((raw_scores > 0).mean())


def test_matched_window_runs_do_not_fill_intervening_gaps():
    assert _contiguous_runs([1, 2, 5, 6, 9]) == [(1, 3), (5, 7), (9, 10)]
