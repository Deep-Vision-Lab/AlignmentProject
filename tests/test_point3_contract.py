"""Point-3 transcript binding, hard objective, and RTL coordinate contract."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from Evaluation import eval_img_align_nw_diagnostic as pair_loader
from Evaluation import eval_point3_hard_paths as hard_paths
from Evaluation import eval_point3_training_paths as training_paths
from Evaluation import point3_core
from Evaluation.sw_dataset import load_pair_manifest
from vlm_restoration_positive_dtw import _soft_dtw_cost_matrix


ROOT = Path(__file__).resolve().parents[1]
GRAY_WEIGHTS = ROOT / "Weights/res18_tinyvit_real_gray_full1024_sigreg_l020/model_latest.pth"


def _native_side(root, side, *, own=True):
    side_dir = root / "DatasetPairs/page_pairs/pair_000001" / side
    images = side_dir / "linesImages"
    texts = side_dir / "text/final/original"
    images.mkdir(parents=True)
    texts.mkdir(parents=True)
    image = images / "line_07.png"
    image.write_bytes(b"native-line-placeholder")
    own_text = texts / "line_07.txt"
    if own:
        own_text.write_text(f"النص {side}", encoding="utf-8")
    wrong = texts / "line_08.txt"
    wrong.write_text("wrong paired line", encoding="utf-8")
    return image, own_text, wrong


@pytest.mark.parametrize("side", ["A", "B"])
def test_native_own_line_wins_for_both_sides(tmp_path, side, capsys):
    image, own, wrong = _native_side(tmp_path, side)
    assert hard_paths._transcript_path_for_line(image, own) == own
    assert hard_paths._transcript_path_for_line(image, wrong) == own
    assert "DTW_TRANSCRIPT_MISMATCH" in capsys.readouterr().out


@pytest.mark.parametrize("side", ["A", "B"])
def test_missing_native_own_line_fails_for_both_sides(tmp_path, side):
    image, _own, wrong = _native_side(tmp_path, side, own=False)
    with pytest.raises(FileNotFoundError, match="No same-line transcript"):
        hard_paths._transcript_path_for_line(image, wrong)


@pytest.mark.parametrize("side", ["A", "B"])
def test_generic_explicit_different_stem_wins_for_both_sides(tmp_path, side):
    root = tmp_path / "generic" / side
    images = root / "images"
    images.mkdir(parents=True)
    image = images / "foo.png"
    image.write_bytes(b"generic-image-placeholder")
    exact = root / "bar.txt"
    exact.write_text("صحيح", encoding="utf-8")
    assert hard_paths._transcript_path_for_line(image, exact) == exact
    with pytest.raises(FileNotFoundError, match="Manifest transcript"):
        hard_paths._transcript_path_for_line(image, root / "missing.txt")


@pytest.mark.parametrize("suffix", ["jsonl", "csv"])
def test_flat_manifest_preserves_both_explicit_text_paths(tmp_path, suffix):
    for name in ("foo.png", "qux.png", "bar.txt", "baz.txt"):
        (tmp_path / name).write_bytes(b"test-placeholder")
    manifest = tmp_path / f"pairs.{suffix}"
    row = {"image1": "foo.png", "image2": "qux.png",
           "text1": "bar.txt", "text2": "baz.txt", "dataset_type": "synthetic"}
    if suffix == "jsonl":
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    else:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    diagnostic = pair_loader._generic_manifest_pairs(manifest)[0]
    sw = load_pair_manifest(manifest, tmp_path)[0]
    for pair in (diagnostic, sw):
        assert pair.text1 == tmp_path / "bar.txt"
        assert pair.text2 == tmp_path / "baz.txt"
        assert hard_paths._transcript_path_for_line(pair.image1, pair.text1) == pair.text1
        assert hard_paths._transcript_path_for_line(pair.image2, pair.text2) == pair.text2


def test_logical_to_physical_intervals_and_window_thumbnails():
    expected = {0: (62, 992, 1024), 1: (61, 976, 1008),
                2: (60, 960, 992), 60: (2, 32, 64),
                61: (1, 16, 48), 62: (0, 0, 32)}
    for logical, mapping in expected.items():
        actual = point3_core.sequence_to_physical_window(
            logical, 63, width=1024, window=32, stride=16, use_flip=True)
        assert actual == mapping
        assert training_paths._physical_window(
            logical, 63, width=1024, window=32, stride=16, use_flip=True) == mapping
    pixels = np.tile(np.arange(1024, dtype=np.uint8), (128, 1))
    crops = hard_paths._extract_window_images(
        Image.fromarray(pixels), [0, 1, 62], window_size=32, stride=16, use_flip=True)
    assert [np.asarray(crop)[0, 0] for crop in crops] == [992 % 256, 976 % 256, 0]


def test_path_csv_names_actual_physical_and_logical_indices(tmp_path):
    result = {
        "path": [(0, 0), (1, 0), (2, 1)],
        "logical_sequence_indices": np.asarray([0, 1, 62]),
        "physical_window_indices": np.asarray([62, 61, 0]),
        "canvas_window_intervals": [(992., 1024.), (976., 1008.), (0., 32.)],
        "letters": ["ا", "ب"],
        "costs": np.zeros((3, 2), dtype=np.float32),
    }
    output = tmp_path / "path.csv"
    hard_paths._write_letter_paths(output, [(1, result), (2, result)])
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    for role in ("1", "2"):
        side = [row for row in rows if row["side"] == role]
        assert [(int(row["logical_sequence_index"]), int(row["physical_window_index"]),
                 int(row["canvas_x0"]), int(row["canvas_x1"])) for row in side] == [
                     (0, 62, 992, 1024), (1, 61, 976, 1008), (62, 0, 0, 32)]


def test_training_wrapper_exports_same_logical_and_physical_coordinates(tmp_path):
    output = tmp_path / "training_path.csv"
    geometry = {"canvas_width": 1024, "offset_x": 0, "crop_width": 1024,
                "scale_x": 1.0, "crop_left": 0}
    training_paths._write_image_text_path(
        output, [(0, 0), (1, 0), (2, 1)], ["ا", "ب"],
        np.zeros((3, 2)), geometry=geometry, window_size=32, stride=16,
        use_flip=True, sequence_indices=[0, 1, 62], total_window_count=63)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row["logical_sequence_index"]), int(row["physical_window_index"]),
             int(float(row["canvas_x0"])), int(float(row["canvas_x1"]))) for row in rows] == [
                 (0, 62, 992, 1024), (1, 61, 976, 1008), (62, 0, 0, 32)]


def test_existing_toy_path_is_unchanged_and_wrappers_share_core():
    costs = np.asarray([[0.1, 2.0, 3.0], [0.2, 0.1, 2.0],
                        [1.5, 0.2, 0.1], [2.0, 1.0, 0.2]], dtype=np.float32)
    kwargs = dict(vertical_penalty=0.05, horizontal_penalty=0.30,
                  position_prior_weight=0.15, disable_horizontal_when_feasible=True)
    result = point3_core.hard_letter_path(costs, **kwargs)
    hard_path, effective = hard_paths._hard_letter_dtw_path(costs, **kwargs)
    other_path, other_dp = training_paths._hard_monotonic_path(
        training_paths._position_prior(costs, 0.15),
        vertical_penalty=0.05, horizontal_penalty=0.30,
        disable_horizontal_when_feasible=True)
    assert result.path == hard_path == other_path == [(0, 0), (1, 1), (2, 2), (3, 2)]
    np.testing.assert_allclose(effective, result.effective_costs, rtol=1e-6)
    np.testing.assert_allclose(other_dp, result.dp, rtol=0, atol=0)
    assert hard_paths.hard_letter_path is point3_core.hard_letter_path
    assert training_paths.hard_letter_path is point3_core.hard_letter_path
    assert training_paths._hard_monotonic_path is point3_core.hard_monotonic_path


def test_hard_objective_includes_vertical_and_horizontal_penalties():
    vertical = point3_core.hard_letter_path(
        np.zeros((3, 2)), vertical_penalty=0.7, horizontal_penalty=0.4,
        position_prior_weight=0.0, disable_horizontal_when_feasible=True)
    assert vertical.vertical_steps == 1 and vertical.horizontal_steps == 0
    assert vertical.mean_path_cell_cost == 0.0
    assert vertical.hard_objective_total == pytest.approx(0.7)
    assert vertical.hard_objective_normalized == pytest.approx(0.7 / 5)
    horizontal = point3_core.hard_letter_path(
        np.zeros((2, 3)), vertical_penalty=0.7, horizontal_penalty=0.4,
        position_prior_weight=0.0, disable_horizontal_when_feasible=False)
    assert horizontal.vertical_steps == 0 and horizontal.horizontal_steps == 1
    assert horizontal.mean_path_cell_cost == 0.0
    assert horizontal.hard_objective_total == pytest.approx(0.4)
    assert horizontal.hard_objective_normalized == pytest.approx(0.4 / 5)


@pytest.mark.parametrize("shape,disable", [((4, 3), True), ((2, 3), False)])
def test_near_zero_gamma_soft_dtw_matches_full_hard_objective(shape, disable):
    costs = np.asarray([[0.11 + 0.13 * i + 0.07 * j + 0.011 * i * j
                         for j in range(shape[1])] for i in range(shape[0])], dtype=np.float64)
    hard = point3_core.hard_letter_path(
        costs, vertical_penalty=0.17, horizontal_penalty=0.23,
        position_prior_weight=0.14, disable_horizontal_when_feasible=disable)
    soft = _soft_dtw_cost_matrix(
        torch.tensor(costs, dtype=torch.float64), gamma=1e-5,
        vertical_penalty=0.17, horizontal_penalty=0.23,
        position_prior_weight=0.14, disable_horizontal_when_feasible=disable)
    assert float(soft) == pytest.approx(hard.hard_objective_normalized, abs=2e-5)


def test_complete_generic_different_stem_evaluation_on_both_sides(tmp_path):
    if not GRAY_WEIGHTS.is_file():
        pytest.skip("Current local checkpoint unavailable")
    inputs = tmp_path / "generic"
    inputs.mkdir()
    for image_name, text_name in (("foo.png", "bar.txt"), ("qux.png", "baz.txt")):
        image = Image.new("L", (1024, 128), 255)
        ImageDraw.Draw(image).rectangle((120, 30, 850, 96), fill=50)
        image.save(inputs / image_name)
        (inputs / text_name).write_text("سلام", encoding="utf-8")
    manifest = inputs / "pairs.jsonl"
    manifest.write_text(json.dumps({"image1": "foo.png", "image2": "qux.png",
                                   "text1": "bar.txt", "text2": "baz.txt"}) + "\n", encoding="utf-8")
    pair = pair_loader._generic_manifest_pairs(manifest)[0]
    torch.set_num_threads(2)
    output = tmp_path / "results"
    rows = hard_paths.evaluate_architecture(
        "checkpoint", GRAY_WEIGHTS, [pair], output, "cpu", "training")
    assert len(rows) == 1
    assert Path(rows[0]["line1_transcript"]) == inputs / "bar.txt"
    assert Path(rows[0]["line2_transcript"]) == inputs / "baz.txt"
    for side in (1, 2):
        assert rows[0][f"line{side}_hard_objective_total"] >= 0
        assert rows[0][f"line{side}_hard_objective_normalized"] == pytest.approx(
            rows[0][f"line{side}_normalized_hard_path_cost"])
    with (output / "checkpoint/pair_00001/letter_dtw_path.csv").open(
            newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["side"] for row in csv_rows} == {"1", "2"}
    assert int(csv_rows[0]["logical_sequence_index"]) == 0
    assert int(csv_rows[0]["physical_window_index"]) == 62
