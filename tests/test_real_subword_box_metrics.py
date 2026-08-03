from pathlib import Path
import json

from openpyxl import Workbook
from PIL import Image, ImageDraw

from Evaluation.real_subword_box_metrics import (
    BoxAnnotations,
    SubwordBox,
    _lcs_pairs,
    _line_metrics,
    aggregate,
)
# Importing the patch installs robust generic-sheet handling and the exact
# foreground-crop geometry used by real evaluation.
from Evaluation.real_subword_box_patch import load_line_annotations


def test_mask_membership_produces_box_precision_recall_and_f1(monkeypatch):
    monkeypatch.setenv("REAL_BOX_IN_MASK_RULE", "center")
    boxes = (
        SubwordBox("كتب", 800, 10, 900, 40, 2),
        SubwordBox("محمد", 600, 10, 700, 40, 3),
        SubwordBox("اليوم", 400, 10, 500, 40, 4),
    )
    annotations = BoxAnnotations(boxes, "boxes.xlsx", "annotations", "ok", "")

    metrics = _line_metrics(
        "line1",
        annotations,
        gt_indices={0, 1},
        interval=(380.0, 720.0),
        width=1024,
        height=128,
    )

    assert metrics["line1_box_tp"] == 1
    assert metrics["line1_box_fp"] == 1
    assert metrics["line1_box_fn"] == 1
    assert metrics["line1_box_tn"] == 0
    assert metrics["line1_box_precision"] == 0.5
    assert metrics["line1_box_recall"] == 0.5
    assert metrics["line1_box_f1"] == 0.5


def test_shared_subwords_are_matched_in_order():
    assert _lcs_pairs(
        ["كتب", "محمد", "اليوم"],
        ["قال", "كتب", "محمد", "غدا"],
    ) == [(0, 1), (1, 2)]


def _save_line(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (200, 60), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 18, 175, 42), fill="black")
    image.save(path)


def _save_boxes(path: Path, include_image_column: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
    if include_image_column:
        sheet.append(["image_name", "subword", "x0", "y0", "x1", "y1"])
        sheet.append(["line_01.png", "كتب", 110, 18, 170, 42])
        sheet.append(["line_01.png", "محمد", 30, 18, 100, 42])
    else:
        sheet.append(["subword", "x0", "y0", "x1", "y1"])
        sheet.append(["كتب", 110, 18, 170, 42])
        sheet.append(["محمد", 30, 18, 100, 42])
    workbook.save(path)


def _set_geometry(monkeypatch):
    monkeypatch.setenv("REAL_BOX_COORDINATE_SPACE", "original")
    monkeypatch.setenv("ZERO_SHOT_PREPROCESS", "1")
    monkeypatch.setenv("ZERO_SHOT_FOREGROUND_CROP", "1")
    monkeypatch.setenv("ZERO_SHOT_PRESERVE_ASPECT", "1")
    monkeypatch.setenv("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")


def test_excel_boxes_follow_real_crop_resize_and_pad_geometry(tmp_path, monkeypatch):
    side = tmp_path / "pair_000001" / "A"
    image_path = side / "linesImages" / "line_01.png"
    _save_line(image_path)
    _save_boxes(side / "annotations" / "subword_boxes.xlsx")
    _set_geometry(monkeypatch)

    annotations = load_line_annotations(image_path, 1024, 128)

    assert annotations.status == "ok"
    assert annotations.workbook.endswith("subword_boxes.xlsx")
    assert [box.text for box in annotations.boxes] == ["كتب", "محمد"]
    assert all(0 <= box.x0 < box.x1 <= 1024 for box in annotations.boxes)
    assert all(0 <= box.y0 < box.y1 <= 128 for box in annotations.boxes)


def test_workbook_can_live_elsewhere_under_arabic_dataset(tmp_path, monkeypatch):
    dataset_root = tmp_path / "ArabicDataset"
    side = dataset_root / "DatasetPairs" / "page_pairs" / "pair_000060" / "A"
    image_path = side / "linesImages" / "line_01.png"
    _save_line(image_path)
    (side / "page_meta.json").write_text(
        json.dumps({"writer_id": "writer_17", "page_id": "page_0042"}),
        encoding="utf-8",
    )

    correct = dataset_root / "Dataset" / "Quran" / "Labels" / "writer_17" / "page_0042.xlsx"
    _save_boxes(correct, include_image_column=True)
    _save_boxes(dataset_root / "unrelated" / "page_9999.xlsx", include_image_column=True)

    monkeypatch.setenv("REAL_BOX_ANNOTATIONS_ROOT", str(dataset_root))
    _set_geometry(monkeypatch)
    annotations = load_line_annotations(image_path, 1024, 128)

    assert annotations.status == "ok"
    assert Path(annotations.workbook) == correct.resolve()
    assert [box.text for box in annotations.boxes] == ["كتب", "محمد"]


def test_batch_summary_reports_micro_and_macro_box_metrics():
    rows = [
        {
            "real_box_evaluated": True,
            "real_box_status": "ok",
            "pair_box_tp": 3,
            "pair_box_fp": 1,
            "pair_box_fn": 1,
            "pair_box_tn": 5,
            "pair_box_precision": 0.75,
            "pair_box_recall": 0.75,
            "pair_box_f1": 0.75,
            "mean_box_interval_iou": 0.8,
            "mean_box_pixel_iou": 0.7,
            "shared_subword_matches": 2,
        },
        {
            "real_box_evaluated": True,
            "real_box_status": "ok",
            "pair_box_tp": 1,
            "pair_box_fp": 0,
            "pair_box_fn": 1,
            "pair_box_tn": 2,
            "pair_box_precision": 1.0,
            "pair_box_recall": 0.5,
            "pair_box_f1": 2 / 3,
            "mean_box_interval_iou": 0.6,
            "mean_box_pixel_iou": 0.5,
            "shared_subword_matches": 1,
        },
    ]

    summary = aggregate(rows)

    assert summary["box_micro_tp"] == 4
    assert summary["box_micro_fp"] == 1
    assert summary["box_micro_fn"] == 2
    assert summary["box_micro_precision"] == 0.8
    assert abs(summary["box_micro_recall"] - (4 / 6)) < 1e-9
    assert summary["mean_box_interval_iou"] == 0.7
