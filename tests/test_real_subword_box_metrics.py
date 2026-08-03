from openpyxl import Workbook
from PIL import Image, ImageDraw

from Evaluation.real_subword_box_metrics import (
    BoxAnnotations,
    SubwordBox,
    aggregate,
    lcs_pairs,
    line_metrics,
    load_annotations,
)


def test_box_mask_confusion_metrics(monkeypatch):
    monkeypatch.setenv("REAL_BOX_IN_MASK_RULE", "center")
    annotations = BoxAnnotations(
        (
            SubwordBox("كتب", 800, 10, 900, 40, 2),
            SubwordBox("محمد", 600, 10, 700, 40, 3),
            SubwordBox("اليوم", 400, 10, 500, 40, 4),
        ),
        "boxes.xlsx",
        "annotations",
        "ok",
        "",
    )
    metrics = line_metrics("line1", annotations, {0, 1}, (380.0, 720.0))
    assert metrics["line1_box_tp"] == 1
    assert metrics["line1_box_fp"] == 1
    assert metrics["line1_box_fn"] == 1
    assert metrics["line1_box_precision"] == 0.5
    assert metrics["line1_box_recall"] == 0.5
    assert metrics["line1_box_f1"] == 0.5


def test_shared_subwords_use_ordered_lcs():
    assert lcs_pairs(
        ["كتب", "محمد", "اليوم"],
        ["قال", "كتب", "محمد", "غدا"],
    ) == [(0, 1), (1, 2)]


def test_excel_parser_discovers_boxes_near_line(tmp_path):
    side = tmp_path / "pair_000001" / "A"
    image_dir = side / "linesImages"
    annotation_dir = side / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    image_path = image_dir / "line_01.png"
    image = Image.new("RGB", (200, 60), "white")
    ImageDraw.Draw(image).rectangle((25, 18, 175, 42), fill="black")
    image.save(image_path)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
    sheet.append(["subword", "x0", "y0", "x1", "y1"])
    sheet.append(["كتب", 110, 18, 170, 42])
    sheet.append(["محمد", 30, 18, 100, 42])
    workbook.save(annotation_dir / "subword_boxes.xlsx")

    annotations = load_annotations(image_path)
    assert annotations.status == "ok"
    assert [box.text for box in annotations.boxes] == ["كتب", "محمد"]


def test_summary_micro_counts():
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
