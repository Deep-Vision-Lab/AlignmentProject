import json
from pathlib import Path

from Evaluation.real_subword_box_json import load_json_annotations


def test_legacy_bbox_json_loader_supports_nested_line_keys(tmp_path, monkeypatch):
    side = tmp_path / "ArabicDataset" / "DatasetPairs" / "page_pairs" / "pair_000001" / "A"
    image = side / "linesImages" / "line_01.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-read-by-parser")

    bbox = side / "bbox.json"
    bbox.write_text(
        json.dumps(
            {
                "lines": {
                    "1": {
                        "subwords": {
                            "كتب": [110, 18, 170, 42],
                            "محمد": [30, 18, 100, 42],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REAL_BOX_JSON", str(side))
    monkeypatch.setenv("REAL_BOX_BBOX_FORMAT", "xyxy")

    annotations = load_json_annotations(image)

    assert annotations.status == "ok"
    assert Path(annotations.workbook) == bbox.resolve()
    assert [box.text for box in annotations.boxes] == ["كتب", "محمد"]
