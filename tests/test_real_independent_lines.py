import json
from pathlib import Path

from PIL import Image

from RealDataSet import ArabicManifestIndependentLineDataset


def _write_line(root: Path, name: str, text: str):
    image = root / f"{name}.png"
    transcript = root / f"{name}.txt"
    Image.new("RGB", (120, 32), "white").save(image)
    transcript.write_text(text, encoding="utf-8")
    return str(image), str(transcript)


def test_independent_manifest_flattens_both_sides_and_deduplicates(tmp_path: Path):
    a_img, a_txt = _write_line(tmp_path, "a", "ألف")
    b_img, b_txt = _write_line(tmp_path, "b", "باء")
    c_img, c_txt = _write_line(tmp_path, "c", "جيم")

    rows = [
        {
            "pair_id": "pair_1",
            "label_type": "no_shared_content",
            "A": {
                "line_image_path": a_img,
                "text_original_path": a_txt,
                "line_idx": 1,
                "original_image": "page_A.png",
            },
            "B": {
                "line_image_path": b_img,
                "text_original_path": b_txt,
                "line_idx": 2,
                "original_image": "page_B.png",
            },
        },
        {
            "pair_id": "pair_1",
            "label_type": "high_match",
            # A is repeated across another pair row and must not be duplicated.
            "A": {
                "line_image_path": a_img,
                "text_original_path": a_txt,
                "line_idx": 1,
                "original_image": "page_A.png",
            },
            "B": {
                "line_image_path": c_img,
                "text_original_path": c_txt,
                "line_idx": 3,
                "original_image": "page_B.png",
            },
        },
    ]
    manifest = tmp_path / "dataset_manifest_full_pairs.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = ArabicManifestIndependentLineDataset(
        manifest,
        transform=None,
        text_key="text_original_path",
        validate_paths=True,
    )

    assert len(dataset) == 3
    assert {sample["line_idx"] for sample in dataset.samples} == {1, 2, 3}
    assert {sample["pair_id"] for sample in dataset.samples} == {
        "page_A.png",
        "page_B.png",
    }

    texts = [dataset[index][0].strip() for index in range(len(dataset))]
    assert set(texts) == {"ألف", "باء", "جيم"}


def test_independent_manifest_ignores_pair_labels(tmp_path: Path):
    a_img, a_txt = _write_line(tmp_path, "a", "سطر")
    b_img, b_txt = _write_line(tmp_path, "b", "آخر")
    manifest = tmp_path / "dataset_manifest_full_pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "pair_id": "pair_9",
                "label_type": "no_shared_content",
                "A": {
                    "line_image_path": a_img,
                    "text_original_path": a_txt,
                    "original_image": "p1.png",
                },
                "B": {
                    "line_image_path": b_img,
                    "text_original_path": b_txt,
                    "original_image": "p2.png",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ArabicManifestIndependentLineDataset(manifest)
    assert len(dataset) == 2
