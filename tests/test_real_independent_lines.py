import json
from pathlib import Path

from PIL import Image

from RealDataSet import (
    ArabicAllPageLinesDataset,
    ArabicManifestIndependentLineDataset,
)


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


def _write_page_side(root: Path, pair_name: str, side: str, page_pixels, lines):
    side_dir = root / "DatasetPairs" / "page_pairs" / pair_name / side
    lines_dir = side_dir / "linesImages"
    text_dir = side_dir / "text" / "final" / "original"
    lines_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (80, 120), page_pixels).save(side_dir / "original_image.png")
    for line_no, text, shade in lines:
        Image.new("RGB", (160, 40), (shade, shade, shade)).save(
            lines_dir / f"line_{line_no:02d}.png"
        )
        (text_dir / f"line_{line_no:02d}.txt").write_text(text, encoding="utf-8")


def test_all_page_lines_includes_lines_without_pair_manifest_entries(
    tmp_path: Path, monkeypatch
):
    # This test verifies population/inclusion only. Keep it independent of
    # launcher preprocessing env; XML crop is covered by the dedicated
    # real-all-page grayscale pipeline integration test.
    monkeypatch.setenv("REAL_BBOX_CROP", "0")
    # Page A has two lines, but pretend only line 1 would have appeared in a
    # line-pair manifest. The direct page scan must still retain line 2.
    _write_page_side(
        tmp_path,
        "pair_000001",
        "A",
        (210, 205, 195),
        [(1, "السطر الأول", 200), (2, "السطر غير المحاذى", 180)],
    )
    _write_page_side(
        tmp_path,
        "pair_000001",
        "B",
        (225, 220, 210),
        [(1, "سطر آخر", 170)],
    )

    dataset = ArabicAllPageLinesDataset(tmp_path, transform=None)

    assert len(dataset) == 3
    texts = {dataset[index][0].strip() for index in range(len(dataset))}
    assert "السطر غير المحاذى" in texts
    assert texts == {"السطر الأول", "السطر غير المحاذى", "سطر آخر"}
    assert dataset.scan_stats["unique_lines"] == 3
    assert dataset.scan_stats["unique_pages"] == 2


def test_all_page_lines_deduplicates_same_page_copied_into_multiple_pairs(
    tmp_path: Path, monkeypatch
):
    # This test verifies source-page deduplication only, not preprocessing.
    monkeypatch.setenv("REAL_BBOX_CROP", "0")
    # Exact copies of one source page appear in two different candidate pairs.
    page_lines = [(1, "واحد", 190), (2, "اثنان", 175)]
    _write_page_side(
        tmp_path,
        "pair_000001",
        "A",
        (200, 198, 190),
        page_lines,
    )
    _write_page_side(
        tmp_path,
        "pair_000002",
        "B",
        (200, 198, 190),
        page_lines,
    )

    dataset = ArabicAllPageLinesDataset(tmp_path, transform=None)

    assert len(dataset) == 2
    assert dataset.scan_stats["unique_pages"] == 1
    assert dataset.scan_stats["duplicate_page_line_copies_removed"] == 2
    assert len({sample["pair_id"] for sample in dataset.samples}) == 1
