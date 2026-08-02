import json

from PIL import Image
from torchvision import transforms

import DataLoader as loader_module
import DataSet as dataset_module
from direct_subword_data import install_dataset_patch


def _write_sample(root, line, index, text):
    image_path = root / "images" / f"img{line}_{index}.png"
    text_path = root / "texts" / f"text{line}_{index}.txt"
    Image.new("RGB", (64, 32), "black").save(image_path)
    text_path.write_text(text, encoding="utf-8")
    return image_path


def test_direct_dataset_loads_both_lines_and_collates_sidecars(tmp_path, monkeypatch):
    (tmp_path / "images").mkdir()
    (tmp_path / "texts").mkdir()
    boxes = tmp_path / "subword_boxes"
    boxes.mkdir()
    image1 = _write_sample(tmp_path, 1, 1, "اب")
    image2 = _write_sample(tmp_path, 2, 1, "اب")
    payload = {
        "validation": {"valid": True, "errors": []},
        "subwords": [
            {"text": "اب", "x0": 0.0, "x1": 64.0, "logical_index": 0}
        ],
    }
    (boxes / "subwords1_1.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    (boxes / "subwords2_1.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setenv("DIRECT_SUBWORD_SUPERVISION", "1")
    monkeypatch.setenv("LOAD_PAIRED_LINES", "1")
    monkeypatch.setenv("DIRECT_SUBWORD_BOX_DIR", str(boxes))
    monkeypatch.setattr(dataset_module, "use_image_pair_contrastive", False)
    monkeypatch.setattr(dataset_module, "image_text_loss_on_both_lines", False)
    install_dataset_patch()

    dataset = dataset_module.TextLineModern(
        new_dataset={
            "images": str(tmp_path / "images"),
            "texts": str(tmp_path / "texts"),
        },
        transform=transforms.ToTensor(),
        num_samples_override=1,
    )
    item = dataset[0]
    assert item["image1"].shape == item["image2"].shape
    assert item["subwords1"][0]["text"] == "اب"
    assert item["subwords2"][0]["text"] == "اب"

    batch = loader_module.custom_collate_fn([item])
    assert batch["images1"].shape[0] == 1
    assert batch["images2"].shape[0] == 1
    assert batch["subwords1"][0][0]["text"] == "اب"
    assert batch["subwords2"][0][0]["text"] == "اب"
    assert image1.name == "img1_1.png" and image2.name == "img2_1.png"
