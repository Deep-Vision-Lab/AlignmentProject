# ArabicDataset route and DataLoader guide

Dataset root:

```text
DataSet/ArabicDataset
```

This dataset contains Quran page images from two writer/book sources, extracted line images, cleaned line text, page metadata, and line-pair labels. It is useful for image-text alignment, retrieval, contrastive learning, and matching one handwritten Arabic line/page against another.

## Main routes

```text
DataSet/ArabicDataset/
├── dataset_manifest.jsonl
├── dataset_summary.json
├── Dataset/
│   └── Quran/
│       ├── images/<writer_id>/<page_id>.png
│       └── Labels/<writer_id>/<page_id>.xml
├── DatasetPairs/
│   ├── pairs_page_index.csv
│   ├── page_pairs/pair_000001/
│   │   ├── pair_meta.json
│   │   ├── A/
│   │   └── B/
│   └── line_pairs/pair_000001/
│       ├── pairs_lines_full.jsonl
│       └── stats.json
└── Surahs/
    └── surah_<number>.json
```

Use `dataset_manifest.jsonl` as the easiest training entry point. It is a sampled, unified list of line-pair samples from all `DatasetPairs/line_pairs/pair_*/pairs_lines_full.jsonl` files.

## What each file/folder contains

`dataset_manifest.jsonl`

Each line is one JSON sample. It points to one line from page side `A` and one line from page side `B`, plus matching scores and a label. Current summary:

```json
{
  "sampled_counts": {
    "high": 325,
    "medium": 607,
    "low": 18,
    "no_shared": 2875
  },
  "final_dataset_size": 3825
}
```

Important fields in each manifest row:

```text
pair_id                  Page-pair id, for example pair_000001.
surah_number/name        Quran surah metadata for the pair.
A.line_image_path        Image path for line A.
A.text_original_path     Cleaned line A text without tashkeel.
A.text_tashkeel_path     Cleaned line A text with tashkeel.
A.line_idx               1-based line number inside the page.
A.ayahs                  Ayah ids covered by that line.
B.*                      Same fields for line B.
scores.text_score        Text overlap/match score.
scores.avg_sim           Average text similarity for matched tokens.
scores.matched_tokens    Number of matched tokens.
scores.coverage_A/B      Token coverage per side.
scores.ayah_overlap      Number of shared ayahs.
label_type               high_match, medium_match, low_match, or no_shared_content.
```

`dataset_summary.json`

Count summary for the sampled manifest. Use this to check class balance.

`DatasetPairs/page_pairs/pair_*/`

Page-level paired data. Each `pair_xxxxxx` contains `A/` and `B/`, which are two pages selected as related candidates.

Inside each page side:

```text
original_image.png                  Full page image.
linesImages/line_01.png             Cropped line image.
text/raw/line_01.txt                Raw line OCR/text.
text/final/original/line_01.txt     Cleaned text without tashkeel.
text/final/tashkeel/line_01.txt     Cleaned text with tashkeel.
text/full_final_original.txt        Full cleaned page text without tashkeel.
text/full_final_tashkeel.txt        Full cleaned page text with tashkeel.
page_meta.json                      Page metadata, ayah range, line-to-ayah mapping.
debug/*.json                        Alignment/debug artifacts.
viz/page_lines.jpg                  Visualization of page line segmentation.
```

`DatasetPairs/page_pairs/pair_*/pair_meta.json`

Metadata for the page pair: writer ids, page ids, ayah ranges, overlap range, confidence/similarity metrics, and pair score.

`DatasetPairs/pairs_page_index.csv`

CSV index of all page pairs. Useful for quickly filtering by `surah_number`, `pair_score`, `overlap_len`, writer id, or page id.

`DatasetPairs/line_pairs/pair_*/pairs_lines_full.jsonl`

Expanded line-pair samples for one page pair. This is the source used to build the global `dataset_manifest.jsonl`.

`DatasetPairs/line_pairs/summary.json`

Global line-pair generation summary and label counts before manifest sampling.

`Dataset/Quran/images/<writer_id>/<page_id>.png`

Original full Quran page images grouped by writer/source id.

`Dataset/Quran/Labels/<writer_id>/<page_id>.xml`

Original XML labels for the full pages.

`Surahs/surah_<number>.json`

Reference Quran text by surah. Each file includes surah name, verse text, verse count, and juz metadata.

## Use the manifest in a PyTorch DataLoader

This is the recommended loader when training on paired line samples.

```python
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ArabicLinePairDataset(Dataset):
    def __init__(self, manifest_path, transform=None, text_key="text_original_path"):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.transform = transform
        self.text_key = text_key

        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def _resolve(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def _read_text(self, path):
        with self._resolve(path).open("r", encoding="utf-8") as f:
            return f.read().strip()

    def _read_image(self, path):
        image = Image.open(self._resolve(path)).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        a = sample["A"]
        b = sample["B"]

        return {
            "image_a": self._read_image(a["line_image_path"]),
            "image_b": self._read_image(b["line_image_path"]),
            "text_a": self._read_text(a[self.text_key]),
            "text_b": self._read_text(b[self.text_key]),
            "label": sample["label_type"],
            "text_score": torch.tensor(sample["scores"]["text_score"], dtype=torch.float32),
            "avg_sim": torch.tensor(sample["scores"]["avg_sim"], dtype=torch.float32),
            "pair_id": sample["pair_id"],
            "line_a": a["line_idx"],
            "line_b": b["line_idx"],
        }
```

Example usage:

```python
transform = transforms.Compose([
    transforms.Resize((128, 1024)),
    transforms.ToTensor(),
])

dataset = ArabicLinePairDataset(
    "DataSet/ArabicDataset/dataset_manifest.jsonl",
    transform=transform,
    text_key="text_original_path",  # use "text_tashkeel_path" if you want diacritics
)

train_size = int(0.6 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_ds, valid_ds, test_ds = random_split(
    dataset,
    [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

batch = next(iter(train_loader))
images_a = batch["image_a"]      # [B, 3, 128, 1024]
images_b = batch["image_b"]      # [B, 3, 128, 1024]
texts_a = batch["text_a"]        # list[str]
texts_b = batch["text_b"]        # list[str]
labels = batch["label"]          # list[str]
```

## Label use

For classification, map `label_type` to integers:

```python
LABEL_TO_ID = {
    "no_shared_content": 0,
    "low_match": 1,
    "medium_match": 2,
    "high_match": 3,
}
```

For contrastive or retrieval training:

```text
high_match / medium_match  -> positive or semi-positive pairs
low_match                  -> weak positive or hard negative, depending on experiment
no_shared_content          -> negative pairs
```

For regression/alignment training, use `scores.text_score`, `scores.avg_sim`, `scores.coverage_A`, `scores.coverage_B`, or `scores.ayah_overlap` as soft targets.

## Use one side as image-text samples

If your model expects one image and one text, not two paired lines, read side `A` or side `B` only:

```python
class ArabicSingleLineDataset(ArabicLinePairDataset):
    def __init__(self, manifest_path, side="A", transform=None, text_key="text_original_path"):
        super().__init__(manifest_path, transform=transform, text_key=text_key)
        self.side = side

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = sample[self.side]
        return self._read_text(item[self.text_key]), self._read_image(item["line_image_path"])
```

This returns the same `(text, image)` shape used by `TextLineModern` in `newDataSet.py`, but it reads from `ArabicDataset` instead of the flat synthetic dataset layout.

## Difference from the existing synthetic DataLoader

The existing loader in `newDataLoader.py` expects this flat structure:

```text
DataSet/Synthetic_Arabic_100000/
├── images/img1_1.png
├── images/img1_2.png
└── texts/text1_1.txt
```

`ArabicDataset` is different. It is organized around Quran page pairs and line pairs:

```text
DataSet/ArabicDataset/dataset_manifest.jsonl
DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000001/A/linesImages/line_01.png
DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000001/A/text/final/original/line_01.txt
```

So use a manifest-based Dataset like `ArabicLinePairDataset` above, or convert/symlink the data into the flat `images/` and `texts/` layout if you need to reuse `newDataLoader.py` unchanged.

## Rebuild notes

The included scripts are:

```text
build_page_pairs_dataset.py   Builds page-level A/B pairs and page index files.
build_line_pairs_single_pair.py
                              Builds line-pair JSONL files for each page pair.
dataset_factory.py            Samples line pairs into dataset_manifest.jsonl.
build_new_quran_dataset.py    Builds the processed Quran page/line dataset.
```

Typical manifest rebuild route:

```bash
python DataSet/ArabicDataset/dataset_factory.py \
  --dataset_root DataSet/ArabicDataset/DatasetPairs/line_pairs \
  --output_file DataSet/ArabicDataset/dataset_manifest.jsonl \
  --summary_file DataSet/ArabicDataset/dataset_summary.json
```

