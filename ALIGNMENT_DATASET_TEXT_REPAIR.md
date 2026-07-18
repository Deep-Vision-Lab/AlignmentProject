# ArabicDataset text repair

This branch adds a conservative audit-and-repair utility for incorrect line transcripts in:

```text
DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_*/{A,B}/text/
```

The common problems it targets are:

- missing spaces between Arabic words;
- incorrect textual forms or OCR substitutions;
- disagreement between `raw`, `final/original`, and optional LLM-generated text;
- missing or stale `final/tashkeel` files;
- stale page-level `full_final_original.txt` and `full_final_tashkeel.txt` files.

The script does **not** alter page images, line images, or raw OCR. It repairs only the final text files, and only when a sequential match against trusted Quran references reaches the requested confidence threshold.

## Added script

```text
scripts/dataset/repair_alignment_texts.py
```

## Safety behavior

The default mode is a dry run.

- No final dataset file is modified unless `--apply` is supplied.
- Low-confidence lines are skipped.
- Proposed files are written under a timestamped `preview/` tree.
- Existing files are backed up under `backup/` before an apply operation.
- Every line receives a CSV and JSONL audit record.
- Images and `text/raw/` files are never modified.
- The manifest does not need to be rewritten because repaired files keep their existing paths.

## Reference file format

The recommended reference format is one verse per line:

```text
surah_number|ayah_number|verse text
```

Example:

```text
1|1|بسم الله الرحمن الرحيم
1|2|الحمد لله رب العالمين
```

The clean and tashkeel files must describe the same Quran verses. The parser also supports compatible JSON/JSONL verse data and equal-length one-verse-per-line text files.

## 1. Audit one page pair

Run this first:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt \
  --pair pair_000001
```

Default minimum confidence:

```text
0.78
```

The command prints the generated report directory, for example:

```text
DataSet/ArabicDataset/text_repair_reports/20260718_183000
```

## 2. Inspect the audit

Open:

```text
text_repair_reports/<timestamp>/AUDIT.md
text_repair_reports/<timestamp>/lines.csv
text_repair_reports/<timestamp>/lines.jsonl
text_repair_reports/<timestamp>/summary.json
```

Inspect proposed repaired files under:

```text
text_repair_reports/<timestamp>/preview/DatasetPairs/page_pairs/
```

Important statuses:

| Status | Meaning |
|---|---|
| `proposed` | Dry-run found an accepted change. |
| `already_correct` | Trusted reference agrees with the existing final files. |
| `skipped_low_confidence` | Best reference match is below the threshold. |
| `skipped_no_source` | No usable final/raw/LLM transcript was found. |
| `skipped_no_reference` | Page metadata could not be connected to the supplied Quran reference. |
| `applied` | The accepted change was written with `--apply`. |

## 3. Apply one verified page pair

After checking the preview:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt \
  --pair pair_000001 \
  --apply
```

Before each existing file is replaced, it is copied to:

```text
text_repair_reports/<timestamp>/backup/
```

## 4. Audit several selected pairs

Repeat `--pair`:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt \
  --pair pair_000001 \
  --pair pair_000017 \
  --pair pair_000203
```

## 5. Small dataset-wide smoke test

Process only the first ten page pairs:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt \
  --max-pairs 10
```

## 6. Full dataset dry run

Omit `--pair` and `--max-pairs`:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt
```

Do not add `--apply` until the audit and previews have been reviewed.

## 7. Full dataset apply

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-simple-clean.txt \
  --reference-tashkeel /path/to/quran-simple.txt \
  --apply
```

## Confidence threshold

Use a stricter threshold when metadata or OCR quality is uncertain:

```bash
--min-confidence 0.85
```

Use a lower threshold only for manual investigation, not an unattended full apply:

```bash
--min-confidence 0.70
```

Low-confidence lines remain unchanged and appear in the audit report.

## How matching works

For each page side, the script:

1. reads `page_meta.json` and `pair_meta.json` to infer the surah and ayah range;
2. loads trusted clean and tashkeel Quran verses;
3. gathers each line's existing `final/original`, raw OCR, and optional LLM/other transcript candidates;
4. searches the page reference sequentially so line order is preserved;
5. scores Arabic character agreement, length agreement, and spaced-text agreement;
6. accepts only matches at or above `--min-confidence`;
7. writes clean and tashkeel line previews;
8. rebuilds the page-level full text from accepted or preserved line text.

Matching normalization removes tashkeel and normalizes common Arabic letter variants only for comparison. The output itself always comes from the trusted reference file.

## Validate after applying

Check that all manifest paths still resolve:

```bash
DATASET_TYPE=real \
DATA_DIR=DataSet/ArabicDataset \
REAL_VALIDATE_PATHS=1 \
REAL_BINARIZE=1 \
BATCH_SIZE=2 \
python - <<'PY'
from AugmentedRealDataLoader import build_dataloaders

train_loader, valid_loader, test_loader = build_dataloaders(
    "DataSet/ArabicDataset"
)

batch = next(iter(train_loader))
print("images1:", tuple(batch["images1"].shape))
print("images2:", tuple(batch["images2"].shape))
print("text1:", batch["texts1"][0])
print("text2:", batch["texts2"][0])
print("train/valid/test:", len(train_loader), len(valid_loader), len(test_loader))
PY
```

The repaired content is picked up automatically through the existing manifest paths.

## Restore a file from backup

Copy its matching backup file back into the dataset tree. Example:

```bash
cp \
  DataSet/ArabicDataset/text_repair_reports/<timestamp>/backup/DatasetPairs/page_pairs/pair_000001/A/text/final/original/line_01.txt \
  DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000001/A/text/final/original/line_01.txt
```

Keep the report directory until the repaired dataset has been inspected and validated.
