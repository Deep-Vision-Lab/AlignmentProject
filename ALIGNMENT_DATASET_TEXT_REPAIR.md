# ArabicDataset text repair

This branch adds a conservative audit-and-repair workflow for incorrect Arabic line transcripts in:

```text
DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_*/{A,B}/text/
```

It targets problems such as:

- missing spaces between Arabic words;
- incorrect textual forms or OCR substitutions;
- disagreement between raw, final, and optional LLM-generated text;
- missing or stale `final/tashkeel` files;
- stale page-level `full_final_original.txt` and `full_final_tashkeel.txt` files.

The workflow does not alter page images, line images, or raw OCR files. It changes only accepted final-text repairs and keeps the existing manifest paths.

## Scripts

```text
scripts/dataset/repair_alignment_texts_from_surahs.py
scripts/dataset/repair_alignment_texts.py
```

Use `repair_alignment_texts_from_surahs.py` for this dataset. It automatically reads the Quran text already included under:

```text
DataSet/ArabicDataset/Surahs/surah_*.json
```

It generates the clean and tashkeel reference files itself, so external Quran reference files are not required.

## Safety behavior

The default mode is a dry run.

- No final dataset file is modified unless `--apply` is supplied.
- Low-confidence lines are skipped.
- Proposed files are written under a timestamped `preview/` tree.
- Existing files are backed up under `backup/` before an apply operation.
- Every line receives CSV and JSONL audit records.
- Images and `text/raw/` files are never modified.
- The manifest does not need to be rewritten because repaired files keep their paths.

## 1. Switch to the repair branch

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject_clone

git fetch origin
git checkout dataset_text_repair
git pull origin dataset_text_repair

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manucripts_align
```

## 2. Confirm that the bundled Surah files exist

```bash
find DataSet/ArabicDataset/Surahs -maxdepth 1 -name 'surah_*.json' | head
```

## 3. Recommended dry run

Audit the complete dataset without changing final text files:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset
```

The wrapper creates derived references under:

```text
DataSet/ArabicDataset/text_repair_references/
```

The repair audit is written under:

```text
DataSet/ArabicDataset/text_repair_reports/<timestamp>/
```

Inspect:

```text
AUDIT.md
lines.csv
lines.jsonl
summary.json
preview/
```

## 4. Apply the complete repair

After checking the dry-run report:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --apply
```

Existing final text files are backed up before replacement.

## 5. Safer one-pair test

Dry run:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --pair pair_000001
```

Apply only that page pair:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --pair pair_000001 \
  --apply
```

## 6. Small dataset-wide smoke test

Process only the first ten page pairs:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --max-pairs 10
```

Apply only those ten pairs:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --max-pairs 10 \
  --apply
```

## 7. Confidence threshold

The default minimum confidence is:

```text
0.78
```

Use a stricter value when the OCR or metadata is uncertain:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --min-confidence 0.85 \
  --apply
```

Low-confidence lines remain unchanged and appear in the audit.

## 8. Generate references only

This verifies that the bundled Surah JSON structure can be parsed without running the repair:

```bash
python scripts/dataset/repair_alignment_texts_from_surahs.py \
  --dataset-root DataSet/ArabicDataset \
  --references-only
```

Expected generated files:

```text
DataSet/ArabicDataset/text_repair_references/quran_from_surahs_clean.txt
DataSet/ArabicDataset/text_repair_references/quran_from_surahs_tashkeel.txt
```

When the Surah JSON contains only one Arabic text form, the wrapper removes diacritics to derive the clean version and retains the available form for the tashkeel reference.

## 9. Inspect the newest report

```bash
latest_report="$(find DataSet/ArabicDataset/text_repair_reports \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"

cat "$latest_report/AUDIT.md"
```

Open the detailed CSV:

```bash
column -s, -t < "$latest_report/lines.csv" | less -S
```

Important statuses:

| Status | Meaning |
|---|---|
| `proposed` | Dry run found an accepted change. |
| `already_correct` | The bundled Quran reference agrees with the final files. |
| `skipped_low_confidence` | The match is below the configured threshold. |
| `skipped_no_source` | No usable final, raw, or LLM transcript was found. |
| `skipped_no_reference` | Page metadata could not be connected to a Surah reference. |
| `applied` | The accepted repair was written with `--apply`. |

## 10. How matching works

For each page side, the workflow:

1. extracts trusted verse text from `Surahs/surah_*.json`;
2. creates deterministic clean and tashkeel reference files;
3. reads `page_meta.json` and `pair_meta.json` to infer the Surah and ayah range;
4. gathers the existing final text, raw OCR, and optional LLM candidates;
5. searches the Quran reference sequentially so line order is preserved;
6. scores character agreement, length agreement, and spaced-text agreement;
7. accepts only matches at or above `--min-confidence`;
8. rebuilds the line-level and page-level final text files.

Comparison normalization removes tashkeel and normalizes common Arabic letter variants. The repaired output itself comes from the bundled Surah reference.

## 11. Validate after applying

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

## 12. Restore from backup

Every apply run stores original files under:

```text
DataSet/ArabicDataset/text_repair_reports/<timestamp>/backup/
```

Example restoration:

```bash
cp \
  DataSet/ArabicDataset/text_repair_reports/<timestamp>/backup/DatasetPairs/page_pairs/pair_000001/A/text/final/original/line_01.txt \
  DataSet/ArabicDataset/DatasetPairs/page_pairs/pair_000001/A/text/final/original/line_01.txt
```

Keep the report directory until the repaired dataset has been inspected and validated.

## Optional external-reference mode

The lower-level script still supports explicitly supplied Quran reference files:

```bash
python scripts/dataset/repair_alignment_texts.py \
  --dataset-root DataSet/ArabicDataset \
  --reference-clean /path/to/quran-clean.txt \
  --reference-tashkeel /path/to/quran-tashkeel.txt \
  --apply
```

This mode is optional and is not required when `DataSet/ArabicDataset/Surahs/` is available.
