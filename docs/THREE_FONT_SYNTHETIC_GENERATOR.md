# Three-font exact-text synthetic Arabic generator

The branch provides a generation-time augmentation pipeline using the three fonts in `Fonts/`. It builds every visible segment from known logical Arabic text before rendering, so the saved UTF-8 transcript matches the generated line.

The recommended entry point is:

```bash
bash scripts/data/run_generate_three_font_synthetic.sh
```

The launcher uses `generateDataArabicThreeFontsCompatible.py`, which preserves the three-font augmentation logic while matching the important settings from `generateDataArabic.py`.

## Settings inherited from `generateDataArabic.py`

```text
Image width: 1024
Image height: 128
Font size: 90
White text on black background
Paired img1/img2 filenames
Paired text1/text2 filenames
Paired mask1/mask2 filenames
Shared-phrase masks span the full 128-pixel image height
```

The mask is a vertical white band over the horizontal range occupied by the shared phrase. It is not limited to the glyph height.

## Recommended dataset mixture

```text
3 auto-discovered fonts from Fonts/
3,000 samples per font
9,000 total paired samples
25% original pairs
45% compact cross-injection pairs
30% aligned-plus-unaligned pairs
4-10 pixel gaps between merged regions
94% target canvas fill
```

## Full generation command

```bash
bash scripts/data/run_generate_three_font_synthetic.sh
```

Equivalent explicit command:

```bash
python generateDataArabicThreeFontsCompatible.py \
  --font-dir Fonts \
  --font-count 3 \
  --samples-per-font 3000 \
  --output-dir DataSet/Synthetic_Arabic_Three_Font_Augmented \
  --original-ratio 0.25 \
  --cross-injection-ratio 0.45 \
  --aligned-unaligned-ratio 0.30 \
  --mixed-font-injection-prob 0.30 \
  --segment-gap-min 4 \
  --segment-gap-max 10 \
  --target-fill-ratio 0.94 \
  --skip-matrices
```

`--skip-matrices` is accepted for compatibility, but the wrapper never creates matrix directories even when that flag is absent.

## Fast validation run

```bash
SAMPLES_PER_FONT=10 \
OUTPUT_DIR=DataSet/Synthetic_Arabic_Three_Font_Preview \
bash scripts/data/run_generate_three_font_synthetic.sh
```

Preview the images with the exact transcripts, segment roles, and fonts:

```bash
python scripts/data/preview_three_font_synthetic_dataset.py \
  --data-dir DataSet/Synthetic_Arabic_Three_Font_Preview \
  --samples 10 \
  --show
```

## Generated outputs

Only these dataset directories are created:

```text
images/
masks/
texts/
```

Additional audit files:

```text
metadata.jsonl
generation_summary.json
```

The following are deliberately not created:

```text
matrices/
similarity_matrices/
subword_boxes/
```

The metadata retains exact segment text, role, and font, but does not expose or save subword boxes.

## Output filenames

```text
images/img1_N.png
images/img2_N.png
masks/mask1_N.png
masks/mask2_N.png
texts/text1_N.txt
texts/text2_N.txt
```

Use `--corpus path/to/arabic_lines.txt` to add project-specific Arabic phrases. Each non-empty line containing at least two words is added to the built-in phrase pools.
