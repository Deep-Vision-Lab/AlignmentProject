# Three-font exact-text synthetic Arabic generator

`generateDataArabicThreeFonts.py` generates original and augmented Arabic line pairs at dataset-generation time. Each visible region is created from an explicit text segment before rendering, so the UTF-8 transcript exactly matches the image.

## Recommended full dataset

```bash
bash scripts/data/run_generate_three_font_synthetic.sh
```

The launcher uses:

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

Equivalent explicit command:

```bash
python generateDataArabicThreeFonts.py \
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
  --target-fill-ratio 0.94
```

To control the exact font order instead of auto-discovery:

```bash
python generateDataArabicThreeFonts.py \
  --font-dir Fonts \
  --fonts FONT_1.ttf FONT_2.ttf FONT_3.ttf \
  --font-count 3 \
  --samples-per-font 3000
```

## Fast validation run

```bash
SAMPLES_PER_FONT=10 \
OUTPUT_DIR=DataSet/Synthetic_Arabic_Three_Font_Preview \
bash scripts/data/run_generate_three_font_synthetic.sh --skip-matrices
```

## Why transcript matching is better

The older training-time augmentation cropped image fractions and estimated the corresponding text from proportional character spans. Arabic glyph widths, joining forms, and whitespace make that estimate unreliable.

This generator instead:

1. chooses complete logical Arabic phrases;
2. assigns every phrase a role: `shared`, `context`, or `injected`;
3. renders each phrase as a tightly cropped patch;
4. places patches in right-to-left logical order with a small explicit gap;
5. joins the same phrase strings to produce the saved transcript.

No image crop is converted back into guessed text.

## Output contract

- `images/img1_N.png`, `images/img2_N.png`: 1024×128 paired line images.
- `texts/text1_N.txt`, `texts/text2_N.txt`: exact logical Arabic transcripts.
- `masks/mask1_N.png`, `masks/mask2_N.png`: bounding masks for the shared segment.
- `matrices/scoreMatrix_N.npy`: Needleman-Wunsch score matrix over exact transcripts with spaces removed.
- `similarity_matrices/similarityMatrix_N.npy`: exact character-equality matrix.
- `metadata.jsonl`: mode, exact segment text, role, font, and pixel box for each visible region.
- `generation_summary.json`: selected fonts, mode counts, and generation configuration.

Use `--corpus path/to/arabic_lines.txt` to add project-specific phrases. Each non-empty line containing at least two words is added to the built-in phrase pools.
