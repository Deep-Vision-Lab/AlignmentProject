# Bridge V3 dataset layout

`anchor_id` is the relationship key. The same ID connects one copied real line, one positive synthetic sentence and mask, and all negative synthetic sentences.

```text
RealSyntheticBridge_v3/
├── images/
│   ├── real/<anchor_id>/real.*
│   ├── positive/<anchor_id>/positive.png
│   └── negative/<anchor_id>/negative_00.png ...
├── texts/
│   ├── real/<anchor_id>/real.txt
│   ├── positive/<anchor_id>/positive.txt
│   └── negative/<anchor_id>/negative_00.txt ...
├── masks/positive/<anchor_id>/positive_mask.png
├── anchors/<anchor_id>/anchor.json
├── positive/<anchor_id>/relation.json
├── negative/<anchor_id>/relations.json
├── anchor_index.jsonl
├── dataset_manifest.jsonl
└── metadata.json
```

For manual inspection, open `anchors/<anchor_id>/anchor.json`. It gives all paths for that real anchor group in one place.
