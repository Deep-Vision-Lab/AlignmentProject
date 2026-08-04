#!/usr/bin/env python3
"""Preview generated three-font pairs with exact Arabic transcripts and roles."""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--text-wrap-width", type=int, default=55)
    return parser.parse_args()


def role_summary(segments: list[dict]) -> str:
    return " | ".join(
        f"{segment['role']}: {segment['text']} [{segment['font']}]"
        for segment in segments
    )


def main() -> None:
    args = parse_args()
    metadata_path = args.data_dir / "metadata.jsonl"
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No metadata records in {metadata_path}")
    chosen = random.Random(args.seed).sample(
        records,
        min(args.samples, len(records)),
    )

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(chosen) * 2,
        1,
        figsize=(18, max(4, len(chosen) * 3.2)),
        squeeze=False,
    )
    for offset, record in enumerate(chosen):
        index = int(record["sample_index"])
        for line_number in (1, 2):
            image = Image.open(
                args.data_dir / "images" / f"img{line_number}_{index}.png"
            )
            text = record[f"text{line_number}"]
            segments = record[f"line{line_number}_segments"]
            axis = axes[offset * 2 + line_number - 1, 0]
            axis.imshow(image)
            title = (
                f"sample={index} mode={record['mode']} line={line_number}\n"
                f"text: {text}\nroles: {role_summary(segments)}"
            )
            axis.set_title(
                "\n".join(textwrap.wrap(title, args.text_wrap_width)),
                fontsize=9,
            )
            axis.axis("off")
            print(
                f"sample={index} mode={record['mode']} line={line_number}\n"
                f"exact_text={text}\n"
                f"roles={role_summary(segments)}\n"
            )

    figure.tight_layout()
    output = args.output or args.data_dir / "three_font_exact_text_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
