#!/usr/bin/env python3
"""Dense/full-line policy wrapper for the resilient Bridge V3 builder."""
from __future__ import annotations

from scripts.data import build_real_conditioned_synthetic_bridge_v3 as core
from scripts.data import build_real_conditioned_synthetic_bridge_v3_resilient as resilient


def main() -> None:
    args = core.parse_args()
    args.min_line_fill_ratio = max(float(args.min_line_fill_ratio), 0.90)
    args.font_size = max(56, min(int(args.font_size), 64))
    args.min_font_size = max(42, int(args.min_font_size))
    args.max_font_size = min(64, int(args.max_font_size))
    if args.max_font_size < args.font_size:
        args.font_size = args.max_font_size
    if args.font_size < args.min_font_size:
        args.font_size = args.min_font_size
    if args.max_font_size < args.min_font_size:
        raise ValueError(
            f"Dense Bridge V3 needs max_font_size >= min_font_size; "
            f"got {args.max_font_size} < {args.min_font_size}"
        )
    resilient.build(args)


if __name__ == "__main__":
    main()
