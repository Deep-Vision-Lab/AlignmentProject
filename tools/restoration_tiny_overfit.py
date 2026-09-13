#!/usr/bin/env python3
"""Recommendation 12: overfit eight distinct windows before full training."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restoration_window_seq2seq import (
    WindowSequenceCNNEncoder,
    WindowSequenceStrokeDecoder,
)

WINDOW_COUNT = 8
WINDOW_SIZE = 32
STRIDE = 16
LINE_WIDTH = WINDOW_SIZE + (WINDOW_COUNT - 1) * STRIDE


def make_line(device):
    # Eight overlapping regions with deliberately different stroke positions.
    line = torch.ones(1, 3, 128, LINE_WIDTH, device=device)
    for index in range(WINDOW_COUNT):
        x = index * STRIDE + 4 + (index % 3)
        y = 12 + index * 12
        line[:, :, y : y + 5, x : min(x + 18, LINE_WIDTH)] = 0.08 + 0.03 * index
        dot_y = max(1, y - 9)
        dot_x = min(LINE_WIDTH - 3, x + 8)
        line[:, :, dot_y : dot_y + 3, dot_x : dot_x + 3] = 0.0
    return line


def main():
    torch.manual_seed(120)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = WindowSequenceCNNEncoder(
        input_height=128,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        embed_dim=32,
        base_channels=16,
    ).to(device)
    decoder = WindowSequenceStrokeDecoder(
        dim=32,
        output_height=128,
        output_width=WINDOW_SIZE,
        channels=64,
    ).to(device)
    line = make_line(device)
    target = encoder.extract_windows(line)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=3e-3,
        weight_decay=0.0,
    )

    losses = []
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        encoded = encoder(line).squeeze(2).transpose(1, 2).contiguous()
        restored = decoder(encoded)
        loss = F.l1_loss(restored, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoded = encoder(line).squeeze(2).transpose(1, 2).contiguous()
        restored = decoder(encoded)
        swapped_tokens = encoded.clone()
        swapped_tokens[:, 0], swapped_tokens[:, -1] = (
            encoded[:, -1].clone(),
            encoded[:, 0].clone(),
        )
        swapped = decoder(swapped_tokens)
        swap_delta = float((restored - swapped).abs().mean())
        output_diversity = float(restored.std(dim=1).mean())

    print(
        f"point12 tiny-overfit device={device} windows={WINDOW_COUNT} "
        f"initial_L1={losses[0]:.6f} final_L1={losses[-1]:.6f} "
        f"output_diversity={output_diversity:.6f} swap_delta={swap_delta:.6f}",
        flush=True,
    )
    if not losses[-1] < losses[0] * 0.90:
        raise SystemExit("FAIL: eight-window reconstruction did not overfit enough")
    if output_diversity <= 1e-4:
        raise SystemExit("FAIL: decoded windows collapsed to the same template")
    if swap_delta <= 1e-4:
        raise SystemExit("FAIL: swapping encoded features did not change outputs")
    print("PASS point 12: distinct windows overfit and decoder depends on their features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
