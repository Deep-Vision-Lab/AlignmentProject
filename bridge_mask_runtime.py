"""Expose optional RealSyntheticBridge V2 alignment masks in training batches.

The generic manifest loader now reads ``B.alignment_mask_path`` when present. This
runtime patches only the expanded-real bridge collate function so ordinary real
training keeps its old batch contract and memory use.
"""
from __future__ import annotations

import torch


def install(legacy_module) -> None:
    if getattr(legacy_module, "_bridge_mask_collate_installed", False):
        return
    original_collate = legacy_module._collate_with_pair_mask

    def collate_with_bridge_mask(batch):
        output = original_collate(batch)
        if not isinstance(output, dict):
            return output
        if not any("alignment_mask2" in sample for sample in batch):
            return output

        images2 = output["images2"]
        height, width = int(images2.shape[-2]), int(images2.shape[-1])
        masks = []
        for sample in batch:
            mask = sample.get("alignment_mask2")
            if mask is None:
                mask = torch.zeros((1, height, width), dtype=torch.float32)
            else:
                mask = mask.float()
                if tuple(mask.shape[-2:]) != (height, width):
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0),
                        size=(height, width),
                        mode="nearest",
                    ).squeeze(0)
            masks.append(mask)

        output["alignment_masks2"] = torch.stack(masks, dim=0)
        output["bridge_shared_island_counts"] = [
            int(sample.get("bridge_shared_island_count", 0) or 0) for sample in batch
        ]
        output["bridge_shared_texts"] = [
            list(sample.get("bridge_shared_texts", [])) for sample in batch
        ]
        output["bridge_shared_boxes_px"] = [
            list(sample.get("bridge_shared_boxes_px", [])) for sample in batch
        ]
        return output

    legacy_module._collate_with_pair_mask = collate_with_bridge_mask
    legacy_module._bridge_mask_collate_installed = True
