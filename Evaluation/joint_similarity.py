"""Combine local and contextual cosine evidence before a single alignment."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def joint_components(local1, local2, contextual1, contextual2, local_weight=0.5):
    weight = float(local_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("local_weight must be finite and between 0 and 1")
    for features in (local1, local2, contextual1, contextual2):
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("Expected finite [windows, dimensions] feature matrices")
    if local1.shape[0] != contextual1.shape[0] or local2.shape[0] != contextual2.shape[0]:
        raise ValueError("Local and contextual features must refer to the same windows")
    local = F.normalize(local1.float(), dim=-1) @ F.normalize(local2.float(), dim=-1).T
    contextual = F.normalize(contextual1.float(), dim=-1) @ F.normalize(contextual2.float(), dim=-1).T
    joint = weight * local + (1.0 - weight) * contextual
    return local, contextual, joint
