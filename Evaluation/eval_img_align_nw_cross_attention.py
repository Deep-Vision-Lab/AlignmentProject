#!/usr/bin/env python3
"""Run the standard NW diagnostic on the FINAL cross-aware cosine features.

This intentionally reuses ``eval_img_align_nw_diagnostic.py`` so metrics,
visualizations, mask IoU/Dice, thresholds, and output layout remain directly
comparable with the no-cross-attention hierarchy.

The only semantic change is pair feature extraction:

    independent C1, C2
        -> bidirectional cross attention
        -> fused F1, F2
        -> cosine(F1, F2)
        -> the existing NW diagnostic

Example:
    python Evaluation/eval_img_align_nw_cross_attention.py \
      --dataset DataSet/Synthetic63 \
      --weights Weights/vit_vlm_cross/model_best.pth \
      --feature contextual \
      --n-samples 100
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from Evaluation import eval_img_align_nw_diagnostic as base
from Evaluation._eval_utils import get_image_features as _independent_get_image_features


class _PairFeatureState:
    def __init__(self):
        self.pending = None
        self.printed = False

    def get(self, models, image_path, dataset_type="synthetic"):
        features = _independent_get_image_features(models, image_path, dataset_type)
        enabled = bool(models.config.get("cross_attention_enabled", False))
        if not enabled:
            return features
        if models.text_model is None or not hasattr(models.text_model, "pair_cross_attention"):
            raise RuntimeError(
                "cross-attention checkpoint requires its text/pair module during evaluation"
            )

        if self.pending is None:
            self.pending = (models, features)
            return features

        first_models, first = self.pending
        self.pending = None
        if first_models is not models:
            raise RuntimeError("pair-aware evaluator received mismatched model instances")

        module = models.text_model.pair_cross_attention
        min_ink = float(models.config.get("cross_attention_min_ink", 0.01))
        with torch.no_grad():
            fused1, fused2, _attn12, _attn21 = module(
                first.contextual.unsqueeze(0),
                features.contextual.unsqueeze(0),
                ink1=first.ink.unsqueeze(0),
                ink2=features.ink.unsqueeze(0),
                min_ink=min_ink,
                return_weights=False,
            )
        first.contextual = F.normalize(fused1[0].float(), p=2, dim=-1)
        features.contextual = F.normalize(fused2[0].float(), p=2, dim=-1)

        if not self.printed:
            self.printed = True
            print(
                "cross-aware evaluation: separate C1/C2 -> bidirectional cross attention "
                f"-> cosine(F1,F2); gate={float(module.gate.detach().item()):.4f}",
                flush=True,
            )
        return features


_state = _PairFeatureState()
_original_load_models = base.load_evaluation_models
_original_parse_args = base.parse_args


def _load_models(weights_path, device="auto", load_text_model=False):
    # Pair cross-attention is checkpointed with the text-side runtime owner, so
    # force it to be reconstructed even though ordinary image-only NW does not
    # need a text model.
    return _original_load_models(weights_path, device, load_text_model=True)


def _parse_args():
    args = _original_parse_args()
    if args.feature != "contextual":
        raise SystemExit(
            "Cross-attention evaluation uses the fused contextual representation; "
            "run with --feature contextual."
        )
    return args


base.get_image_features = _state.get
base.load_evaluation_models = _load_models
base.parse_args = _parse_args


if __name__ == "__main__":
    base.main()
