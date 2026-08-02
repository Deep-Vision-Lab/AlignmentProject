"""Opt-in direct connected-subword supervision for synthetic Arabic lines."""
from __future__ import annotations

import os

import torch.nn as nn

from direct_subword_data import enabled, flag, install_dataset_patch, integer, number
from direct_subword_loss import make_batch_loss

_INSTALLED = False


def config() -> dict:
    return {
        "direct_subword_supervision": enabled(),
        "direct_subword_box_dir": os.environ.get("DIRECT_SUBWORD_BOX_DIR", ""),
        "direct_subword_temperature": number("DIRECT_SUBWORD_TEMPERATURE", 0.07),
        "direct_subword_bce_temperature": number(
            "DIRECT_SUBWORD_BCE_TEMPERATURE", 0.10
        ),
        "direct_subword_similarity_threshold": number(
            "DIRECT_SUBWORD_SIMILARITY_THRESHOLD", 0.20
        ),
        "direct_subword_focal_gamma": number("DIRECT_SUBWORD_FOCAL_GAMMA", 1.5),
        "direct_subword_positive_boost": number(
            "DIRECT_SUBWORD_POSITIVE_BOOST", 2.0
        ),
        "direct_subword_region_weight": number(
            "DIRECT_SUBWORD_REGION_WEIGHT", 1.0
        ),
        "direct_subword_context_region_weight": number(
            "DIRECT_SUBWORD_CONTEXT_REGION_WEIGHT", 0.15
        ),
        "direct_subword_localization_weight": number(
            "DIRECT_SUBWORD_LOCALIZATION_WEIGHT", 1.0
        ),
        "direct_subword_context_localization_weight": number(
            "DIRECT_SUBWORD_CONTEXT_LOCALIZATION_WEIGHT", 0.25
        ),
        "direct_subword_attention_weight": number(
            "DIRECT_SUBWORD_ATTENTION_WEIGHT", 0.10
        ),
        "direct_subword_outside_weight": number(
            "DIRECT_SUBWORD_OUTSIDE_WEIGHT", 0.25
        ),
        "direct_subword_outside_margin": number(
            "DIRECT_SUBWORD_OUTSIDE_MARGIN", 0.25
        ),
        "direct_subword_outside_top_k": integer(
            "DIRECT_SUBWORD_OUTSIDE_TOP_K", 8
        ),
        "direct_subword_use_ink_weighting": flag(
            "DIRECT_SUBWORD_USE_INK_WEIGHTING", True
        ),
        "direct_subword_ink_floor": number("DIRECT_SUBWORD_INK_FLOOR", 0.05),
        "direct_subword_outside_min_ink": number(
            "DIRECT_SUBWORD_OUTSIDE_MIN_INK", 0.01
        ),
        "direct_subword_strict_boxes": flag("DIRECT_SUBWORD_STRICT_BOXES", True),
        "direct_subword_objective": (
            "local_class_contrastive+context_class_contrastive+"
            "soft_interval_focal_bce+context_interval_bce+attention+outside_margin"
        ),
        "synthetic_alignment_backend": "renderer_subword_intervals_no_dtw",
    }


class DirectCriterion(nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Direct connected-subword mode bypasses Span-DTW")


def _install_wandb_metrics(train_module) -> None:
    def log(run, epoch, train_loss, validation_loss, stats):
        if run is None:
            return
        train_module.wandb.log(
            {
                "loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "direct/local_region_loss": float(
                    stats.get("direct_region_loss", 0.0)
                ),
                "direct/context_region_loss": float(
                    stats.get("direct_context_region_loss", 0.0)
                ),
                "direct/localization_bce": float(
                    stats.get("direct_localization_loss", 0.0)
                ),
                "direct/context_localization_bce": float(
                    stats.get("direct_context_localization_loss", 0.0)
                ),
                "direct/attention_loss": float(
                    stats.get("direct_attention_loss", 0.0)
                ),
                "direct/outside_loss": float(stats.get("direct_outside_loss", 0.0)),
                "direct/positive_similarity": float(
                    stats.get("direct_positive_similarity", 0.0)
                ),
                "direct/negative_similarity": float(
                    stats.get("direct_negative_similarity", 0.0)
                ),
                "direct/subword_regions": float(
                    stats.get("direct_subword_regions", 0.0)
                ),
                "direct/unique_subwords": float(
                    stats.get("direct_unique_subwords", 0.0)
                ),
                "direct/window_iou": float(stats.get("direct_window_iou", 0.0)),
                "direct/center_error_px": float(
                    stats.get("direct_center_error_px", 0.0)
                ),
                "direct/boundary_error_px": float(
                    stats.get("direct_boundary_error_px", 0.0)
                ),
                "direct/start_error_px": float(
                    stats.get("direct_start_error_px", 0.0)
                ),
                "direct/end_error_px": float(
                    stats.get("direct_end_error_px", 0.0)
                ),
                "gap": float(stats.get("gap", float("nan"))),
            },
            step=int(epoch),
            commit=True,
        )

    train_module.wandb_log_epoch_metrics = log


def install(train_module) -> dict:
    """Install no-DTW training only when DIRECT_SUBWORD_SUPERVISION is enabled."""
    global _INSTALLED
    resolved = config()
    if not enabled() or _INSTALLED:
        return resolved
    if os.environ.get("DATASET_TYPE", "synthetic").strip().lower() != "synthetic":
        raise RuntimeError(
            "DIRECT_SUBWORD_SUPERVISION requires DATASET_TYPE=synthetic. "
            "Real lines have no exact subword intervals; continue them with Span-DTW."
        )
    if flag("ZERO_SHOT_PROFILE", False):
        raise RuntimeError(
            "ZERO_SHOT_PROFILE applies geometric transforms that invalidate fixed "
            "subword intervals. Keep it disabled for direct supervision."
        )
    install_dataset_patch()
    train_module.compute_batch_loss = make_batch_loss(train_module)
    train_module.build_criterion = lambda: DirectCriterion()
    _install_wandb_metrics(train_module)
    _INSTALLED = True
    return resolved
