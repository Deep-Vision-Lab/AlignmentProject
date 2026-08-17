"""Route expanded-real image-image training to the selected objective.

``absolute`` preserves the fixed-cosine ablation.
``ranking`` preserves the transcript-group relative ranking ablation.
``sequence_ranking`` trains directly on local-window sequence alignment.
``joint_real`` is the clean Stage-1 -> original+online-augmented real curriculum.
``joint_partial_overlap`` adds train-only multi-island partial-overlap positives.
``synthetic_bridge`` trains on the offline real-conditioned bridge V2 corpus with
AraBERT frozen, the shared-space text projection trainable, alignment masks exposed,
and bridge text ranking restricted to the actual shared islands.
"""
from __future__ import annotations

import os

from extra_real_training_v2_absolute import _eligible_groups, _positive_pair_loss


def install(base) -> None:
    objective = os.environ.get("NO_SHARED_IMAGE_OBJECTIVE", "absolute").strip().lower()
    if objective in {"synthetic_bridge", "real_synthetic_bridge", "bridge"}:
        import extra_real_training as legacy
        import real_synthetic_bridge_training as bridge_training
        from bridge_frozen_text import install as install_bridge_text_policy
        from bridge_mask_runtime import install as install_bridge_masks
        from bridge_multi_island_runtime import install as install_multi_island_runtime

        # Bridge V2 positives deliberately contain unrelated distractor regions.
        # The old generic sequence-ranking objective assumes the entire positive
        # line should align and would therefore reward alignment through those
        # distractors. Disable it inside the runtime, independent of launcher env.
        # The bridge-specific direct ranking installed below still trains on the
        # exact 1-3 shared islands only.
        base.P.use_sequence_alignment_ranking = False
        base.P.sequence_ranking_weight = 0.0

        install_bridge_text_policy(base)
        install_bridge_masks(legacy)
        install_multi_island_runtime(bridge_training)
        bridge_training.install(base)
        return
    if objective in {"joint_partial_overlap", "partial_overlap", "multi_island"}:
        from partial_overlap_runtime_fix import install as install_partial_overlap

        install_partial_overlap(base)
        return
    if objective in {"joint", "joint_real", "joint_discrimination"}:
        from joint_real_training_v5 import install as install_joint_real

        install_joint_real(base)
        return
    if objective in {"sequence", "sequence_ranking", "sw_ranking"}:
        from extra_real_training_v4 import install as install_sequence_ranking

        install_sequence_ranking(base)
        return
    if objective == "ranking":
        from extra_real_training_v3 import install as install_ranking

        install_ranking(base)
        return
    if objective == "absolute":
        from extra_real_training_v2_absolute import install as install_absolute

        install_absolute(base)
        return
    raise ValueError(
        "NO_SHARED_IMAGE_OBJECTIVE must be 'absolute', 'ranking', "
        "'sequence_ranking', 'joint_real', 'joint_partial_overlap', or "
        f"'synthetic_bridge', got {objective!r}."
    )
