"""Route expanded-real image-image training to the selected objective.

``absolute`` preserves the fixed-cosine ablation.
``ranking`` preserves the transcript-group relative ranking ablation.
``sequence_ranking`` trains directly on local-window sequence alignment.
``joint_real`` is the clean Stage-1 -> original+online-augmented real curriculum.
``joint_partial_overlap`` adds train-only multi-island partial-overlap positives.
"""
from __future__ import annotations

import os

from extra_real_training_v2_absolute import _eligible_groups, _positive_pair_loss


def install(base) -> None:
    objective = os.environ.get("NO_SHARED_IMAGE_OBJECTIVE", "absolute").strip().lower()
    if objective in {
        "joint_partial_overlap",
        "partial_overlap",
        "multi_island",
    }:
        from joint_real_training_partial_overlap import install as install_partial_overlap

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
        "'sequence_ranking', 'joint_real', or 'joint_partial_overlap', "
        f"got {objective!r}."
    )
