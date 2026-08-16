"""Route expanded-real image-image training to the selected objective.

``absolute`` preserves the fixed-cosine ablation.
``ranking`` preserves the transcript-group relative ranking ablation.
``sequence_ranking`` trains directly on local-window sequence alignment.
``sequence_ranking_partial_overlap`` keeps that objective and changes only the
train mixture to include partial-overlap positives.
"""
from __future__ import annotations

import os

from extra_real_training_v2_absolute import _eligible_groups, _positive_pair_loss


def install(base) -> None:
    objective = os.environ.get("NO_SHARED_IMAGE_OBJECTIVE", "absolute").strip().lower()
    if objective in {
        "sequence_ranking_partial_overlap",
        "partial_overlap",
        "sequence_partial_overlap",
    }:
        from extra_real_training_partial_overlap import install as install_partial_overlap

        install_partial_overlap(base)
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
        "'sequence_ranking', or 'sequence_ranking_partial_overlap', "
        f"got {objective!r}."
    )
