"""Route expanded-real image-image training to the selected objective.

``absolute`` preserves the previous fixed-cosine ablation exactly.
``ranking`` selects the corrected relative positive-vs-no-shared objective.
"""
from __future__ import annotations

import os

from extra_real_training_v2_absolute import _eligible_groups, _positive_pair_loss


def install(base) -> None:
    objective = os.environ.get("NO_SHARED_IMAGE_OBJECTIVE", "absolute").strip().lower()
    if objective == "ranking":
        from extra_real_training_v3 import install as install_ranking

        install_ranking(base)
        return
    if objective == "absolute":
        from extra_real_training_v2_absolute import install as install_absolute

        install_absolute(base)
        return
    raise ValueError(
        "NO_SHARED_IMAGE_OBJECTIVE must be 'absolute' or 'ranking', "
        f"got {objective!r}."
    )
