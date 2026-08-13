#!/usr/bin/env python3
"""Real-data NW diagnostic entry point; canonical evaluator remains unchanged."""
from Evaluation import eval_img_align_nw as _canonical
from Evaluation.real_subword_box_patch import install as _install_bbox

_runner = _canonical._implementation
_physical_resolve = _runner.resolve_score_mode


def _resolve_score_mode(score_mode, dataset_type):
    value = str(score_mode).strip().lower().replace("_", "-")
    if value == "auto" and str(dataset_type).strip().lower() == "real":
        return "mutual-z"
    return _physical_resolve(score_mode, dataset_type)


# Importing the canonical launcher above already installed its Arabic RTL
# physical-coordinate and real-region mapping. Only restore score semantics and
# add the same real bbox metrics used by the SW launcher.
_runner.resolve_score_mode = _resolve_score_mode
_runner._resolve_score_mode = _resolve_score_mode
_install_bbox(_runner)

globals().update(
    {name: getattr(_runner, name) for name in dir(_runner) if not name.startswith("__")}
)

if __name__ == "__main__":
    _runner.main()
