#!/usr/bin/env python3
"""Evaluate true local/context/fused Point-2 representations with Yelda alignment.

This is a thin wrapper around Evaluation.eval_yelda. It patches only model
reconstruction and image feature extraction, while reusing exactly the same
pair selection, image preprocessing, Needleman-Wunsch alignment, trace support,
and mask metrics as the existing evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from Evaluation.point2_runtime import (
    POINT2_MODES,
    load_point2_visual_models,
    point2_pair_features,
)


def _arg_value(argv: list[str], name: str):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--point2-representation", required=True, choices=POINT2_MODES)
    known, remaining = probe.parse_known_args(argv)
    mode = known.point2_representation

    # eval_yelda already has a fully tested local-vs-contextual selection API.
    # For local, preserve its local path. For all other Point-2 modes, inject the
    # requested vector into ImageFeatures.contextual and use its primary path.
    downstream_representation = "local" if mode == "local" else "primary"
    downstream = list(remaining) + ["--representation", downstream_representation]

    import Evaluation.yelda_runtime as runtime

    runtime.load_visual_models = load_point2_visual_models
    runtime.pair_features = (
        lambda models, image1, image2, _representation="primary":
        point2_pair_features(models, image1, image2, mode)
    )

    os.environ["POINT2_EVAL_REPRESENTATION"] = mode

    from Evaluation import eval_yelda

    eval_yelda.main(downstream)

    output_dir = _arg_value(remaining, "--output-dir")
    if output_dir:
        run_path = Path(output_dir).expanduser().resolve() / "run.json"
        if run_path.is_file():
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            payload["point2_representation"] = mode
            payload["point2_feature_definition"] = {
                "local": "normalized ResNet local vector L_t",
                "context": "normalized raw ViT contextual vector C_t",
                "fused": "trained normalized fusion F(L_t,C_t)",
                "fused_wrong_context": (
                    "trained fusion F(L_t,C_pi(t)); valid context positions are "
                    "deterministically permuted differently on the two paired lines "
                    "to destroy contextual correspondence while preserving local vectors"
                ),
            }[mode]
            payload.setdefault("arguments", {})["point2_representation"] = mode
            run_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
