"""Balanced supported-region extraction for Needleman-Wunsch evaluation.

The global Needleman-Wunsch dynamic program and traceback are not changed.
This module only interprets which parts of that fixed path are sufficiently
supported to be rendered/evaluated as aligned image regions.

Compared with the strict discontinuous extractor, this version uses a small
amount of hysteresis: short noisy breaks are reconnected, while isolated or
non-distinctive high-similarity islands are removed.  No ground-truth mask is
used to decide which regions survive.
"""
from __future__ import annotations

import os

import numpy as np

from Evaluation import nw_core
from Evaluation import nw_discontinuous_regions as base


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _step_reward(step, matrix: np.ndarray, gap_penalty: float) -> float:
    if step.index1 is None or step.index2 is None:
        return float(gap_penalty)
    return float(matrix[int(step.index1), int(step.index2)])


def _mutual_z(matrix: np.ndarray, row: int, col: int) -> float:
    """How distinctive a cell is relative to both its row and its column."""
    value = float(matrix[row, col])
    row_values = np.asarray(matrix[row], dtype=np.float32)
    col_values = np.asarray(matrix[:, col], dtype=np.float32)

    row_std = max(float(np.std(row_values)), 1e-4)
    col_std = max(float(np.std(col_values)), 1e-4)
    row_z = (value - float(np.mean(row_values))) / row_std
    col_z = (value - float(np.mean(col_values))) / col_std
    return float(min(row_z, col_z))


def _gap_can_bridge(
    steps,
    previous_position: int,
    next_position: int,
    matrix: np.ndarray,
    gap_penalty: float,
    max_bridge_steps: int,
    bridge_mean_floor: float,
    bridge_hard_floor: float,
) -> bool:
    missing = int(next_position) - int(previous_position) - 1
    if missing <= 0:
        return True
    if missing > max_bridge_steps:
        return False

    rewards = [
        _step_reward(steps[position], matrix, gap_penalty)
        for position in range(previous_position + 1, next_position)
    ]
    if not rewards:
        return True
    return (
        float(np.mean(rewards)) >= bridge_mean_floor
        and float(np.min(rewards)) >= bridge_hard_floor
    )


def _make_record(group, steps, matrix: np.ndarray, gap_penalty: float):
    step_start = int(group[0])
    step_end = int(group[-1]) + 1
    pairs = [
        (int(steps[position].index1), int(steps[position].index2))
        for position in group
    ]
    match_values = [float(matrix[row, col]) for row, col in pairs]
    mutual_values = [_mutual_z(matrix, row, col) for row, col in pairs]
    full_rewards = [
        _step_reward(steps[position], matrix, gap_penalty)
        for position in range(step_start, step_end)
    ]
    return {
        "positions": list(group),
        "path": pairs,
        "step_start": step_start,
        "step_end": step_end,
        "traceback_steps": step_end - step_start,
        "score": float(np.sum(full_rewards)),
        "mean_match_score": float(np.mean(match_values)),
        "mean_mutual_z": float(np.mean(mutual_values)),
        "support_density": float(len(group) / max(1, step_end - step_start)),
    }


def _merge_records(left, right, steps, matrix: np.ndarray, gap_penalty: float):
    positions = sorted(set(left["positions"] + right["positions"]))
    return _make_record(positions, steps, matrix, gap_penalty)


def _supported_runs(steps, match_scores: np.ndarray, gap_penalty: float):
    """Return connected, distinctive, confidence-filtered runs on the NW path."""
    if not steps:
        return [], 0.0

    matrix = np.asarray(match_scores, dtype=np.float32)
    outer_start, outer_end, _ = nw_core._best_positive_segment(
        list(steps), matrix, float(gap_penalty)
    )
    if outer_end <= outer_start:
        return [], 0.0

    # A cell must first have positive diagonal evidence.  Connectivity and
    # confidence filtering below decide whether it becomes part of a mask.
    support_floor = _env_float("NW_REGION_SUPPORT_FLOOR", 0.0)
    max_bridge = max(0, _env_int("NW_REGION_MAX_BRIDGE_STEPS", 2))
    bridge_mean_floor = _env_float("NW_REGION_BRIDGE_MEAN_FLOOR", -0.12)
    bridge_hard_floor = _env_float("NW_REGION_BRIDGE_HARD_FLOOR", -0.35)
    force_connect_hole = max(0, _env_int("NW_REGION_FORCE_CONNECT_HOLE_STEPS", 1))

    min_matches = max(1, _env_int("NW_REGION_MIN_MATCH_STEPS", 3))
    min_mean_score = _env_float("NW_REGION_MIN_MEAN_SCORE", 0.12)
    min_mutual_z = _env_float("NW_REGION_MIN_MUTUAL_Z", 0.15)
    min_density = min(
        1.0, max(0.0, _env_float("NW_REGION_MIN_SUPPORT_DENSITY", 0.55))
    )
    min_run_score = _env_float("NW_REGION_MIN_RUN_SCORE", 1.50)
    min_relative_score = min(
        1.0, max(0.0, _env_float("NW_REGION_MIN_RELATIVE_RUN_SCORE", 0.15))
    )

    supported_positions = []
    for position in range(outer_start, outer_end):
        step = steps[position]
        if step.index1 is None or step.index2 is None:
            continue
        if float(matrix[int(step.index1), int(step.index2)]) > support_floor:
            supported_positions.append(position)

    if not supported_positions:
        return [], 0.0

    # First pass: reconnect small weak valleys only when the evidence in the
    # valley is not strongly contradictory.
    groups = []
    group = [supported_positions[0]]
    for position in supported_positions[1:]:
        if _gap_can_bridge(
            steps,
            group[-1],
            position,
            matrix,
            float(gap_penalty),
            max_bridge,
            bridge_mean_floor,
            bridge_hard_floor,
        ):
            group.append(position)
        else:
            groups.append(group)
            group = [position]
    groups.append(group)

    candidates = [
        _make_record(group, steps, matrix, float(gap_penalty))
        for group in groups
        if len(group) >= min_matches
    ]

    # A run must be sustained, reasonably strong, and distinctive in both the
    # row and column directions.  This removes isolated visually-plausible but
    # ambiguous cells that an early checkpoint often produces.
    candidates = [
        record
        for record in candidates
        if record["mean_match_score"] >= min_mean_score
        and record["mean_mutual_z"] >= min_mutual_z
        and record["support_density"] >= min_density
        and record["score"] >= min_run_score
    ]
    if not candidates:
        return [], 0.0

    strongest_score = max(float(record["score"]) for record in candidates)
    relative_floor = strongest_score * min_relative_score
    candidates = [
        record
        for record in candidates
        if float(record["score"]) >= max(min_run_score, relative_floor)
    ]
    if not candidates:
        return [], 0.0

    # Second pass: if two already-confident runs are separated by only one path
    # step, join them even when that single step was a sharp outlier.  A changed
    # word normally occupies several windows, so its longer valley remains a
    # real hole rather than being painted red.
    candidates.sort(key=lambda record: int(record["step_start"]))
    merged = [candidates[0]]
    for record in candidates[1:]:
        previous = merged[-1]
        hole = int(record["step_start"]) - int(previous["step_end"])
        if 0 <= hole <= force_connect_hole:
            merged[-1] = _merge_records(
                previous, record, steps, matrix, float(gap_penalty)
            )
        else:
            merged.append(record)

    total_score = float(sum(float(record["score"]) for record in merged))
    return merged, total_score


def install(runner) -> None:
    """Install balanced region interpretation; global NW DP stays untouched."""
    base._supported_runs = _supported_runs
    base.install(runner)
