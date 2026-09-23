"""Shared hard interpretation of the training letter-DTW cost surface.

The differentiable Soft-DTW recurrence lives in training code. This module
only constructs the matching effective cells and reports its hard argmin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardLetterPath:
    path: list[tuple[int, int]]
    effective_costs: np.ndarray
    dp: np.ndarray
    mean_path_cell_cost: float
    hard_objective_total: float
    hard_objective_normalized: float
    vertical_steps: int
    horizontal_steps: int
    effective_horizontal_penalty: float


def effective_cell_costs(costs: np.ndarray, weight: float) -> np.ndarray:
    """Add the training position prior to [logical window, logical letter] costs."""
    result = np.asarray(costs, dtype=np.float64).copy()
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"Expected non-empty [windows, letters] costs, got {result.shape}")
    windows, letters = result.shape
    if weight > 0.0 and windows > 1 and letters > 1:
        wpos = np.linspace(0.0, 1.0, windows)[:, None]
        lpos = np.linspace(0.0, 1.0, letters)[None, :]
        result += float(weight) * np.abs(wpos - lpos)
    return result


def horizontal_transition_penalty(windows: int, letters: int, penalty: float,
                                  disable_when_feasible: bool) -> float:
    return 1e4 if disable_when_feasible and windows >= letters else float(penalty)


def hard_monotonic_path(cell_costs: np.ndarray, *, vertical_penalty: float,
                        horizontal_penalty: float,
                        disable_horizontal_when_feasible: bool):
    """One recurrence for both Point-3 wrappers; ties prefer diagonal, then vertical."""
    costs = np.asarray(cell_costs, dtype=np.float64)
    if costs.ndim != 2 or min(costs.shape) < 1:
        raise ValueError(f"Expected non-empty [windows, letters] costs, got {costs.shape}")
    windows, letters = costs.shape
    horizontal = horizontal_transition_penalty(
        windows, letters, horizontal_penalty, disable_horizontal_when_feasible)
    dp = np.full((windows, letters), np.inf, dtype=np.float64)
    back = np.full((windows, letters), -1, dtype=np.int8)
    dp[0, 0] = costs[0, 0]
    for i in range(windows):
        for j in range(letters):
            if i == 0 and j == 0:
                continue
            choices = []
            if i > 0 and j > 0:
                choices.append((dp[i - 1, j - 1], 0))
            if i > 0:
                choices.append((dp[i - 1, j] + float(vertical_penalty), 1))
            if j > 0:
                choices.append((dp[i, j - 1] + horizontal, 2))
            previous, direction = min(choices, key=lambda item: (item[0], item[1]))
            dp[i, j] = costs[i, j] + previous
            back[i, j] = direction
    i, j = windows - 1, letters - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        direction = int(back[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError(f"Invalid hard-DTW traceback at {(i, j)}")
        path.append((i, j))
    path.reverse()
    return path, dp


def hard_letter_path(costs: np.ndarray, *, vertical_penalty: float,
                     horizontal_penalty: float, position_prior_weight: float,
                     disable_horizontal_when_feasible: bool) -> HardLetterPath:
    effective = effective_cell_costs(costs, position_prior_weight)
    path, dp = hard_monotonic_path(
        effective,
        vertical_penalty=vertical_penalty,
        horizontal_penalty=horizontal_penalty,
        disable_horizontal_when_feasible=disable_horizontal_when_feasible,
    )
    vertical = sum(b[0] - a[0] == 1 and b[1] == a[1]
                   for a, b in zip(path, path[1:]))
    horizontal = sum(b[0] == a[0] and b[1] - a[1] == 1
                     for a, b in zip(path, path[1:]))
    horizontal_penalty_used = horizontal_transition_penalty(
        *effective.shape, horizontal_penalty, disable_horizontal_when_feasible)
    total = float(dp[-1, -1])
    return HardLetterPath(
        path=path, effective_costs=effective, dp=dp,
        mean_path_cell_cost=float(np.mean([effective[i, j] for i, j in path])),
        hard_objective_total=total,
        hard_objective_normalized=total / float(sum(effective.shape)),
        vertical_steps=vertical, horizontal_steps=horizontal,
        effective_horizontal_penalty=horizontal_penalty_used,
    )


def sequence_to_physical_window(sequence_index: int, count: int, *, width: int,
                                window: int, stride: int, use_flip: bool):
    """Return physical left-to-right index and its half-open canvas interval."""
    index = int(sequence_index)
    if not 0 <= index < int(count):
        raise IndexError(f"Logical sequence index {index} outside {count} windows")
    physical = int(count) - 1 - index if use_flip else index
    x0 = physical * int(stride)
    return physical, float(x0), float(min(int(width), x0 + int(window)))
