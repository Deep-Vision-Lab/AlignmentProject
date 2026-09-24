"""Image-only partial matching and support; no transcripts or alignment labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from PIL import Image

from Evaluation.sw_core import smith_waterman_affine
from Evaluation.yelda_geometry import source_intervals
from Evaluation.point3_core import sequence_to_physical_window


@dataclass(frozen=True)
class RegionSettings:
    score_mode: str = "background"
    cosine_threshold: float = 0.60
    contrast_margin: float = 0.05
    gap_open: float = 0.20
    gap_extend: float = 0.05
    min_windows: int = 5
    max_internal_gap: int = 1
    min_region_score: float = 0.0
    max_candidates: int = 128

    def validate(self):
        if self.score_mode not in {"background", "raw"}:
            raise ValueError("score_mode must be background or raw")
        if not np.isfinite([self.cosine_threshold, self.contrast_margin, self.gap_open,
                            self.gap_extend, self.min_region_score]).all():
            raise ValueError("Scoring parameters must be finite")
        if not -1 <= self.cosine_threshold <= 1 or self.contrast_margin < 0:
            raise ValueError("Require cosine_threshold in [-1,1] and contrast_margin >= 0")
        if not self.gap_open >= self.gap_extend >= 0 or self.min_region_score < 0:
            raise ValueError("Require gap_open >= gap_extend >= 0 and min_region_score >= 0")
        if self.min_windows < 1 or self.max_internal_gap < 0 or self.max_candidates < 1:
            raise ValueError("Invalid support, gap, or candidate limit")


def match_rewards(cosine, settings):
    settings.validate()
    c = np.asarray(cosine)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("Cosine matrix must be finite and two-dimensional")
    # Never modify c; the caller saves the original matrix verbatim.
    if settings.score_mode == "raw" or not c.size:
        return c.astype(np.float64) - settings.cosine_threshold
    baseline = np.maximum(settings.cosine_threshold,
                          np.median(c, axis=1)[:, None] + settings.contrast_margin)
    baseline = np.maximum(baseline, np.median(c, axis=0)[None, :] + settings.contrast_margin)
    return c.astype(np.float64) - baseline


def affine_trace_score(steps, rewards, gap_open, gap_extend):
    """Scalar for a trimmed traceback, with each affine gap run charged once."""
    total, previous = 0., None
    for i, j in steps:
        if i is not None and j is not None:
            total += float(rewards[i, j])
            previous = None
        else:
            kind = "line1" if i is None else "line2"
            total -= gap_extend if previous == kind else gap_open
            previous = kind
    return total


def extract_regions(cosine, physical1, physical2, settings=RegionSettings()):
    """Greedy affine SW, then trim/split by positive physical-window anchors.

    Accepted spans exclude their rows/columns AND crossing quadrants from later
    matches. Rejected positive cells are suppressed too, so a short high-scoring
    peak cannot cause an infinite retry. This greedy suppression can miss a
    better joint solution; it is not globally optimal multi-region alignment.
    """
    rewards = match_rewards(cosine, settings)
    physical = [np.asarray(p, dtype=int) for p in (physical1, physical2)]
    for size, p in zip(rewards.shape, physical):
        if p.shape != (size,) or len(set(p.tolist())) != size or (p < 0).any():
            raise ValueError("Physical indices must be unique nonnegative valid-window identities")
        if size > 1 and not ((np.diff(p) > 0).all() or (np.diff(p) < 0).all()):
            raise ValueError("Physical indices must preserve model sequence order")
    populations = [set(p.tolist()) for p in physical]
    allowed = np.ones(rewards.shape, dtype=bool)
    regions, rejected = [], []
    rows, cols = np.indices(rewards.shape)
    attempts = 0
    while rewards.size and attempts < settings.max_candidates:
        candidate = smith_waterman_affine(np.where(allowed, rewards, -np.inf),
                                         settings.gap_open, settings.gap_extend)
        if candidate.score <= 0:
            break
        attempts += 1
        anchors = [(k, i, j) for k, (i, j) in enumerate(candidate.steps)
                   if i is not None and j is not None and rewards[i, j] > 0]
        if not anchors:
            raise RuntimeError("Positive local objective has no positive diagonal anchor")
        groups = []
        for anchor in anchors:
            if groups:
                previous = groups[-1][-1]
                adjacent = True
                for side, p in enumerate(physical, start=1):
                    a, b = sorted((int(p[previous[side]]), int(p[anchor[side]])))
                    adjacent &= b-a-1 <= settings.max_internal_gap
                    # A removed/invalid/padding position must never be gap-filled.
                    adjacent &= set(range(a, b+1)).issubset(populations[side-1])
                if adjacent:
                    groups[-1].append(anchor)
                    continue
            groups.append([anchor])
        for group in groups:
            pairs = [(i, j) for _, i, j in group]
            support = [len({int(p[pair[s]]) for pair in pairs}) for s, p in enumerate(physical)]
            steps = candidate.steps[group[0][0]:group[-1][0]+1]
            objective = affine_trace_score(steps, rewards, settings.gap_open, settings.gap_extend)
            reason = ("insufficient_distinct_positive_windows" if min(support) < settings.min_windows
                      else "trimmed_objective_below_minimum" if objective <= settings.min_region_score else None)
            if reason:
                rejected.append(dict(reason=reason, pairs=pairs, support=support, score=objective))
                continue
            spans = [(min(pair[s] for pair in pairs), max(pair[s] for pair in pairs)) for s in (0, 1)]
            supported = [sorted({int(p[pair[s]]) for pair in pairs}) for s, p in enumerate(physical)]
            filled = [sorted(set(range(min(ids), max(ids)+1)) - set(ids)) for ids in supported]
            regions.append(dict(pairs=pairs, steps=steps, support=support, score=objective,
                                supported_physical=supported, filled_physical=filled,
                                spans=spans, candidate_score=candidate.score))
            (a, b), (c, d) = spans
            allowed &= ((rows < a) & (cols < c)) | ((rows > b) & (cols > d))
        for _, i, j in anchors:
            allowed[i, j] = False
    regions.sort(key=lambda r: r["spans"][0][0])
    if not regions and not rejected:
        rejected.append(dict(reason="no_positive_local_alignment", support=[0, 0]))
    capped = attempts == settings.max_candidates and bool(np.any(allowed & (rewards > 0)))
    if capped:
        rejected.append(dict(reason="candidate_limit_reached"))
    return dict(regions=regions, rejected=rejected, rewards=rewards, attempts=attempts,
                candidate_limit_reached=capped, parameters=asdict(settings))


def valid_window_geometry(features, geometry, contract, use_flip):
    """Use encoder-owned physical IDs, or the shared fixed-grid RTL mapping."""
    valid = features.token_valid.detach().cpu().numpy().astype(bool)
    if valid.ndim != 1 or len(valid) != features.contextual.shape[0]:
        raise ValueError("Features and token-valid mask disagree")
    packed = features.physical_window_indices
    physical_ids = packed.detach().cpu().numpy() if packed is not None else None
    if physical_ids is not None and physical_ids.shape != valid.shape:
        raise ValueError("Physical-index shape disagrees with token-valid mask")
    grid_count = 1 + (int(geometry["canvas_width"]) - contract.window_size) // contract.stride
    if physical_ids is None and len(valid) != grid_count:
        raise ValueError("Packed window output requires explicit physical-window identities")
    entries = []
    for logical in np.flatnonzero(valid):
        index = int(physical_ids[logical]) if physical_ids is not None else int(logical)
        physical, x0, x1 = sequence_to_physical_window(
            index, grid_count, width=int(geometry["canvas_width"]), window=contract.window_size,
            stride=contract.stride, use_flip=use_flip if physical_ids is None else False)
        mapped = source_intervals([[x0, x1]], geometry)
        if mapped:
            entries.append(dict(logical_index=int(logical), physical_index=physical,
                                model_interval=[x0, x1], source_interval=mapped[0]))
    return entries


def intervals_mask(intervals, source_size):
    """Half-open source intervals rasterized outward; full source height."""
    width, height = map(int, source_size)
    if width <= 0 or height <= 0:
        raise ValueError("Source image dimensions must be positive")
    mask = np.zeros((height, width), dtype=np.uint8)
    for a, b in intervals:
        if not np.isfinite([a, b]).all() or b < a:
            raise ValueError("Invalid source interval")
        lo, hi = max(0, min(width, math.floor(a))), max(0, min(width, math.ceil(b)))
        mask[:, lo:hi] = 255
    return Image.fromarray(mask, mode="L")


def region_source_intervals(region, side, geometry, contract):
    ids = sorted(region["supported_physical"][side] + region["filled_physical"][side])
    return source_intervals([[p*contract.stride, p*contract.stride+contract.window_size] for p in ids], geometry)
