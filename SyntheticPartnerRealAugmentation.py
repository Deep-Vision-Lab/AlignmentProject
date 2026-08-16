"""Create one-sided synthetic partners for clean no-shared real lines.

The anchor line is NEVER modified.  Its original ``no_shared_content`` mate is
used as a realistic distractor canvas.  One to three complete subword strips in
that mate are replaced by donor strips whose exact canonical text occurs in the
anchor.  Donors come from different leakage-safe real training images, so each
inserted aligned island has genuine different-handwriting pixels.

This deliberately differs from generic real-data augmentation:

* canonical high/medium real pairs stay completely untouched;
* the no-shared anchor stays completely untouched;
* only the newly created partner is synthetic;
* every injected region is bbox-exact and contains complete subwords;
* multiple aligned regions preserve RTL reading order;
* unmatched content from the original no-shared mate remains in the partner.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from scripts.data.augment_real_bbox_strip_injection import (
    DonorRun,
    LineState,
    SourceLine,
    _build_donor_index,
    _canonical,
    _compact,
    _copy_source,
    _run_is_x_isolated,
    _run_text,
    _splice_full_height_run,
    _target_candidates,
    _x_bounds,
)


@dataclass(frozen=True)
class PartnerSynthesisConfig:
    height: int = 128
    min_regions: int = 1
    max_regions: int = 3
    max_run_boxes: int = 3
    min_chars: int = 4
    max_chars: int = 28
    width_ratio_min: float = 0.50
    width_ratio_max: float = 2.00
    max_attempts: int = 120


def build_training_donor_index(
    lines: Iterable[SourceLine], config: PartnerSynthesisConfig
):
    return _build_donor_index(
        lines,
        max_run_boxes=int(config.max_run_boxes),
        min_chars=int(config.min_chars),
        max_chars=int(config.max_chars),
    )


def _anchor_runs(
    anchor: SourceLine,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    config: PartnerSynthesisConfig,
) -> list[DonorRun]:
    runs: list[DonorRun] = []
    boxes = anchor.boxes
    max_boxes = min(int(config.max_run_boxes), len(boxes))
    for size in range(1, max_boxes + 1):
        for start in range(0, len(boxes) - size + 1):
            if not _run_is_x_isolated(boxes, start, size):
                continue
            selected = boxes[start : start + size]
            text = _run_text(selected)
            canonical = _canonical(text)
            compact_len = len(_compact(canonical))
            if not canonical or not (
                int(config.min_chars) <= compact_len <= int(config.max_chars)
            ):
                continue
            key = (size, canonical)
            donors = donor_index.get(key, [])
            if not any(run.line.image_path != anchor.image_path for run in donors):
                continue
            x0, x1 = _x_bounds(selected)
            runs.append(DonorRun(anchor, start, size, text, canonical, x0, x1))
    return runs


def _non_overlapping(runs: list[DonorRun]) -> bool:
    ordered = sorted(runs, key=lambda run: run.start)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.start + previous.size > current.start:
            return False
    return True


def _choose_anchor_runs(
    rng: random.Random,
    candidates: list[DonorRun],
    count: int,
) -> list[DonorRun] | None:
    if count <= 0 or not candidates:
        return None
    for _ in range(80):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        chosen: list[DonorRun] = []
        used_texts: set[str] = set()
        for run in shuffled:
            if run.canonical_text in used_texts:
                continue
            trial = chosen + [run]
            if not _non_overlapping(trial):
                continue
            chosen.append(run)
            used_texts.add(run.canonical_text)
            if len(chosen) == count:
                return sorted(chosen, key=lambda item: item.start)
    return None


def _choose_donor(
    rng: random.Random,
    anchor_run: DonorRun,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    forbidden_images: set,
) -> DonorRun | None:
    donors = [
        run
        for run in donor_index.get((anchor_run.size, anchor_run.canonical_text), [])
        if run.line.image_path not in forbidden_images
    ]
    if not donors:
        return None
    return rng.choice(donors)


def synthesize_partner(
    rng: random.Random,
    anchor: SourceLine,
    unrelated_mate: SourceLine,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    regions: int,
    config: PartnerSynthesisConfig,
) -> tuple[LineState, dict] | None:
    """Return a synthetic mate containing ``regions`` anchor-matching islands.

    The original anchor is returned only through metadata and is never mutated.
    Target replacement positions are constrained to the same semantic RTL order
    as the selected anchor spans so the aligned islands form a monotonic path.
    """
    regions = int(regions)
    if regions < int(config.min_regions) or regions > int(config.max_regions):
        return None

    candidates = _anchor_runs(anchor, donor_index, config)
    if not candidates:
        return None

    for _ in range(max(1, int(config.max_attempts))):
        anchor_runs = _choose_anchor_runs(rng, candidates, regions)
        if not anchor_runs:
            return None

        state = _copy_source(unrelated_mate)
        forbidden_images = {anchor.image_path, unrelated_mate.image_path}
        previous_target_end = -1
        details = []
        success = True

        for region_index, anchor_run in enumerate(anchor_runs, start=1):
            donor = _choose_donor(rng, anchor_run, donor_index, forbidden_images)
            if donor is None:
                success = False
                break

            donor_width = donor.x1 - donor.x0
            target_starts = _target_candidates(
                state,
                donor.size,
                donor_width,
                donor.text,
                float(config.width_ratio_min),
                float(config.width_ratio_max),
            )
            # Preserve the same RTL sequence order as the anchor runs and avoid
            # replacing any previously inserted island.
            target_starts = [
                start
                for start in target_starts
                if start >= previous_target_end
            ]
            if not target_starts:
                success = False
                break

            rng.shuffle(target_starts)
            placed = None
            placed_start = None
            for target_start in target_starts:
                trial = _splice_full_height_run(
                    state,
                    donor,
                    target_start,
                    int(config.height),
                )
                if trial is not None:
                    placed = trial
                    placed_start = target_start
                    break
            if placed is None or placed_start is None:
                success = False
                break

            state = placed
            previous_target_end = int(placed_start + donor.size)
            forbidden_images.add(donor.line.image_path)
            details.append(
                {
                    "region": region_index,
                    "shared_text": anchor_run.canonical_text,
                    "anchor_box_start": int(anchor_run.start),
                    "anchor_box_count": int(anchor_run.size),
                    "partner_target_box_start": int(placed_start),
                    "donor_image": str(donor.line.image_path),
                    "donor_pair_id": str(donor.line.pair_id),
                    "operation": state.operations[-1],
                }
            )

        if success and len(details) == regions:
            return state, {
                "mode": "clean_anchor_one_sided_bbox_synthetic_partner",
                "regions": regions,
                "anchor_image": str(anchor.image_path),
                "base_unrelated_mate": str(unrelated_mate.image_path),
                "region_details": details,
                "anchor_modified": False,
                "partner_modified": True,
                "order_policy": "aligned islands preserve anchor RTL order",
                "text_policy": "partner transcript rebuilt from final bbox sequence",
                "complete_subword_policy": "bbox-exact full-height strips only",
            }

    return None
