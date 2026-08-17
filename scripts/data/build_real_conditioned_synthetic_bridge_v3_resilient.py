#!/usr/bin/env python3
"""Resilient Bridge V3 builder.

This uses the scientific/rendering primitives from the canonical V3 builder but
changes candidate selection semantics: a single sentence that cannot satisfy the
readable-font/full-width constraints is rejected and resampled instead of aborting
the complete offline dataset job.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from scripts.data import build_real_conditioned_synthetic_bridge_v3 as core


_RENDER_RETRY_MARKERS = (
    "Full sentence does not fit",
    "Generated sentence is too sparse",
)


def _render_with_retries(
    segments: list[dict],
    fonts: list[Path],
    output: Path,
    mask_output: Path | None,
    rng: random.Random,
    args,
    *,
    attempts: int = 8,
):
    """Retry random safe font/gap assignments; return None for layout rejection."""
    last_error: RuntimeError | None = None
    for _ in range(max(1, int(attempts))):
        try:
            return core.render_segments(segments, fonts, output, mask_output, rng, args)
        except RuntimeError as exc:
            if not any(marker in str(exc) for marker in _RENDER_RETRY_MARKERS):
                raise
            last_error = exc
    if last_error is not None:
        return None
    return None


def _ordered_candidate_sentences(values: list[str], args) -> list[str]:
    """Prefer medium-length lines; very long candidates are most likely to overflow."""
    midpoint = (int(args.min_sentence_chars) + int(args.max_sentence_chars)) / 2.0
    unique: dict[str, str] = {}
    for value in values:
        key = core.normalize_match_text(value)
        if key:
            unique.setdefault(key, value)
    return sorted(
        unique.values(),
        key=lambda value: (abs(len(core.compact(value)) - midpoint), len(core.compact(value))),
    )


def _choose_positive(
    candidate_sentences: list[str],
    span_records: list[dict],
    fonts: list[Path],
    output_dir: Path,
    pos_image_rel: Path,
    pos_mask_rel: Path,
    rng: random.Random,
    args,
):
    max_islands = min(3, int(args.max_shared_islands))
    attempts = 0
    # Each candidate gets several independently sampled island configurations.
    for base_sentence in _ordered_candidate_sentences(candidate_sentences, args):
        island_counts = list(range(1, max_islands + 1))
        rng.shuffle(island_counts)
        for requested in island_counts:
            for _ in range(3):
                attempts += 1
                shared = core.choose_nonoverlapping_shared(span_records, rng, requested)
                segments = core.positive_segments(base_sentence, shared, rng, args.max_font_chunk_words)
                rendered_result = _render_with_retries(
                    segments,
                    fonts,
                    output_dir / pos_image_rel,
                    output_dir / pos_mask_rel,
                    rng,
                    args,
                )
                if rendered_result is None:
                    continue
                rendered, effective_size, pos_aug = rendered_result
                positive_text = " ".join(s["text"] for s in segments).strip()
                return {
                    "base_sentence": base_sentence,
                    "shared": shared,
                    "segments": segments,
                    "positive_text": positive_text,
                    "rendered": rendered,
                    "font_size": effective_size,
                    "augmentation": pos_aug,
                    "attempts": attempts,
                }
    return None


def _render_negatives(
    anchor_index: int,
    anchor_text: str,
    candidate_sentences: list[str],
    selected_base: str,
    texts: list[str],
    pool: list[tuple[int, str]],
    fonts: list[Path],
    anchor_id: str,
    output_dir: Path,
    rng: random.Random,
    args,
):
    needed = int(args.negatives_per_anchor)
    results: list[dict] = []
    selected_key = core.normalize_match_text(selected_base)
    seen: set[str] = {selected_key}
    render_attempts = 0

    def try_sentence(sentence: str) -> bool:
        nonlocal render_attempts
        key = core.normalize_match_text(sentence)
        if not key or key in seen:
            return False
        seen.add(key)
        neg_segments = core.sentence_segments(sentence, rng, args.max_font_chunk_words)
        neg_index = len(results)
        image_rel = Path("images") / f"{anchor_id}_neg_{neg_index:02d}.png"
        text_rel = Path("texts") / f"{anchor_id}_neg_{neg_index:02d}.txt"
        render_attempts += 1
        rendered_result = _render_with_retries(
            neg_segments, fonts, output_dir / image_rel, None, rng, args
        )
        if rendered_result is None:
            return False
        rendered_neg, neg_size, neg_aug = rendered_result
        results.append({
            "text": sentence,
            "segments": neg_segments,
            "rendered": rendered_neg,
            "font_size": neg_size,
            "augmentation": neg_aug,
            "image_rel": image_rel,
            "text_rel": text_rel,
        })
        return True

    for candidate in _ordered_candidate_sentences(candidate_sentences, args):
        if len(results) >= needed:
            break
        try_sentence(candidate)

    # If natural/precomposed candidates were exhausted, compose additional safe
    # sentences one at a time. Each new sentence is still checked by the canonical
    # no-overlap and glyph-safe construction before rendering.
    compose_attempts = 0
    while len(results) < needed and compose_attempts < 160:
        compose_attempts += 1
        sentence = core.compose_safe_sentence(
            anchor_index, anchor_text, pool, fonts, seen, rng, args
        )
        if sentence is None:
            break
        try_sentence(sentence)

    if len(results) != needed:
        return None, render_attempts, compose_attempts
    return results, render_attempts, compose_attempts


def build(args) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    for name in ("images", "texts", "masks"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    fonts = core._font_candidates(args.fonts)
    valid_pages, test_pages = core._positive_eval_pages(str(data_dir))
    heldout_pages = set(valid_pages) | set(test_pages)
    records = [
        r for r in core._all_unique_records(str(data_dir))
        if str(r["page_id"]) not in heldout_pages
    ]
    records.sort(key=lambda r: str(r["image_path"]))
    if args.max_anchors > 0:
        records = records[: args.max_anchors]

    usable, texts, spans = [], [], []
    for record in records:
        text = core._read(Path(record["text_path"]))
        candidates = core.candidate_span_records(
            text,
            min_chars=args.min_positive_chars,
            max_chars=args.max_phrase_chars,
            max_words=args.max_phrase_words,
        )
        candidates = [c for c in candidates if core.supported_fonts(c["text"], fonts)]
        if candidates:
            usable.append(record)
            texts.append(text)
            spans.append(candidates)
    if not usable:
        raise RuntimeError("No leakage-safe usable anchors with glyph-safe shared spans")

    rng = random.Random(args.seed)
    pool = core.phrase_pool(texts, fonts)
    rng.shuffle(pool)

    stats = {
        "dataset_version": core.DATASET_VERSION,
        "dataset_revision": core.DATASET_REVISION,
        "dataset_semantics": "full_sentence_multi_island_mixed_font_white_on_black_glyphsafe_fullwidth_resampled",
        "anchors_considered": len(records),
        "anchors_written": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "positive_shared_islands_1": 0,
        "positive_shared_islands_2": 0,
        "positive_shared_islands_3": 0,
        "positive_full_sentence_rows": 0,
        "mixed_font_positive_rows": 0,
        "mixed_font_negative_rows": 0,
        "positive_render_attempts": 0,
        "negative_render_attempts": 0,
        "negative_compose_attempts": 0,
        "layout_resampling": True,
        "negatives_per_anchor": args.negatives_per_anchor,
        "negative_ngram": args.negative_ngram,
        "min_overlap_word_chars": args.min_overlap_word_chars,
        "sentence_min_words": args.sentence_min_words,
        "sentence_max_words": args.sentence_max_words,
        "min_sentence_chars": args.min_sentence_chars,
        "max_sentence_chars": args.max_sentence_chars,
        "min_line_fill_ratio": args.min_line_fill_ratio,
        "font_size": args.font_size,
        "min_font_size": args.min_font_size,
        "max_font_size": args.max_font_size,
        "image_polarity": "white_text_on_black_background",
        "font_mixing": "per_segment_glyph_safe",
        "font_validation": "fonttools_unicode_cmap_on_shaped_text",
        "appearance_augmentation": {
            "geometric": False,
            "blur_prob": args.blur_prob,
            "blur_max_radius": args.blur_max_radius,
            "noise_prob": args.noise_prob,
            "noise_sigma_max": args.noise_sigma_max,
            "contrast_range": [args.contrast_min, args.contrast_max],
            "brightness_range": [args.brightness_min, args.brightness_max],
        },
        "mask_semantics": "white=shared synthetic x-region; black=distractor/gap/background",
        "heldout_page_count": len(heldout_pages),
        "fonts": [p.name for p in fonts],
        "seed": args.seed,
    }

    manifest_path = output_dir / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for anchor_index, (record, anchor_text, span_records) in enumerate(zip(usable, texts, spans)):
            # Ask for extra safe candidates. choose_safe_sentences may return fewer if
            # the source pool is small; the negative stage can compose more later.
            desired_candidates = 1 + int(args.negatives_per_anchor) + 8
            candidate_sentences = core.choose_safe_sentences(
                anchor_index, anchor_text, texts, pool, fonts,
                desired_candidates, rng, args,
            )
            if len(candidate_sentences) < 1 + int(args.negatives_per_anchor):
                raise RuntimeError(
                    f"Could not construct enough safe sentence candidates for {record['image_path']} "
                    f"({len(candidate_sentences)}/{1 + int(args.negatives_per_anchor)})"
                )

            anchor_id = core._anchor_id(record)
            pair_id = f"bridge_{anchor_id}"
            pos_image_rel = Path("images") / f"{anchor_id}_pos_00.png"
            pos_text_rel = Path("texts") / f"{anchor_id}_pos_00.txt"
            pos_mask_rel = Path("masks") / f"{anchor_id}_pos_00_mask.png"

            positive = _choose_positive(
                candidate_sentences, span_records, fonts, output_dir,
                pos_image_rel, pos_mask_rel, rng, args,
            )
            if positive is None:
                raise RuntimeError(
                    f"Could not render a readable positive for {record['image_path']} after resampling "
                    f"candidate sentences/shared islands; min_font_size={args.min_font_size}"
                )
            stats["positive_render_attempts"] += int(positive["attempts"])

            negative_results, neg_render_attempts, neg_compose_attempts = _render_negatives(
                anchor_index, anchor_text, candidate_sentences, positive["base_sentence"],
                texts, pool, fonts, anchor_id, output_dir, rng, args,
            )
            stats["negative_render_attempts"] += int(neg_render_attempts)
            stats["negative_compose_attempts"] += int(neg_compose_attempts)
            if negative_results is None:
                raise RuntimeError(
                    f"Could not render {args.negatives_per_anchor} readable negatives for "
                    f"{record['image_path']} after candidate resampling"
                )

            positive_text = positive["positive_text"]
            rendered = positive["rendered"]
            shared = positive["shared"]
            effective_size = positive["font_size"]
            pos_aug = positive["augmentation"]
            (output_dir / pos_text_rel).write_text(core.clean_render_text(positive_text), encoding="utf-8")

            shared_boxes = [s["bbox_px"] for s in rendered if s["kind"] == "shared"]
            shared_texts = [s["text"] for s in rendered if s["kind"] == "shared"]
            pos_fonts = sorted({s["font"] for s in rendered})
            shared_chars = sum(len(core.compact(t)) for t in shared_texts)
            anchor_chars = max(1, len(core.compact(anchor_text)))
            positive_chars = max(1, len(core.compact(positive_text)))

            row = {
                "pair_id": pair_id,
                "label_type": "medium_match",
                "A_page_id": str(record["page_id"]),
                "B_page_id": f"synthetic:{pair_id}",
                "A": core._side(
                    str(Path(record["image_path"]).resolve()),
                    str(Path(record["text_path"]).resolve()),
                ),
                "B": core._side(
                    pos_image_rel.as_posix(), pos_text_rel.as_posix(),
                    mask_path=pos_mask_rel.as_posix(),
                ),
                "scores": {
                    "text_score": 1.0,
                    "avg_sim": 1.0,
                    "coverage_A": min(1.0, shared_chars / anchor_chars),
                    "coverage_B": min(1.0, shared_chars / positive_chars),
                },
                "bridge": {
                    "dataset_version": core.DATASET_VERSION,
                    "dataset_revision": core.DATASET_REVISION,
                    "relation": "positive_full_sentence_multi_island",
                    "anchor_id": anchor_id,
                    "base_sentence": positive["base_sentence"],
                    "positive_full_sentence": positive_text,
                    "shared_island_count": len(shared),
                    "shared_texts": shared_texts,
                    "shared_boxes_px": shared_boxes,
                    "segments": rendered,
                    "alignment_mask_path": pos_mask_rel.as_posix(),
                    "fonts": pos_fonts,
                    "font_size": effective_size,
                    "image_polarity": "white_text_on_black_background",
                    "appearance_augmentation": pos_aug,
                    "glyph_safe": True,
                    "layout_resampled": positive["attempts"] > 1,
                    "render_attempts": positive["attempts"],
                    "negative_ngram_guarantee": args.negative_ngram,
                    "min_overlap_word_chars": args.min_overlap_word_chars,
                },
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["positive_rows"] += 1
            stats["positive_full_sentence_rows"] += 1
            stats[f"positive_shared_islands_{len(shared)}"] += 1
            if len(pos_fonts) > 1:
                stats["mixed_font_positive_rows"] += 1

            for neg_index, neg in enumerate(negative_results):
                (output_dir / neg["text_rel"]).write_text(
                    core.clean_render_text(neg["text"]), encoding="utf-8"
                )
                neg_fonts = sorted({s["font"] for s in neg["rendered"]})
                neg_row = {
                    "pair_id": pair_id,
                    "label_type": "no_shared_content",
                    "A_page_id": str(record["page_id"]),
                    "B_page_id": f"synthetic:{pair_id}:neg{neg_index}",
                    "A": core._side(
                        str(Path(record["image_path"]).resolve()),
                        str(Path(record["text_path"]).resolve()),
                    ),
                    "B": core._side(neg["image_rel"].as_posix(), neg["text_rel"].as_posix()),
                    "scores": {
                        "text_score": 0.0,
                        "avg_sim": 0.0,
                        "coverage_A": 0.0,
                        "coverage_B": 0.0,
                    },
                    "bridge": {
                        "dataset_version": core.DATASET_VERSION,
                        "dataset_revision": core.DATASET_REVISION,
                        "relation": "negative_full_sentence_no_shared_content",
                        "anchor_id": anchor_id,
                        "negative_text": neg["text"],
                        "segments": neg["rendered"],
                        "fonts": neg_fonts,
                        "font_size": neg["font_size"],
                        "image_polarity": "white_text_on_black_background",
                        "appearance_augmentation": neg["augmentation"],
                        "glyph_safe": True,
                        "negative_ngram_guarantee": args.negative_ngram,
                        "min_overlap_word_chars": args.min_overlap_word_chars,
                    },
                }
                manifest.write(json.dumps(neg_row, ensure_ascii=False) + "\n")
                stats["negative_rows"] += 1
                if len(neg_fonts) > 1:
                    stats["mixed_font_negative_rows"] += 1

            stats["anchors_written"] += 1
            if (anchor_index + 1) % 100 == 0:
                print(f"bridge_v3 anchors={anchor_index + 1}/{len(usable)}", flush=True)

    (output_dir / "metadata.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=== REAL-SYNTHETIC BRIDGE V3 RESILIENT ===")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output={output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    build(core.parse_args())
