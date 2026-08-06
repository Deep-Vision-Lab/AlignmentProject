from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connected_subword_mode import (
    BOUNDARY_TOKEN,
    SPACE_TOKEN,
    connected_span_slices,
    connected_units,
    minimum_connected_spans,
    render_connected_units,
    split_connected_word,
)


def test_joining_runs_follow_arabic_connectivity():
    assert split_connected_word("كتب") == ["كتب"]
    assert split_connected_word("كتاب") == ["كتا", "ب"]
    assert split_connected_word("الرحمن") == ["ا", "لر", "حمن"]


def test_boundaries_and_spaces_are_distinct():
    assert render_connected_units("الرحمن الرحيم") == [
        "ا",
        BOUNDARY_TOKEN,
        "لر",
        BOUNDARY_TOKEN,
        "حمن",
        SPACE_TOKEN,
        "ا",
        BOUNDARY_TOKEN,
        "لر",
        BOUNDARY_TOKEN,
        "حيم",
    ]


def test_combining_marks_stay_with_the_base_cluster():
    units = connected_units("بِسم")
    subwords = [unit.text for unit in units if unit.kind == "subword"]
    assert subwords == ["بِسم"]


def test_hamza_is_non_joining():
    assert split_connected_word("سءل") == ["س", "ء", "ل"]


def test_variable_spans_can_cover_multiple_connected_units():
    units = connected_units("الرحمن")
    assert (0, 3) in connected_span_slices(units, max_units=3)
    assert minimum_connected_spans("الرحمن", max_units=3) < len(units)


def test_variable_spans_do_not_cross_word_spaces():
    units = connected_units("كتب علم")
    space_index = next(index for index, unit in enumerate(units) if unit.kind == "space")
    for start, length in connected_span_slices(units, max_units=3):
        covered = range(start, start + length)
        assert not (space_index in covered and length > 1)
