from connected_subword_mode import (
    BOUNDARY_TOKEN,
    SPACE_TOKEN,
    connected_units,
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
