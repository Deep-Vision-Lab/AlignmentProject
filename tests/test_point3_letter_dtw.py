import numpy as np

from Evaluation import eval_point3_hard_paths as point3


def test_letter_dtw_helpers_are_installed():
    assert callable(point3._letter_dtw_side)
    assert callable(point3._hard_letter_dtw_path)
    assert callable(point3.save_letter_dtw_overview)
    assert callable(point3._extract_window_images)
    assert callable(point3._plot_window_image_axis)


def test_hard_letter_dtw_monotonic_and_complete():
    costs = np.asarray(
        [
            [0.1, 2.0, 3.0],
            [0.2, 0.1, 2.0],
            [1.5, 0.2, 0.1],
            [2.0, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    path, matrix = point3._hard_letter_dtw_path(
        costs,
        vertical_penalty=0.05,
        horizontal_penalty=0.30,
        position_prior_weight=0.15,
        disable_horizontal_when_feasible=True,
    )
    assert matrix.shape == (4, 3)
    assert path[0] == (0, 0)
    assert path[-1] == (3, 2)
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert (i1 - i0, j1 - j0) in {(1, 0), (1, 1), (0, 1)}


def test_letter_panel_uses_images_not_text_for_x_axis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    result = {
        "costs": np.ones((6, 3), dtype=np.float32),
        "letters": ["ا", "ب", "ت"],
        "physical_window_indices": np.asarray([4, 5, 6, 7, 8, 9]),
        "path": [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)],
    }

    fig, (heat_ax, window_ax) = plt.subplots(2, 1)
    point3._plot_letter_panel(heat_ax, result, "test")
    labels = [tick.get_text() for tick in heat_ax.get_xticklabels()]
    assert not any(label.startswith("W") for label in labels)

    windows = [
        Image.fromarray(np.full((128, 32), 255 - 20 * i, dtype=np.uint8))
        for i in range(6)
    ]
    point3._plot_window_image_axis(window_ax, windows)
    assert len(window_ax.images) == 6
    assert len(window_ax.get_xticks()) == 0
    left, right = window_ax.get_xlim()
    assert left > right  # RTL: first model window is displayed on the right.
    plt.close(fig)



def test_manifest_transcript_path_overrides_filename_guess(tmp_path):
    side = tmp_path / "A"
    images = side / "linesImages"
    guessed = side / "text" / "final" / "original"
    exact = side / "manifest_text"
    images.mkdir(parents=True)
    guessed.mkdir(parents=True)
    exact.mkdir(parents=True)

    image = images / "line_01.png"
    image.write_bytes(b"not-an-image-needed-for-path-test")
    guessed_text = guessed / "line_01.txt"
    guessed_text.write_text("خطأ", encoding="utf-8")
    exact_text = exact / "different_name.txt"
    exact_text.write_text("صحيح", encoding="utf-8")

    resolved = point3._transcript_path_for_line(image, exact_text)
    assert resolved == exact_text
    assert resolved.read_text(encoding="utf-8") == "صحيح"



def test_native_line_transcript_wins_over_mismatched_manifest(tmp_path):
    side = tmp_path / "DatasetPairs" / "page_pairs" / "pair_000001" / "A"
    images = side / "linesImages"
    texts = side / "text" / "final" / "original"
    images.mkdir(parents=True)
    texts.mkdir(parents=True)

    image = images / "line_07.png"
    image.write_bytes(b"placeholder")
    native = texts / "line_07.txt"
    native.write_text("الصحيح", encoding="utf-8")

    wrong = tmp_path / "wrong.txt"
    wrong.write_text("خطأ", encoding="utf-8")

    resolved = point3._transcript_path_for_line(image, wrong)
    assert resolved == native


def test_native_line_without_same_stem_transcript_fails(tmp_path):
    import pytest

    side = tmp_path / "DatasetPairs" / "page_pairs" / "pair_000001" / "A"
    images = side / "linesImages"
    texts = side / "text" / "final" / "original"
    images.mkdir(parents=True)
    texts.mkdir(parents=True)

    image = images / "line_07.png"
    image.write_bytes(b"placeholder")
    wrong = texts / "line_08.txt"
    wrong.write_text("غير صحيح", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        point3._transcript_path_for_line(image, wrong)
