from torch.utils.data import Subset

from real_span_feasibility import (
    filter_subset_by_span_feasibility,
    minimum_required_spans,
)


class _FakeRealDataset:
    text_key = "text"

    def __init__(self):
        self.samples = [
            {
                "pair_id": "short",
                "A": {"text": "باب", "line_image_path": "short_a.png"},
                "B": {"text": "باب باب", "line_image_path": "short_b.png"},
            },
            {
                "pair_id": "long",
                # DAL is right-joining, so consecutive DAL characters form
                # separate connected runs with an explicit boundary between
                # every pair: 70 runs + 69 boundaries = 139 states.
                "A": {"text": "د" * 70, "line_image_path": "long_a.png"},
                "B": {"text": "باب", "line_image_path": "long_b.png"},
            },
        ]

    @staticmethod
    def _read_text(value):
        return value


def test_connected_mode_counts_explicit_boundary_and_space_states(monkeypatch):
    monkeypatch.setenv("SPAN_TOKENIZATION_MODE", "connected_subword")
    assert minimum_required_spans("باب باب", max_span_chars=2) == 7


def test_connected_mode_filters_native_line_longer_than_window_lattice(monkeypatch):
    monkeypatch.setenv("SPAN_TOKENIZATION_MODE", "connected_subword")
    dataset = _FakeRealDataset()
    subset = Subset(dataset, [0, 1])

    filtered, stats = filter_subset_by_span_feasibility(
        dataset,
        subset,
        split_name="train",
        max_image_windows=125,
        max_span_chars=2,
    )

    assert list(filtered.indices) == [0]
    assert stats.kept == 1
    assert stats.removed == 1
    assert stats.max_required_spans == 139
    assert "pair_id=long" in stats.examples[0]
