import numpy as np

from Evaluation._eval_utils import NWStep
from Evaluation import nw_discontinuous_regions
from Evaluation import nw_physical_mapping


def test_arabic_heatmap_cell_mapping_uses_physical_window_centers(monkeypatch):
    monkeypatch.setenv("NW_VIS_WINDOW_SIZE", "32")
    monkeypatch.setenv("NW_VIS_WINDOW_STRIDE", "8")
    monkeypatch.setenv("NW_VIS_REGION_MAPPING", "cell")

    # Logical Arabic windows 88..97 are physical windows 27..36 after RTL flip.
    # Their centers are 232 and 304 px, so heatmap-cell boundaries are 228..308.
    left, right = nw_physical_mapping.patch_range_to_pixels(
        88, 98, 125, 1024, True
    )
    assert np.isclose(left, 228.0)
    assert np.isclose(right, 308.0)


def test_footprint_mapping_is_wider_than_heatmap_cell_mapping(monkeypatch):
    monkeypatch.setenv("NW_VIS_WINDOW_SIZE", "32")
    monkeypatch.setenv("NW_VIS_WINDOW_STRIDE", "8")

    monkeypatch.setenv("NW_VIS_REGION_MAPPING", "footprint")
    footprint = nw_physical_mapping.patch_range_to_pixels(88, 98, 125, 1024, True)

    monkeypatch.setenv("NW_VIS_REGION_MAPPING", "cell")
    cells = nw_physical_mapping.patch_range_to_pixels(88, 98, 125, 1024, True)

    assert footprint == (216.0, 320.0)
    assert cells == (228.0, 308.0)


def test_short_nw_holes_stay_in_one_real_region(monkeypatch):
    monkeypatch.setenv("NW_REGION_SUPPORT_FLOOR", "0.0")
    monkeypatch.setenv("NW_REGION_MAX_BRIDGE_STEPS", "3")
    monkeypatch.setenv("NW_REGION_MIN_MATCH_STEPS", "2")

    # Two small gap steps between strong diagonal correspondences should be one
    # visual aligned region, not three separate red masks.
    steps = [
        NWStep(0, 0, "match", 0.8),
        NWStep(1, None, "gap_in_line2", None),
        NWStep(2, None, "gap_in_line2", None),
        NWStep(3, 1, "match", 0.8),
    ]
    scores = np.full((4, 2), -0.5, dtype=np.float32)
    scores[0, 0] = 0.8
    scores[3, 1] = 0.8

    runs, _score = nw_discontinuous_regions._supported_runs(
        steps, scores, gap_penalty=-0.3
    )

    assert len(runs) == 1
    assert runs[0]["path"] == [(0, 0), (3, 1)]
    assert runs[0]["traceback_steps"] == 4
