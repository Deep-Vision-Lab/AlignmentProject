import numpy as np
from PIL import Image

from alignment_overlays import aligned_window_overlays, window_color_labels


def test_window_colors_follow_exact_pairs_with_rtl_and_crop():
    originals = [Image.new('RGB', (150, 40), 'white') for _ in range(2)]
    geometry = [dict(crop=[10, 2, 120, 38], scale_x=2),
                dict(crop=[12, 2, 142, 38], scale_x=2)]
    masks = [np.zeros((40, 150), np.uint8) for _ in range(2)]
    for mask, geo in zip(masks, geometry):
        mask[:, geo['crop'][0]:geo['crop'][2]] = 255
    before = [mask.copy() for mask in masks]
    overlays = aligned_window_overlays(
        originals, masks, [dict(pairs=[(0, 2), (1, 0), (2, 1)])],
        [np.array([7, 4, 1]), np.array([9, 6, 3, 0])], geometry, 32, 20)
    colors = []
    for a, b in [(80, 42), (50, 102), (20, 72)]:
        np.testing.assert_array_equal(overlays[0][15, a + 5], overlays[1][15, b + 5])
        colors.append(tuple(overlays[0][15, a + 5]))
    assert len(set(colors)) == 3
    # B's unmatched window stays unmasked even if a neighboring mask is present.
    assert (overlays[1][15, 17] == 255).all()
    for image, overlay, mask, original_mask in zip(originals, overlays, masks, before):
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(overlay[mask == 0], np.asarray(image)[mask == 0])


def test_repeat_matches_share_color_transitively_across_regions():
    physical = [np.array([4, 3, 2, 1]), np.array([8, 7, 6, 5])]
    regions = [dict(pairs=[(0, 0), (0, 1)]),
               dict(pairs=[(1, 1), (2, 2)]), dict(pairs=[(3, 3)])]
    labels = window_color_labels(regions, physical)
    assert labels[0][4] == labels[1][8] == labels[1][7] == labels[0][3]
    assert labels[0][2] == labels[1][6]
    assert labels[0][1] == labels[1][5]
    assert len(set(labels[0].values())) == 3


def test_overlap_keeps_pure_window_colors_with_different_source_scales():
    originals = [Image.new('RGB', (50, 10), 'white') for _ in range(2)]
    geometry = [dict(crop=[5, 1, 45, 9], scale_x=1),
                dict(crop=[3, 1, 43, 9], scale_x=2)]
    masks = [np.zeros((10, 50), np.uint8) for _ in range(2)]
    masks[0][:, 5:45] = 255
    masks[1][:, 3:33] = 255
    overlays = aligned_window_overlays(
        originals, masks, [dict(pairs=[(0, 0), (1, 1), (2, 2)])],
        [np.array([2, 1, 0]), np.array([4, 2, 0])], geometry, 20, 10)
    colors = []
    for a, b in [(35, 28), (25, 18), (15, 8)]:
        np.testing.assert_array_equal(overlays[0][5, a], overlays[1][5, b])
        colors.append(tuple(overlays[0][5, a]))
    assert len(set(colors)) == 3
    # Full footprints overlap by half, but the display uses three clean bands.
    assert {tuple(pixel) for pixel in overlays[0][5, 5:45]} == set(colors)
    np.testing.assert_array_equal(overlays[0][5, 19], overlays[0][5, 15])
    np.testing.assert_array_equal(overlays[0][5, 20], overlays[0][5, 25])
    np.testing.assert_array_equal(overlays[0][5, 30], overlays[0][5, 35])


def test_gap_filled_windows_get_independent_colors_without_feature_rows():
    physical = [np.array([0, 2]), np.array([0, 2])]
    regions = [dict(pairs=[(0, 0), (1, 1)],
                    supported_physical=[[0, 2], [0, 2]], filled_physical=[[1], [1]])]
    labels = window_color_labels(regions, physical)
    assert labels[0][0] == labels[1][0]
    assert labels[0][2] == labels[1][2]
    assert len(set(labels[0].values()) | set(labels[1].values())) == 4
    originals = [Image.new('RGB', (30, 10), 'white') for _ in range(2)]
    masks = [np.ones((10, 30), np.uint8) * 255 for _ in range(2)]
    geometry = [dict(crop=[0, 0, 30, 10], scale_x=1)] * 2
    overlays = aligned_window_overlays(originals, masks, regions, physical, geometry, 10, 10)
    assert not np.array_equal(overlays[0][5, 15], overlays[1][5, 15])
    assert len({tuple(overlays[0][5, x]) for x in (5, 15, 25)}) == 3


def test_more_than_ten_windows_do_not_recycle_colors():
    originals = [Image.new('RGB', (96, 10), 'white') for _ in range(2)]
    masks = [np.ones((10, 96), np.uint8) * 255 for _ in range(2)]
    geometry = [dict(crop=[0, 0, 96, 10], scale_x=1)] * 2
    physical = [np.arange(12)[::-1]] * 2
    overlays = aligned_window_overlays(originals, masks, [dict(pairs=[(i, i) for i in range(12)])],
                                       physical, geometry, 8, 8)
    np.testing.assert_array_equal(*overlays)
    assert len({tuple(overlays[0][5, x]) for x in range(4, 96, 8)}) == 12


def test_empty_prediction_leaves_originals_unchanged():
    originals = [Image.new('RGB', (30, 10), 'white') for _ in range(2)]
    geometry = [dict(crop=[0, 0, 30, 10], scale_x=1)] * 2
    masks = [np.zeros((10, 30), np.uint8) for _ in range(2)]
    overlays = aligned_window_overlays(originals, masks, [], [[], []], geometry, 4, 2)
    for image, overlay in zip(originals, overlays):
        np.testing.assert_array_equal(overlay, np.asarray(image))


def test_notebook_refresh_reloads_stale_plotter_and_redraws_cached_sample(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import evaluation_utils

    book = json.loads((Path(__file__).parents[1] / 'notebooks/model_evaluation.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in book['cells']
                  if cell.get('id') == 'window-color-refresh')
    sides = []
    for side in range(2):
        path = tmp_path / f'line_{side}.png'
        Image.new('RGB', (150, 40), 'white').save(path)
        sides.append(dict(image=str(path)))
    geometry = dict(crop=[10, 2, 120, 38], scale_x=2)
    result = dict(pair=dict(sample_id='cached-pair', sides=sides), representation='fused',
                  masks=[np.ones((40, 150), np.uint8) * 255 for _ in range(2)],
                  match=dict(regions=[dict(pairs=[(0, 0), (1, 1), (2, 2)])]),
                  lines=[dict(physical=np.array([7, 4, 1]), geometry=geometry) for _ in range(2)])
    captured = []
    namespace = dict(sample_slots={1: dict(results=dict(fused=result))},
                     session=SimpleNamespace(config=SimpleNamespace(window_width=32, window_stride=20, rtl=True)),
                     display=captured.append)
    stale_plotter = lambda *args, **kwargs: None
    monkeypatch.setattr(evaluation_utils, 'plot_pair_alignment', stale_plotter)
    exec(source, namespace)
    assert namespace['plot_pair_alignment'] is not stale_plotter
    assert namespace['sample_slots'][1]['results']['fused'] is result
    assert len(captured) == 1 and len(captured[0].axes) == 2
    assert all('window colors' in ax.get_title() for ax in captured[0].axes)
    overlays = namespace['refreshed_window_overlays']
    np.testing.assert_array_equal(overlays[0], overlays[1])
    assert len({tuple(overlays[0][15, x]) for x in (25, 55, 85)}) == 3
