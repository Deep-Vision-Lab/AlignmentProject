"""Small image-cosine matrices for evaluation-only many-to-many alignment."""
import numpy as np

from evaluate import local_repeat_regions, region_mask, shared_regions


def run(matrix, **settings):
    a, b = matrix.shape
    return local_repeat_regions(matrix, np.arange(a), np.arange(b),
                                score_mode='raw', threshold=.5,
                                min_distinct_windows_a=3,
                                min_distinct_windows_b=3,
                                min_matched_pairs=4, **settings)


def geometry(width):
    return dict(source_size=[width, 8], crop=[0, 0, width, 8], scale_x=1.)


def test_three_by_five_horizontal_repeats_support_every_window():
    cosine = np.full((3, 5), .1)
    expected = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4)]
    for pair in expected:
        cosine[pair] = .95
    result = run(cosine)
    assert len(result['regions']) == 1
    region = result['regions'][0]
    assert region['pairs'] == expected
    assert region['horizontal_repeats'] == 2 and region['vertical_repeats'] == 0
    assert region['support'] == [3, 5] and region['matched_pairs'] == 5
    assert region['supported_physical'] == [[0, 1, 2], [0, 1, 2, 3, 4]]
    assert [step['transition'] for step in region['path_steps']] == [
        'diagonal', 'horizontal_repeat', 'horizontal_repeat', 'diagonal', 'diagonal']
    assert all(step['cosine'] == .95 and step['reward'] > 0 for step in region['path_steps'])
    np.testing.assert_array_equal(region_mask([region], 0, geometry(30), 10, 10), 255)
    np.testing.assert_array_equal(region_mask([region], 1, geometry(50), 10, 10), 255)
    assert not shared_regions(cosine, np.arange(3), np.arange(5), threshold=.5,
                              score_mode='raw', min_windows=5)['regions']


def test_five_by_three_vertical_repeats_support_every_window():
    cosine = np.full((5, 3), .1)
    expected = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2)]
    for pair in expected:
        cosine[pair] = .95
    region = run(cosine)['regions'][0]
    assert region['pairs'] == expected
    assert region['vertical_repeats'] == 2 and region['horizontal_repeats'] == 0
    assert region['support'] == [5, 3] and region['matched_pairs'] == 5
    np.testing.assert_array_equal(region_mask([region], 0, geometry(50), 10, 10), 255)
    np.testing.assert_array_equal(region_mask([region], 1, geometry(30), 10, 10), 255)


def test_true_gap_skips_unrelated_middle_without_masking_it():
    cosine = np.full((5, 4), .1)
    for pair in [(0, 0), (1, 1), (3, 2), (4, 3)]:
        cosine[pair] = .95
    region = run(cosine)['regions'][0]
    assert region['pairs'] == [(0, 0), (1, 1), (3, 2), (4, 3)]
    assert region['true_gaps'] == 1
    assert any(step['transition'] == 'gap_a' and step['i'] == 2
               for step in region['path_steps'])
    assert region['supported_physical'][0] == [0, 1, 3, 4]
    assert 2 not in region['filled_physical'][0]
    mask = region_mask([region], 0, geometry(50), 10, 10)
    assert not mask[:, 20:30].any()
    assert mask[:, :20].all() and mask[:, 30:].all()


def test_isolated_similarity_does_not_pass_region_support():
    cosine = np.full((7, 7), .1)
    cosine[3, 4] = .99
    result = run(cosine)
    assert result['regions'] == []
    assert any(item['reason'] == 'insufficient_repeat_support'
               for item in result['rejected'])


def test_repeat_limit_prevents_one_to_many_overmatching():
    cosine = np.full((1, 20), .95)
    result = local_repeat_regions(cosine, [0], np.arange(20), threshold=.5,
                                  score_mode='raw', max_consecutive_repeats=2,
                                  min_distinct_windows_a=1,
                                  min_distinct_windows_b=1, min_matched_pairs=1)
    assert sum(region['matched_pairs'] for region in result['regions']) <= 3
    assert all(region['horizontal_repeats'] <= 2 for region in result['regions'])


def test_rtl_logical_path_projects_to_physical_source_pixels():
    cosine = np.full((3, 5), .1)
    for pair in [(0, 0), (0, 1), (0, 2), (1, 3), (2, 4)]:
        cosine[pair] = .95
    result = local_repeat_regions(cosine, [2, 1, 0], [4, 3, 2, 1, 0],
                                  score_mode='raw', threshold=.5,
                                  min_distinct_windows_a=3,
                                  min_distinct_windows_b=3, min_matched_pairs=4)
    region = result['regions'][0]
    assert region['pairs'][0] == (0, 0)
    assert region['supported_physical'] == [[0, 1, 2], [0, 1, 2, 3, 4]]
    mask_a = region_mask([region], 0, geometry(30), 10, 10)
    mask_b = region_mask([region], 1, geometry(50), 10, 10)
    assert mask_a[:, 0:30].all() and mask_b[:, 0:50].all()


def test_separate_local_regions_do_not_cross_or_reuse_windows():
    cosine = np.full((14, 14), .1)
    for i in (1, 2, 3, 4, 8, 9, 10, 11):
        cosine[i, i + 1] = .95
    regions = run(cosine)['regions']
    assert len(regions) == 2
    assert regions[0]['logical_ranges'][0][1] < regions[1]['logical_ranges'][0][0]
    assert regions[0]['logical_ranges'][1][1] < regions[1]['logical_ranges'][1][0]
    assert not set(regions[0]['supported_physical'][0]) & set(regions[1]['supported_physical'][0])
    assert not set(regions[0]['supported_physical'][1]) & set(regions[1]['supported_physical'][1])
