"""Selection must favor recovered foreground without rewarding black pixels."""
import numpy as np
from PIL import Image
import pytest

from scripts.eval import select_book_pair_examples as selector


@pytest.mark.parametrize('group', ['A_B', 'A_C'])
def test_bounded_search_matches_exhaustive_white_recovery_ranking(tmp_path, monkeypatch, group):
    # Full white recovery with a little extra foreground should outrank a cleaner
    # prediction that misses white, even though its foreground IoU is lower.
    specs = [('exact', 20, 20, 20), ('recover_white', 20, 25, 20),
             ('miss_white', 20, 17, 17), ('shift', 20, 20, 18),
             ('miss_more', 20, 16, 16), ('flood_black', 80, 90, 80),
             ('large_truth_small_prediction', 60, 20, 20)]
    scored, predictions, exhaustive = [], {}, []
    for name, truth_size, predicted_size, intersection in specs:
        gt = np.zeros((1, 100), dtype=bool)
        gt[:, :truth_size] = True
        pred = np.zeros_like(gt)
        pred[:, :intersection] = True
        pred[:, truth_size:truth_size + predicted_size-intersection] = True
        path = tmp_path / (name + '.png')
        Image.fromarray(gt.astype('uint8') * 255).save(path)
        row = dict(pair_id=name, book_pair=group,
                   **{b: dict(mask_path=path.name) for b in group.split('_')})
        iou = intersection / (truth_size + predicted_size-intersection)
        scored.append(dict(row=row, a_iou=iou, b_iou=iou))
        predictions[name] = [pred, pred]
        precision = intersection/predicted_size
        specificity = 1-(predicted_size-intersection)/(100-truth_size)
        if precision >= .8 and specificity >= .9 and (group != 'A_B' or
                truth_size <= 45 and .1 <= pred.mean() <= .45 and
                np.flatnonzero(pred.any(0))[-1]+1 <= 45):
            exhaustive.append((name, 5*intersection/(4*truth_size+predicted_size), iou, specificity))
    monkeypatch.setattr(selector, 'predict_candidate', lambda session, dataset, row, representation:
                        dict(masks=predictions[row['pair_id']]))
    session = type('FakeSession', (), dict(feature_cache={}))()
    actual = selector.choose_foreground(scored, session, tmp_path, 'fused', group)
    expected = sorted(exhaustive, key=lambda x: (-x[1], -x[2], -x[3], x[0]))[:5]
    assert [r['row']['pair_id'] for r in actual] == [r[0] for r in expected]
    assert len(actual) == 5
    assert actual[1]['row']['pair_id'] == 'recover_white'
    assert next(r for r in actual if r['row']['pair_id'] == 'miss_white')['a_iou'] > actual[1]['a_iou']
    # Reusing measured masks produces the same selection without new inference.
    monkeypatch.setattr(selector, 'predict_candidate', lambda *a: pytest.fail('Unnecessary inference'))
    repeated = selector.choose_foreground(scored, session, tmp_path, 'fused', group)
    assert [r['row']['pair_id'] for r in repeated] == [r[0] for r in expected]
