"""The recall trade-off uses validation only and quarantines shared images."""
import pytest
from scripts.eval.final_fragments import exclude_reviewed, select_final


def trial(name, recall, fpr, iou=.6):
    return dict(name=name, metrics=dict(recall=recall, negative_pair_fpr=fpr, positive_iou=iou))


def test_recall_selection_respects_false_positive_budget_and_iou_tiebreak():
    high_recall = trial('outside budget', .99, .11)
    eligible = trial('recall', .85, .10)
    precise = trial('iou', .80, .01, .9)
    assert select_final([high_recall, eligible, precise], .1) is eligible
    tied = trial('tie', .85, .09, .8)
    assert select_final([eligible, tied], .1) is tied
    with pytest.raises(ValueError, match='No validation preset'):
        select_final([high_recall], .1)


def test_reviewed_source_images_are_excluded_even_in_another_pair():
    def item(name, *hashes):
        return dict(record=dict(sample_id=name, images=[dict(sha256=h) for h in hashes]))
    source = [item('reviewed', 'A', 'B'), item('shares B', 'C', 'B'), item('independent', 'D', 'E')]
    retained, excluded = exclude_reviewed(source, {'A', 'B'})
    assert retained == [source[-1]]
    assert excluded == ['reviewed', 'shares B']
