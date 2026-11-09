"""Physical books, scope, checkpoint binding and immutable base settings."""
import ast
import json
from pathlib import Path

import pytest

from evaluate import resolved_match_settings
from evaluation_presets import physical_book_pair, settings_for_pair, load_pair_overrides, CONFIDENCE_KEYS


@pytest.mark.parametrize('a,b,result', [('A','B','A_B'), ('B','A','A_B'), ('A','C','A_C'), ('B','C','B_C')])
def test_physical_book_labels_come_from_images_not_positional_a_b(a, b, result):
    pair = dict(sides=[dict(image=f'catalog/images{a}/line.png'), dict(image=f'catalog/images{b}/line.png')])
    assert physical_book_pair(pair) == result
    pair['sides'] = [dict(image=f'Book{a}/images/line.png'), dict(image=f'Book{b}/images/line.png')]
    assert physical_book_pair(pair) == result


def test_ab_override_preserves_c_and_unknown_pairs_and_does_not_mutate_inputs():
    base = resolved_match_settings(dict(word_min_distinct_types=2))
    overrides = {'A_B': dict(word_min_distinct_types=3, word_min_cosine=.6)}
    for book in ['B', 'C', 'unknown']:
        pair = dict(sides=[dict(image='imagesA/a.png'), dict(image=f'images{book}/b.png')])
        result = settings_for_pair(pair, base, overrides)
        assert result['word_min_distinct_types'] == (3 if book == 'B' else 2)
    assert base['word_min_distinct_types'] == 2
    assert overrides == {'A_B': dict(word_min_distinct_types=3, word_min_cosine=.6)}


def test_presets_require_the_exact_checkpoint_and_only_change_confidence(tmp_path):
    path = tmp_path/'selected.json'
    settings = resolved_match_settings(dict(word_min_cosine=.6))
    path.write_text(json.dumps(dict(checkpoint_sha256='correct', settings=settings)))
    assert set(load_pair_overrides(path, 'correct')['A_B']) == set(CONFIDENCE_KEYS)
    with pytest.raises(ValueError, match='different checkpoint'):
        load_pair_overrides(path, 'wrong')


def test_notebook_session_routes_actual_predictions_and_cache_keys_by_book_pair():
    path = Path(__file__).resolve().parents[1] / 'notebooks/model_evaluation.ipynb'
    notebook = json.loads(path.read_text())
    definitions = [node for cell in notebook['cells'] if cell['cell_type']=='code'
        for node in ast.parse(''.join(cell['source'])).body
        if isinstance(node, ast.ClassDef) and node.name=='FolderMaskEvaluationSession']
    assert len(definitions) == 1
    class Base:
        def matching_cache_key(self, pair, representation): return ('base-cache-key',)
    namespace = dict(EvaluationSession=Base, MASK_GT_POLICY='fixture', json=json,
        settings_for_pair=settings_for_pair, PAIR_MATCH_OVERRIDES={'A_B': dict(word_min_cosine=.6)},
        predict_cached_pair=lambda session, pair, lines, representation, settings: settings)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    cls = namespace['FolderMaskEvaluationSession']; session = object.__new__(cls)
    session.settings = resolved_match_settings(dict(word_min_cosine=.4))
    session.get_line_features = lambda side: {}
    ab = dict(sides=[dict(image='imagesA/a.png'), dict(image='imagesB/b.png')])
    ac = dict(sides=[dict(image='imagesA/a.png'), dict(image='imagesC/c.png')])
    assert session.predict_pair(ab)['word_min_cosine'] == .6
    assert session.predict_pair(ac)['word_min_cosine'] == .4
    ab_key, ac_key = session.matching_cache_key(ab), session.matching_cache_key(ac)
    namespace['PAIR_MATCH_OVERRIDES']['A_B']['word_min_cosine'] = .7
    assert session.matching_cache_key(ab) != ab_key
    assert session.matching_cache_key(ac) == ac_key
    assert session.settings['word_min_cosine'] == .4
