"""Scope validation-selected confidence settings to their physical book pair.

Book identity comes from the supplied image catalog, never transcript content,
alignment labels or masks. Unknown/C pairs keep the caller's base settings.
"""
import json
from pathlib import Path

from evaluate import resolved_match_settings

ROOT = Path(__file__).resolve().parent
CONFIDENCE_KEYS = ('word_max_nll', 'word_min_posterior', 'word_min_posterior_short', 'word_min_cosine',
                   'word_min_distinct_types', 'word_min_total_characters', 'word_visual_weight',
                   'word_strong_pair_max_nll','word_strong_pair_min_cosine')


def physical_book_pair(pair):
    books = []
    for side in pair['sides']:
        value = side.get('book_id')
        if value and value.startswith('Book'): value = value[4:]
        if value not in ('A', 'B', 'C'):
            path = Path(side['image'])
            value = next((name[len(prefix):] for name in (path.parent.name, path.parent.parent.name)
                for prefix in ('images', 'Book') if name.startswith(prefix) and name[len(prefix):] in ('A', 'B', 'C')), None)
        books.append(value)
    if len(books) != 2 or any(value is None for value in books): return None
    return '_'.join(sorted(books))


def load_pair_overrides(path, checkpoint_sha256):
    saved = json.loads(Path(path).read_text())
    if saved['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Book-pair confidence preset belongs to a different checkpoint')
    return {'A_B': {key: saved['settings'][key] for key in CONFIDENCE_KEYS}}


def settings_for_pair(pair, base, overrides):
    options = dict(base)
    options.update(overrides.get(physical_book_pair(pair), {}))
    return resolved_match_settings(options)
