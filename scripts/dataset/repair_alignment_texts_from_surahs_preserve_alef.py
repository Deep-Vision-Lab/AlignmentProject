#!/usr/bin/env python3
"""Run the Surahs-based dataset repair while preserving Quranic alef marks.

The original Surahs wrapper removes U+0670 ARABIC LETTER SUPERSCRIPT ALEF
(dagger alef) together with ordinary tashkeel. In Uthmani text, that character
can carry a long /a/ where no full-size alef exists, so deleting it can produce
clean forms with a missing alef.

This compatibility wrapper converts dagger alef and alef wasla to a normal
Arabic alef *before* the remaining diacritics are removed, then delegates to
``repair_alignment_texts_from_surahs.py``. It accepts exactly the same command
line options.
"""
from __future__ import annotations

import importlib.util
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

DAGGER_ALEF = "\u0670"
ALEF_WASLA = "\u0671"
PLAIN_ALEF = "\u0627"


def _load_base_module():
    base_path = Path(__file__).with_name("repair_alignment_texts_from_surahs.py")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Surahs repair wrapper not found: {base_path}")

    module_name = "_alignment_repair_from_surahs_base"
    spec = importlib.util.spec_from_file_location(module_name, base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the base Surahs repair wrapper: {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_text_preserving_alef(text: str, base_module) -> str:
    """Remove tashkeel without deleting Quranic alef information."""
    normalized = unicodedata.normalize("NFKC", str(text))

    # U+0670 is included in the base module's diacritic regex. Convert it first
    # so long-a information is not silently deleted. U+0671 is a full alef
    # letter with a wasla sign; normalize it to the plain alef used by the
    # character-level alignment text.
    normalized = normalized.replace(DAGGER_ALEF, PLAIN_ALEF)
    normalized = normalized.replace(ALEF_WASLA, PLAIN_ALEF)

    return base_module.ARABIC_DIACRITICS_RE.sub("", normalized).replace("ـ", "")


def main(argv: Sequence[str] | None = None) -> int:
    base_module = _load_base_module()

    # choose_texts() and collect_verses() resolve this function dynamically from
    # the module globals, so replacing it fixes every clean-reference path:
    # explicit clean fields, generic text fields, and numeric verse maps.
    base_module.strip_diacritics = (
        lambda text: clean_text_preserving_alef(text, base_module)
    )

    print(
        "Alef-preserving mode: dagger alef (U+0670) and alef wasla "
        "(U+0671) are converted to plain alef in clean references.",
        flush=True,
    )
    return int(base_module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
