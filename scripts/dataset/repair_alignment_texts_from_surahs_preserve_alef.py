#!/usr/bin/env python3
"""Run Surahs-based repair using standard Imlaei spelling when available.

A Quranic dagger alef cannot safely be converted to a normal alef everywhere:
``بَعَثْنَٰكُمْ`` needs the modern form ``بعثناكم``, while words such as
``هَٰذَا`` and ``الرَّحْمَٰن`` must remain ``هذا`` and ``الرحمن``. Therefore
this wrapper does not guess from Uthmani marks. It gives priority to local
``text_imlaei_simple`` / ``text_imlaei`` fields (and common aliases) in the
Surah JSON, which encode modern Arabic spelling, then delegates to the normal
repair pipeline.

It accepts the same command-line arguments as
``repair_alignment_texts_from_surahs.py``.
"""
from __future__ import annotations

import importlib.util
import sys
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

ALEF_WASLA = "\u0671"
PLAIN_ALEF = "\u0627"

IMLAEI_SIMPLE_KEYS = (
    "text_imlaei_simple",
    "imlaei_simple",
    "text_imlai_simple",
    "imlai_simple",
    "text_modern_simple",
    "modern_simple",
)

IMLAEI_TASHKEEL_KEYS = (
    "text_imlaei",
    "imlaei_text",
    "imlaei",
    "text_imlai",
    "imlai_text",
    "imlai",
    "text_modern",
    "modern_text",
)


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


def _strip_imlaei_marks(text: str, base_module) -> str:
    """Remove tashkeel from Imlaei text without inventing missing letters."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace(ALEF_WASLA, PLAIN_ALEF)
    normalized = base_module.ARABIC_DIACRITICS_RE.sub("", normalized)
    return base_module.collapse_spaces(normalized.replace("ـ", ""))


def _find_text(node: Mapping[str, Any], keys, base_module) -> str:
    return base_module.find_text(node, keys)


def _mark_source(verse, priority: int, source: str):
    if verse is not None:
        setattr(verse, "_spelling_priority", int(priority))
        setattr(verse, "_spelling_source", source)
    return verse


def main(argv: Sequence[str] | None = None) -> int:
    base_module = _load_base_module()
    original_choose_texts = base_module.choose_texts
    original_verse_quality = base_module.verse_quality
    original_load_surahs = base_module.load_surahs

    def choose_texts_preferring_imlaei(node: Mapping[str, Any]):
        imlaei_simple = _find_text(node, IMLAEI_SIMPLE_KEYS, base_module)
        imlaei_vocalized = _find_text(node, IMLAEI_TASHKEEL_KEYS, base_module)

        if imlaei_simple or imlaei_vocalized:
            clean_source = imlaei_simple or imlaei_vocalized
            clean = _strip_imlaei_marks(clean_source, base_module)
            tashkeel = base_module.collapse_spaces(imlaei_vocalized or imlaei_simple)
            if clean and tashkeel:
                priority = 3 if imlaei_simple else 2
                source = "imlaei_simple" if imlaei_simple else "imlaei"
                return _mark_source(
                    base_module.VerseText(clean=clean, tashkeel=tashkeel),
                    priority,
                    source,
                )

        # Keep the original local clean field as the fallback. Do not convert
        # every dagger alef: that would create false forms such as هاذا/الرحمان.
        return _mark_source(original_choose_texts(node), 0, "fallback")

    def verse_quality_preferring_imlaei(verse):
        return (
            int(getattr(verse, "_spelling_priority", 0)),
            *original_verse_quality(verse),
        )

    def load_surahs_with_diagnostic(surahs_dir: Path):
        verses = original_load_surahs(surahs_dir)
        imlaei_count = sum(
            int(getattr(verse, "_spelling_priority", 0)) > 0
            for verse in verses.values()
        )
        fallback_count = len(verses) - imlaei_count
        print(
            "Standard-spelling references: "
            f"imlaei={imlaei_count} fallback={fallback_count}",
            flush=True,
        )
        if imlaei_count == 0:
            raise ValueError(
                "No Imlaei Quran fields were found in Surahs/surah_*.json. "
                "Uthmani dagger alef alone cannot safely determine whether the "
                "modern spelling needs a full alef (for example بعثناكم) or must "
                "omit it (for example هذا and الرحمن). Add text_imlaei_simple or "
                "text_imlaei fields to the Surah JSON before applying repairs."
            )
        if fallback_count:
            print(
                "warning: some verses lack Imlaei fields and will retain their "
                "existing local clean spelling; inspect skipped/proposed rows.",
                file=sys.stderr,
            )
        return verses

    base_module.choose_texts = choose_texts_preferring_imlaei
    base_module.verse_quality = verse_quality_preferring_imlaei
    base_module.load_surahs = load_surahs_with_diagnostic

    print(
        "Standard Imlaei mode: modern spelling is preferred over Uthmani "
        "dagger-alef guessing (for example بعثناكم, not بعثنكم).",
        flush=True,
    )
    return int(base_module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
