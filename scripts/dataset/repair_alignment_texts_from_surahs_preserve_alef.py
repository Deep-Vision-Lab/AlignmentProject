#!/usr/bin/env python3
"""Repair ArabicDataset text using a verified standard-Imlaei Quran dump.

The local ``Surahs/surah_*.json`` files may contain only Uthmani spelling.
Uthmani dagger alef cannot be converted mechanically into modern spelling:
``بَعَثْنَـٰكُمْ`` must become ``بعثناكم``, while ``هَٰذَا`` remains ``هذا``.

This wrapper downloads a complete ``imlaei_simple`` Quran JSON dump once,
validates all 6,236 verse keys, caches the unmodified response under
``DATASET_ROOT/text_repair_references/``, and uses it as the clean reference.
The local Surah JSON remains the source for vocalized/tashkeel text.

It accepts all options of ``repair_alignment_texts_from_surahs.py`` plus:

``--imlaei-reference PATH``
    Use a local JSON or ``surah|ayah|text`` file instead of downloading.

``--refresh-imlaei-reference``
    Ignore the cache and download a fresh copy.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_VERSE_COUNT = 6236
CACHE_FILENAME = "quran_imlaei_simple_full.json"
NOTICE_FILENAME = "IMLAEI_REFERENCE_SOURCE.txt"

DOWNLOAD_URLS = (
    "https://api.islamic.app/v1/quran/verses/imlaei_simple",
    "https://api.quran.com/api/v4/quran/verses/imlaei_simple",
)

SOURCE_NOTICE = """Standard Imlaei Quran reference

Primary download source:
https://api.islamic.app/v1/quran/verses/imlaei_simple

The endpoint provides one JSON dump containing all 6,236 Quran verses in
``imlaei_simple`` script. The downloaded response is cached verbatim and used
only as the clean-spelling reference. Local Surahs/surah_*.json files remain
the source for vocalized/tashkeel text.

The repair wrapper validates the complete verse count and known samples,
including verse 2:56 containing بعثناكم, before it can modify dataset files.
"""


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


def _extract_wrapper_args(
    argv: Sequence[str] | None,
) -> tuple[list[str], Path | None, bool]:
    raw = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = []
    local_reference: Path | None = None
    refresh = False

    index = 0
    while index < len(raw):
        value = raw[index]
        if value == "--imlaei-reference":
            if index + 1 >= len(raw):
                raise ValueError("--imlaei-reference requires a file path")
            local_reference = Path(raw[index + 1]).expanduser().resolve()
            index += 2
            continue
        if value.startswith("--imlaei-reference="):
            local_reference = Path(value.split("=", 1)[1]).expanduser().resolve()
            index += 1
            continue
        if value == "--refresh-imlaei-reference":
            refresh = True
            index += 1
            continue
        forwarded.append(value)
        index += 1

    refresh = refresh or os.environ.get(
        "REFRESH_IMLAEI_QURAN", os.environ.get("REFRESH_TANZIL_QURAN", "0")
    ).lower() in {"1", "true", "yes", "on"}
    return forwarded, local_reference, refresh


def _parse_verse_key(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    left, right = value.split(":", 1)
    try:
        surah = int(left)
        ayah = int(right)
    except ValueError:
        return None
    if not (1 <= surah <= 114 and ayah >= 1):
        return None
    return surah, ayah


def _parse_json_reference(data: Any) -> dict[tuple[int, int], str]:
    records: Any = None
    if isinstance(data, Mapping):
        for key in ("ayahs", "verses", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            direct: dict[tuple[int, int], str] = {}
            for raw_key, raw_text in data.items():
                key = _parse_verse_key(raw_key)
                if key is not None and isinstance(raw_text, str) and raw_text.strip():
                    direct[key] = " ".join(raw_text.split())
            if direct:
                return direct
    elif isinstance(data, list):
        records = data

    verses: dict[tuple[int, int], str] = {}
    if not isinstance(records, list):
        return verses

    for item in records:
        if not isinstance(item, Mapping):
            continue
        key = _parse_verse_key(item.get("verse_key"))
        if key is None:
            surah = item.get("surah") or item.get("chapter_id")
            ayah = item.get("ayah") or item.get("verse_number")
            try:
                key = int(surah), int(ayah)
            except (TypeError, ValueError):
                continue
        text = (
            item.get("text")
            or item.get("text_imlaei_simple")
            or item.get("text_imlaei")
        )
        if isinstance(text, str) and text.strip():
            verses[key] = " ".join(text.split())
    return verses


def _parse_pipe_reference(text: str) -> dict[tuple[int, int], str]:
    verses: dict[tuple[int, int], str] = {}
    for raw_line in text.lstrip("\ufeff").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        try:
            key = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        verse_text = " ".join(parts[2].split())
        if verse_text:
            verses[key] = verse_text
    return verses


def _validate_reference(verses: Mapping[tuple[int, int], str]) -> None:
    if len(verses) != EXPECTED_VERSE_COUNT:
        raise ValueError(
            "Invalid Imlaei reference: expected "
            f"{EXPECTED_VERSE_COUNT} verses, found {len(verses)}."
        )

    required = {
        (1, 1): "بسم الله الرحمن الرحيم",
        (2, 56): "بعثناكم",
    }
    for key, fragment in required.items():
        actual = verses.get(key, "")
        if fragment not in actual:
            raise ValueError(
                f"Imlaei validation failed for {key[0]}:{key[1]}; "
                f"expected {fragment!r}, got {actual!r}."
            )


def _parse_reference_bytes(content: bytes) -> dict[tuple[int, int], str]:
    text = content.decode("utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        verses = _parse_pipe_reference(text)
    else:
        verses = _parse_json_reference(data)
    _validate_reference(verses)
    return verses


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _download_reference() -> tuple[bytes, str]:
    errors: list[str] = []
    for url in DOWNLOAD_URLS:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "User-Agent": "AlignmentProject-ArabicDataset-TextRepair/2.0",
                "Accept": "application/json,text/plain;q=0.9,*/*;q=0.5",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read()
            _parse_reference_bytes(content)
            return content, url
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            ValueError,
        ) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download a verified Imlaei Quran reference. Attempts:\n  - "
        + "\n  - ".join(errors)
        + "\nYou can download the JSON on another machine and pass it with "
        "--imlaei-reference /path/to/file.json."
    )


def _load_verified_imlaei(
    dataset_root: Path,
    local_reference: Path | None,
    refresh: bool,
) -> dict[tuple[int, int], str]:
    reference_dir = dataset_root / "text_repair_references"
    cache_path = reference_dir / CACHE_FILENAME
    notice_path = reference_dir / NOTICE_FILENAME

    if local_reference is not None:
        if not local_reference.is_file():
            raise FileNotFoundError(
                f"Local Imlaei reference not found: {local_reference}"
            )
        verses = _parse_reference_bytes(local_reference.read_bytes())
        print(f"Using local Imlaei reference: {local_reference}", flush=True)
        return verses

    if cache_path.is_file() and not refresh:
        try:
            verses = _parse_reference_bytes(cache_path.read_bytes())
            print(f"Using cached Imlaei reference: {cache_path}", flush=True)
            return verses
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(
                f"warning: cached Imlaei reference is invalid; refreshing: {exc}",
                file=sys.stderr,
            )

    print("Downloading complete Imlaei Simple Quran reference...", flush=True)
    content, source_url = _download_reference()
    verses = _parse_reference_bytes(content)
    _atomic_write_bytes(cache_path, content)
    _atomic_write_text(notice_path, SOURCE_NOTICE + f"\nDownloaded from: {source_url}\n")
    print(f"Cached Imlaei reference: {cache_path}", flush=True)
    return verses


def main(argv: Sequence[str] | None = None) -> int:
    forwarded_argv, local_reference, refresh = _extract_wrapper_args(argv)
    base_module = _load_base_module()
    original_load_surahs = base_module.load_surahs

    def load_surahs_with_verified_imlaei(surahs_dir: Path):
        local_verses = original_load_surahs(surahs_dir)
        dataset_root = surahs_dir.resolve().parent
        imlaei = _load_verified_imlaei(
            dataset_root=dataset_root,
            local_reference=local_reference,
            refresh=refresh,
        )

        missing = sorted(set(local_verses) - set(imlaei))
        if missing:
            rendered = ", ".join(f"{surah}:{ayah}" for surah, ayah in missing[:20])
            raise ValueError(
                "The local Surah data contains verse keys missing from the Imlaei "
                f"reference ({len(missing)} missing; first keys: {rendered})."
            )

        for key, verse in local_verses.items():
            verse.clean = imlaei[key]
            setattr(verse, "_spelling_source", "verified_imlaei_simple")

        sample = local_verses.get((2, 56))
        if sample is None or "بعثناكم" not in sample.clean:
            raise ValueError(
                "Final Imlaei validation failed: verse 2:56 does not contain بعثناكم."
            )

        print(
            "Standard-spelling references: "
            f"verified_imlaei={len(local_verses)} "
            f"local_tashkeel={len(local_verses)}",
            flush=True,
        )
        print(f"Validation 2:56: {sample.clean}", flush=True)
        return local_verses

    base_module.load_surahs = load_surahs_with_verified_imlaei

    print(
        "Verified Imlaei mode: clean spelling comes from a complete "
        "imlaei_simple Quran dump; vocalized text comes from local Surah JSON.",
        flush=True,
    )
    return int(base_module.main(forwarded_argv))


if __name__ == "__main__":
    raise SystemExit(main())
