"""Language filtering for the multilingual retrieval pipeline.

MSMARCO-XI is multilingual: every chunk carries language metadata
(``language``, plus ``source_lang`` / ``target_lang``). These helpers let the
pipeline filter or prefer chunks by language without hardcoding any language.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .models import RetrievalResult, chunk_language

# Languages actually present in the processed corpus (Segment 1 stats). The
# implementation is generic — this set is only used to decide whether a
# detected language is known.
SUPPORTED_LANGUAGES = ("eng_Latn", "urd_Arab", "hin_Deva")

# Arabic-script ranges cover Urdu (urd_Arab) and other Arabic-script languages.
_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")

# Devanagari script (Hindi)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")


def detect_script_language(text: str) -> Optional[str]:
    """Coarse script-based language detection (deterministic, no model).

    * Any Devanagari character              -> ``hin_Deva``
    * Any Arabic-script character            -> ``urd_Arab``
    * Only Latin-script characters (ASCII)   -> ``eng_Latn``
    * Otherwise                              -> ``None`` (unknown)
    """
    text = text or ""
    # Devanagari first (Hindi)
    if _DEVANAGARI_RE.search(text):
        return "hin_Deva"
    if _ARABIC_RE.search(text):
        return "urd_Arab"
    if text.isascii() and text.strip():
        return "eng_Latn"
    return None


def known_language(language: Optional[str]) -> bool:
    return bool(language) and language in SUPPORTED_LANGUAGES


def match_language(chunk: dict, language: Optional[str]) -> bool:
    """Whether a chunk matches ``language`` (True for unknown/missing language).

    The chunk's explicit ``language`` field is authoritative. Only when that
    field is absent do we fall back to the record-level ``source_lang`` /
    ``target_lang`` pair.
    """
    if not language:
        return True
    lang_field = chunk.get("language")
    if lang_field:
        return str(lang_field) == language
    return chunk.get("source_lang") == language or chunk.get("target_lang") == language


def filter_results(
    results: Iterable[RetrievalResult],
    language: Optional[str],
) -> list[RetrievalResult]:
    """Keep results matching ``language``.

    An unknown / unsupported language leaves the list untouched, so the
    corpus is never accidentally eliminated.
    """
    if not known_language(language):
        return list(results)
    filtered = [r for r in results if match_language(r.metadata, language)]
    return filtered
