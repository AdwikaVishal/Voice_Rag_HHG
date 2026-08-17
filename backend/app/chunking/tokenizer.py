"""Deterministic, language-neutral token approximation.

We do not depend on a trained subword tokenizer (e.g. tiktoken / BERT) for
this benchmark:

* MSMARCO-XI is multilingual (English + Indic/Arabic scripts), and BPE-style
  tokenizers trained on English are a poor proxy for Indic text.
* The goal of this segment is structural chunking statistics, not retrieval
  quality.

A token is therefore any maximal run of Unicode word characters (letters,
digits, underscore). This is fast, deterministic, and does not break Indic
scripts. The approximation is documented so downstream segments know that
"token count" here means "Unicode word tokens".
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Return the word tokens in ``text`` (empty string -> empty list)."""
    if not text:
        return []
    return _WORD_RE.findall(text)


def count_tokens(text: str) -> int:
    """Count Unicode word tokens in ``text``."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def word_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of every token in ``text``.

    Offsets index into the original string, so chunks can be sliced from the
    original text and keep the original whitespace / punctuation layout.
    """
    if not text:
        return []
    return [match.span() for match in _WORD_RE.finditer(text)]
