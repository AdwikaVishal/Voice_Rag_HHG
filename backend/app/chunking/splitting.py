"""Text splitting primitives shared by all chunking strategies.

Everything here is deterministic and operates on Unicode-safe boundaries so
that Indic scripts (Devanagari, Bengali, Gujarati, Arabic) are never
corrupted or split mid-script.

Sentence boundary heuristic (documented, deterministic — not a trained
splitter):

* Split after ``. ! ? । ۔ ؟ …`` (Latin, Devanagari danda, Arabic full stop,
  Arabic question mark, ellipsis) when followed by whitespace or end of line.
* Do NOT split after a known abbreviation (``Mr.``, ``e.g.``, ``etc.``), a
  single-letter initial (``A. Smith``) or a dotted acronym (``U.S.``).
* Decimals (``3.5``) are never treated as boundaries because the period is
  not followed by whitespace.
"""

from __future__ import annotations

import re

from .tokenizer import count_tokens, word_spans

# Paragraph boundary: one or more line breaks (with surrounding whitespace).
_PARAGRAPH_RE = re.compile(r"\s*\n\s*")

# Characters that terminate a sentence (ASCII ".", "!", "?", Devanagari
# danda "।", Arabic full stop "۔", Arabic question mark "؟", ellipsis "…").
_SENTENCE_END = ".!?।۔؟…"

# Quotes / brackets that may immediately follow a sentence terminator.
_END_CLOSERS = {'"', "'", "\u201d", "\u2019", "\u201c", "\u2018", ")", "]", "}"}

# Common abbreviations after which a "." is NOT a sentence boundary.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
        "etc", "cf", "al", "no", "vol", "fig", "sec", "dept", "inc",
        "ltd", "co", "corp", "univ", "est", "approx", "jan", "feb",
        "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
        "dec", "p", "pp", "ch", "ex", "rev", "gen", "col", "lt", "sgt",
        "cap", "gov", "min", "max", "avg", "e.g", "i.e", "u.s", "u.k",
        "a.m", "p.m", "vs", "eq", "fig", "hr", "sec", "min", "tel",
    }
)

# Dotted acronyms such as "U.S.", "U.S.A.", "a.k.a.".
_ACRONYM_RE = re.compile(r"(?:[a-z]\.){2,}$", re.IGNORECASE)


def split_paragraphs(text: str) -> list[str]:
    """Split ``text`` on line breaks into non-empty paragraphs."""
    if not text:
        return []
    return [part for part in _PARAGRAPH_RE.split(text) if part and part.strip()]


def _prev_letter_run(text: str, index: int) -> str:
    """Lowercased run of letters immediately before ``text[index]``."""
    i = index - 1
    while i >= 0 and text[i] in _END_CLOSERS:
        i -= 1
    end = i + 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    return text[i + 1 : end].lower()


def _prev_is_acronym(text: str, index: int) -> bool:
    """True if a dotted acronym like ``U.S.`` ends at ``index``."""
    i = index
    while i >= 0 and text[i] in _END_CLOSERS:
        i -= 1
    if i < 0 or text[i] != ".":
        return False
    window = text[max(0, i - 12) : i + 1]
    return bool(_ACRONYM_RE.search(window))


def _should_split_at(text: str, index: int) -> bool:
    """Decide whether the sentence terminator at ``index`` ends a sentence.

    Note: a decimal such as ``3.5`` never reaches this helper — the caller
    only considers a boundary when the terminator is followed by whitespace,
    so ``3.5 million`` (digit after the period) is never split.
    """
    if text[index] == ".":
        word = _prev_letter_run(text, index)
        if len(word) == 1:
            return False  # single-letter initial, e.g. "A. Smith"
        if word in _ABBREVIATIONS:
            return False
        if _prev_is_acronym(text, index):
            return False
    return True  # "!", "?", danda, Arabic full stop, ellipsis always split


def _split_line(text: str) -> list[str]:
    """Split one line (no newlines) into sentences using the heuristic above."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _SENTENCE_END:
            j = i + 1
            while j < n and text[j] in _SENTENCE_END:
                j += 1
            end = j
            while end < n and text[end] in _END_CLOSERS:
                end += 1
            if end >= n or text[end].isspace():
                if _should_split_at(text, i):
                    sentence = text[start:end].strip()
                    if sentence:
                        sentences.append(sentence)
                    start = end
                    i = end
                    continue
            i = j
            continue
        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, honouring paragraph line breaks."""
    if not text:
        return []
    sentences: list[str] = []
    for part in _PARAGRAPH_RE.split(text):
        sentences.extend(_split_line(part))
    return [s for s in sentences if s]


def split_by_tokens(text: str, chunk_size: int, overlap: float = 0.0) -> list[str]:
    """Fixed-size chunking at token boundaries with optional overlap.

    * ``chunk_size``: maximum tokens per chunk (a chunk never exceeds it).
    * ``overlap``:   fraction of ``chunk_size`` shared between neighbours.
      It is clamped so the stride is always >= 1 token, which guarantees
      termination even for degenerate configurations (e.g. overlap >= 1).
    * Order is preserved.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    text = text.strip()
    if not text:
        return []
    spans = word_spans(text)
    if not spans:
        return [text]

    overlap_tokens = 0
    if overlap > 0:
        overlap_tokens = int(round(chunk_size * overlap))
        overlap_tokens = max(0, min(overlap_tokens, chunk_size - 1))
    step = chunk_size - overlap_tokens
    if step < 1:
        step = 1

    chunks: list[str] = []
    start = 0
    total = len(spans)
    while start < total:
        end = min(start + chunk_size, total)
        piece = text[spans[start][0] : spans[end - 1][1]].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        start += step
    return chunks
