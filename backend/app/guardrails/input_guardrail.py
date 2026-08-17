"""Lightweight deterministic input guardrail (Segment 4B).

The guardrail decides whether a query may proceed to retrieval. It is a set of
simple, explainable, unit-testable rules — no LLM, no classifiers:

1. Empty input            -> ``empty_input``
2. Whitespace-only input  -> ``whitespace_only``
3. Extremely long input   -> ``too_long``
4. Unsupported language   -> ``unsupported_language`` (only when a caller
   *asserts* a language hint that is not one of the supported languages)
5. Obvious prompt
   injection / malicious
   manipulation of the RAG -> ``prompt_injection``

Normalization is deliberately conservative: leading/trailing trim, collapse of
repeated whitespace, and nothing else. Unicode, Urdu/Indic scripts and the
semantic meaning of the query are preserved — the text is never transliterated
or ASCII-normalized.
"""

from __future__ import annotations

import re
from typing import Optional

from ..config import GR_MAX_INPUT_CHARS
from ..retrieval.filters import detect_script_language
from .models import GuardrailResult

__all__ = ["InputGuardrail", "GuardrailResult", "normalize_language", "stt_language_code"]

# Canonical script-form codes used by the retrieval layer.
ENG = "eng_Latn"
URD = "urd_Arab"

# Language aliases -> script-form code. Only English and Urdu are in the corpus.
_LANGUAGE_ALIASES: dict[str, str] = {
    "en": ENG, "eng": ENG, "english": ENG,
    "eng_latn": ENG, "eng-latn": ENG, "latin": ENG,
    "ur": URD, "urd": URD, "urdu": URD,
    "urd_arab": URD, "urd-arab": URD, "arabic": URD,
}

# Same aliases -> ISO-639-1 code for the STT language hint.
_STT_ALIASES: dict[str, str] = {
    "en": "en", "eng": "en", "english": "en", "eng_latn": "en", "eng-latn": "en",
    "ur": "ur", "urd": "ur", "urdu": "ur", "urd_arab": "ur", "urd-arab": "ur",
}

# Whisper codes that map onto the corpus languages and are safe to assert
# automatically from STT output (see pipeline language resolution).
SUPPORTED_STT_LANGUAGES = frozenset({"en", "ur"})
SUPPORTED_SCRIPT_LANGUAGES = frozenset({ENG, URD})


def normalize_language(language: Optional[str]) -> Optional[str]:
    """Resolve a language value to a script-form code, or ``None`` if unknown."""
    if not language:
        return None
    return _LANGUAGE_ALIASES.get(language.strip().lower())


def stt_language_code(language: Optional[str]) -> Optional[str]:
    """Resolve a language value to the ISO-639-1 code faster-whisper expects."""
    if not language:
        return None
    return _STT_ALIASES.get(language.strip().lower())


# Prompt-injection / RAG-manipulation trigger patterns. Every pattern is
# multi-token and semantically tied to instruction/context manipulation, so a
# normal question that merely contains a word like "ignore", "prompt" or
# "reveal" is never rejected.
_INJECTION_PATTERNS: tuple[str, ...] = (
    # ignore / disregard the instructions
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:above|prior|previous)\s+(?:instructions?|prompts?|rules|guidelines|messages?)\b",
    r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:above|prior|previous)\s+(?:instructions?|prompts?|rules|guidelines|messages?)\b",
    r"\bdo\s+not\s+follow\s+(?:your|the)\s+(?:above|prior|previous)\s+(?:instructions?|prompts?|rules|guidelines)\b",
    # reveal / dump the system prompt or instructions
    r"\b(?:reveal|show|print|display|output|leak|dump|expose|give)\s+(?:me\s+)?(?:your|the)\s+(?:system|internal|hidden|secret)\s+(?:prompt|instructions?|guidelines|rules)\b",
    r"\b(?:reveal|show|print|display|output|leak|dump|expose)\b.{0,60}\bsystem\s+prompt\b",
    r"\bsystem\s+prompt\b.{0,60}\b(?:reveal|show|print|display|output|leak|dump|expose)\b",
    # pretend to be unrestricted / jailbreak
    r"\b(?:act|behave|respond)\s+as\s+if\s+(?:you\s+)?(?:have|had|are)\s+no\s+(?:restrictions?|rules|guardrails?|limits)\b",
    r"\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|jailbroken)\b",
    r"\bbypass\s+(?:the\s+)?(?:guardrail|guardrails|safety|filter|restrictions)\b",
    r"\boverride\s+(?:your|the)\s+(?:instructions?|rules|guidelines|prompt)\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    # manipulating the retrieved context / sources
    r"\bignore\s+(?:the\s+)?(?:retrieved|search|rag|context|sources?)\b",
    r"\bdisregard\s+(?:the\s+)?(?:retrieved|search|rag|context|sources?)\b",
    r"\bfabricate\s+(?:sources|facts|citations)\b",
    r"\bmake\s+up\s+(?:sources|facts|answers|citations)\b",
)

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


class InputGuardrail:
    """Deterministic, explainable input checks before retrieval.

    ``max_chars`` caps the normalized query length; it defaults to
    :data:`app.config.GR_MAX_INPUT_CHARS`.
    """

    def __init__(self, max_chars: int = GR_MAX_INPUT_CHARS) -> None:
        self.max_chars = int(max_chars)

    def normalize(self, text: str) -> str:
        """Trim and collapse repeated whitespace; preserve Unicode/scripts."""
        return _WS_RE.sub(" ", (text or "").strip())

    def _detect(self, text: str, language: Optional[str]) -> Optional[str]:
        if language:
            return normalize_language(language)
        return detect_script_language(text)

    def check(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> GuardrailResult:
        """Return a verdict for ``text``.

        ``language`` is the caller-asserted language (from an explicit API
        hint or from STT when it is one of the supported codes). When it is
        supplied but unsupported, the input is rejected; when it is ``None``
        the query's script is used for detection.
        """
        if text is None or text == "":
            return GuardrailResult(allowed=False, reason="empty_input", normalized_text="", language=None)

        stripped = (text or "").strip()
        if stripped == "":
            return GuardrailResult(allowed=False, reason="whitespace_only", normalized_text="", language=None)

        normalized = self.normalize(stripped)
        if len(normalized) > self.max_chars:
            return GuardrailResult(
                allowed=False,
                reason="too_long",
                normalized_text=normalized,
                language=self._detect(normalized, language),
            )

        resolved = self._detect(normalized, language)
        if language and resolved is None:
            return GuardrailResult(
                allowed=False,
                reason="unsupported_language",
                normalized_text=normalized,
                language=None,
            )

        if _INJECTION_RE.search(normalized):
            return GuardrailResult(
                allowed=False,
                reason="prompt_injection",
                normalized_text=normalized,
                language=resolved,
            )

        return GuardrailResult(allowed=True, reason=None, normalized_text=normalized, language=resolved)
