"""Input guardrail (Segment 4B): deterministic pre-retrieval query checks.

Rejects clearly invalid / unsafe / non-useful input (empty, whitespace-only,
over-long, unsupported-language assertions, obvious prompt injection) before
RAG retrieval. No LLM is used — every rule is a simple, explainable,
unit-testable check.
"""

from __future__ import annotations

from .input_guardrail import (
    SUPPORTED_SCRIPT_LANGUAGES,
    SUPPORTED_STT_LANGUAGES,
    InputGuardrail,
    normalize_language,
    stt_language_code,
)
from .models import GuardrailResult

__all__ = [
    "InputGuardrail",
    "GuardrailResult",
    "normalize_language",
    "stt_language_code",
    "SUPPORTED_STT_LANGUAGES",
    "SUPPORTED_SCRIPT_LANGUAGES",
]
