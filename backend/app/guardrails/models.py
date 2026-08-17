"""Typed models for the input guardrail (Segment 4B)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class GuardrailResult(BaseModel):
    """Deterministic verdict on a candidate input query.

    ``allowed`` is False when the input should not reach retrieval. When
    rejected, ``reason`` explains why (one of ``empty_input``,
    ``whitespace_only``, ``too_long``, ``unsupported_language``,
    ``prompt_injection``). ``normalized_text`` is the cleaned query that would
    be passed to retrieval; ``language`` is the resolved language in script
    form (``eng_Latn`` / ``urd_Arab``) or ``None`` when unknown.
    """

    allowed: bool
    reason: Optional[str] = None
    normalized_text: str
    language: Optional[str] = None
