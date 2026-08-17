"""Post-generation grounding verifier (Segment 6).

After the RAG LLM generates an answer, this verifier asks the same LLM
(low temperature, structured JSON output) whether every claim in the answer
is actually supported by the supplied evidence.

It does NOT answer the question itself — it only evaluates the generated
answer against the evidence.

Return contract::

    {
        "grounded": true/false,
        "confidence": 0.0-1.0,
        "reason": "...",
        "unsupported_claims": [...]
    }

If the LLM call fails or returns unparseable output the verifier returns a
safe ``grounded=False`` result so the pipeline can recover rather than
silently returning an unverified answer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("rag.grounding")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = (
    "You are a grounding verifier for a retrieval-augmented QA system.\n\n"
    "Your ONLY job is to check whether the GENERATED ANSWER is fully supported "
    "by the EVIDENCE passages below. You must NOT answer the question yourself.\n\n"
    "Rules:\n"
    "- A claim is supported if it can be directly inferred from the evidence.\n"
    "- A claim is unsupported if it introduces facts not present in the evidence.\n"
    "- If the answer is empty or is an abstention message, return grounded=false.\n\n"
    "Respond with ONLY a JSON object — no prose, no markdown fences:\n"
    '{"grounded": true/false, "confidence": 0.0-1.0, "unsupported_claims": []}'
)


class GroundingResult(BaseModel):
    grounded: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)


class GroundingVerifier:
    """LLM-based post-generation grounding verifier.

    Reuses the existing LLM provider — no second model or client is created.
    The verifier is intentionally stateless; one instance can be shared across
    requests.
    """

    def __init__(self, llm=None) -> None:
        # ``llm`` is the LLMService (injected for tests, resolved lazily in prod).
        self._llm = llm

    @property
    def _backend(self):
        if self._llm is None:
            from ..llm import get_llm_service
            self._llm = get_llm_service()
        return self._llm

    def verify(
        self,
        query: str,
        answer: str,
        evidence: list[str],
        language: Optional[str] = None,
    ) -> GroundingResult:
        """Check whether ``answer`` is supported by ``evidence``.

        Returns a safe ``grounded=False`` result on any failure so the caller
        can recover without crashing.
        """
        answer = (answer or "").strip()
        usable = [(e or "").strip() for e in (evidence or []) if (e or "").strip()]

        if not answer:
            return GroundingResult(grounded=False, confidence=0.0, reason="empty_answer")
        if not usable:
            return GroundingResult(grounded=False, confidence=0.0, reason="no_evidence")

        evidence_block = "\n\n".join(
            f"[Evidence {i}]\n{e}" for i, e in enumerate(usable, 1)
        )
        user_msg = (
            f"QUERY: {query}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Is every claim in the GENERATED ANSWER supported by the EVIDENCE? "
            "Respond with ONLY the JSON object."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        backend = self._backend
        model = getattr(backend, "model", "")
        # Low temperature for deterministic structured output.
        temperature = 0.0
        max_tokens = 200

        start = time.perf_counter()
        try:
            result = backend.provider.chat(model, messages, temperature, max_tokens)
            raw = (result.content or "").strip()
        except Exception as exc:
            logger.error("GroundingVerifier LLM call failed: %s", exc)
            return GroundingResult(
                grounded=False, confidence=0.0, reason="verifier_error"
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.debug("GroundingVerifier LLM call %.1f ms", latency_ms)

        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> GroundingResult:
        """Extract and validate the JSON verdict from the LLM output."""
        match = _JSON_RE.search(raw)
        if not match:
            logger.warning("GroundingVerifier: no JSON in output: %r", raw[:200])
            return GroundingResult(grounded=False, confidence=0.0, reason="parse_error")
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning("GroundingVerifier: JSON parse error: %s", exc)
            return GroundingResult(grounded=False, confidence=0.0, reason="parse_error")

        grounded = bool(data.get("grounded", False))
        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5
        unsupported = [str(c) for c in (data.get("unsupported_claims") or [])]
        reason = "grounded" if grounded else ("unsupported_claims" if unsupported else "not_grounded")
        return GroundingResult(
            grounded=grounded,
            confidence=round(confidence, 4),
            reason=reason,
            unsupported_claims=unsupported,
        )


# Process-wide singleton (lazy).
_verifier: Optional[GroundingVerifier] = None


def get_grounding_verifier() -> GroundingVerifier:
    global _verifier
    if _verifier is None:
        _verifier = GroundingVerifier()
    return _verifier
