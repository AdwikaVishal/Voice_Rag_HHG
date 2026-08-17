"""Answer-support verifier (Segment 5.1).

The Segment 4 answerability gate is a deterministic, score-based layer: a
retrieved passage can be strongly similar to a query without actually
answering it ("What is the capital of France?" over a passage that mentions
France and Versailles). The verifier sits between the gate and the RAG LLM and
re-checks the EVIDENCE that would be used to generate: does it DIRECTLY answer
the exact question?

Segment 5.1 originally used an LLM judge for this step, but live testing showed
the shared 3B model is not robust: it rejected the same directly-answering FAQ
passage ("Which airport is CDG? CDG is officially named Roissy Charles de
Gaulle...") when presented with realistic retrieved context, while accepting
noise. Because the user's directive prefers the simplest robust approach and
explicitly allows deterministic textual heuristics "if they are more robust",
this verifier is now deterministic:

* A passage is treated as DIRECTLY answer-supporting iff it contains the FULL
  set of the query's content tokens (a complete restatement, which makes
  wrong-entity matches essentially impossible) AND has question-answer
  structure (a question mark — the passage literally poses and answers the
  question). This is high precision: it only confirms evidence that is
  unambiguous.
* Evidence that is not confirmed is rejected and the pipeline routes to the
  general-knowledge fallback. A wrong RAG answer is worse than falling back, so
  the step is deliberately precision-biased (over-rejection degrades
  gracefully; under-rejection reproduces the Versailles false positive).

Behavior:

* Runs only when the gate verdict is ANSWERABLE and the gate confidence is
  below :data:`ANSWER_SUPPORT_VERIFY_CEILING` (a clearly-grounded query skips
  verification entirely).
* No LLM call, no network: the check is pure token matching over the supporting
  evidence, so it adds no model latency and is fully deterministic.
* The verdict is a structured :class:`SupportVerdict` (``supports_answer``,
  ``confidence``, ``reason``) so the pipeline contract is unchanged.

The final RAG answer is therefore never generated from evidence the support
verifier rejected.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from .answerability import _content_tokens

logger = logging.getLogger("rag.verifier")

# Question marks in ASCII and Arabic script — the signature of a passage that
# poses the question itself (FAQ-style, directly answerable evidence).
_QUESTION_MARKS = frozenset({"?", "\u061f", "\u2047", "\u2048"})

# Urdu surface variants where a final alef/heh alternates (اڈا / اڈہ) and a
# final alef/yeh alternates (ے / ی). Tokens are compared after folding these,
# so "ہوائی اڈا" and "ہوائی اڈہ" match.
_URDU_FOLD = {
    "\u0627": "\u06cc",  # alef -> yeh (final-vowel neutralization)
    "\u06c1": "\u06cc",  # heh -> yeh
    "\u06be": "\u06cc",  # heh-doachashmee -> yeh
    "\u06cc": "\u06cc",  # yeh -> yeh
    "\u06d2": "\u06cc",  # alef-maksura -> yeh
    "\u06d3": "\u06cc",  # yeh-barree -> yeh
}


def _fold_token(token: str) -> str:
    """Normalize a single token for evidence matching (Urdu surface variants)."""
    if not token:
        return token
    head, last = token[:-1], token[-1]
    folded = _URDU_FOLD.get(last, last)
    return head + folded


def _question_answer_passage(passage: str) -> bool:
    """True if the passage itself poses a question (Q&A/FAQ structure)."""
    return any(mark in passage for mark in _QUESTION_MARKS)


class SupportVerdict(BaseModel):
    """Structured verdict of one answer-support check (Segment 5.1).

    ``supports_answer`` is the machine decision; ``confidence`` the verifier's
    confidence in its own decision in [0, 1]; ``reason`` a short human-readable
    justification used for logging.
    """

    supports_answer: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = Field("", description="Short justification (logging only).")


class AnswerSupportVerifier:
    """Deterministically confirms the supporting RAG evidence answers the query.

    ``verify(query, context, language)`` returns a structured
    :class:`SupportVerdict`. The check is pure token matching over the
    supporting evidence — no LLM call, no network, no provider dependency.
    """

    #: Confidence when a directly-answer-supporting passage is found.
    _SUPPORT_CONFIDENCE = 0.95

    def __init__(
        self,
        llm=None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        # These parameters are accepted for API/signature compatibility with the
        # original LLM-based implementation but are unused by the deterministic
        # check. ``llm`` is kept so injected fakes still work in tests.
        self._llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def enabled(self) -> bool:
        """Master switch; the pipeline skips verification when false."""
        from ..config import ANSWER_SUPPORT_VERIFY_ENABLED

        return bool(ANSWER_SUPPORT_VERIFY_ENABLED)

    @property
    def provider_name(self) -> str:
        """Backend name: this deterministic check has no provider."""
        return "deterministic"

    @staticmethod
    def _max_coverage(query_tokens: list[str], passages: list[str]) -> float:
        """Highest query-content coverage reached by any passage (0..1)."""
        query_set = {_fold_token(t) for t in query_tokens}
        if not query_set:
            return 0.0
        best = 0.0
        for passage in passages:
            tokens = {_fold_token(t) for t in _content_tokens(passage)}
            if not tokens:
                continue
            covered = len(query_set & tokens) / len(query_set)
            if covered > best:
                best = covered
        return best

    def verify(
        self,
        query: str,
        context: list[str],
        language: Optional[str] = None,
    ) -> SupportVerdict:
        """Judge whether ``context`` directly answers ``query``.

        ``context`` is the exact evidence the RAG generator would use
        (supporting chunks). Empty context is rejected without any check.
        """
        usable = [(c or "").strip() for c in (context or []) if (c or "").strip()]
        if not usable:
            return SupportVerdict(supports_answer=False, confidence=0.0, reason="no_context")

        query_tokens = _content_tokens(query)
        query_set = {_fold_token(t) for t in query_tokens}
        if not query_set:
            return SupportVerdict(supports_answer=False, confidence=0.0, reason="no_query_terms")

        direct = None
        for passage in usable:
            passage_set = {_fold_token(t) for t in _content_tokens(passage)}
            if query_set <= passage_set and _question_answer_passage(passage):
                direct = passage
                break

        if direct is not None:
            verdict = SupportVerdict(
                supports_answer=True,
                confidence=self._SUPPORT_CONFIDENCE,
                reason="answer_supported",
            )
        elif any(query_set <= {_fold_token(t) for t in _content_tokens(p)} for p in usable):
            # Every content term is present but no passage poses the question —
            # a partial/tangential match (e.g. the Versailles narrative).
            coverage = self._max_coverage(query_tokens, usable)
            verdict = SupportVerdict(
                supports_answer=False,
                confidence=round(min(max(0.5 + 0.4 * (1.0 - coverage), 0.1), 0.9), 3),
                reason="coverage_without_qa_structure",
            )
        else:
            coverage = self._max_coverage(query_tokens, usable)
            verdict = SupportVerdict(
                supports_answer=False,
                confidence=round(min(max(0.5 + 0.4 * (1.0 - coverage), 0.1), 0.9), 3),
                reason="no_full_answer_coverage",
            )

        logger.info(
            "Answer-support verdict: supports=%s confidence=%.3f reason=%r "
            "evidence=%d for query %r",
            verdict.supports_answer,
            verdict.confidence,
            verdict.reason,
            len(usable),
            query,
        )
        return verdict
