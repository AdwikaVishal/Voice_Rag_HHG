"""Deterministic RAG answerability gate (Segment 4).

Placed between retrieval and the LLM: it decides whether the retrieved context
is strong enough to answer the query at all. It never compares raw BM25, dense
or RRF scores against each other — every signal is normalized into [0, 1] on
its own scale and combined into a single confidence.

Signals considered (all deterministic, no model):

* ``answer_presence``  — what fraction of the query's content terms actually
  appear in the best chunk. This is the primary "the text contains the answer"
  signal and the decisive gate: a query whose answer terms are absent (e.g.
  "capital of France" over CDG airport chunks) is never answerable even when
  retrieval rank and scores look healthy.
* ``hybrid_rank``      — normalized rank of the best matching chunk.
* ``retrieval_agreement`` — normalized relative score fall-off between the
  top and second result, used as a proxy for dense <-> BM25 agreement: a chunk
  that both retrievers surface (RRF fusion) usually leads by a clear margin.
* ``related_chunks``   — how many other chunks are on the same topic
  (normalized content-token Jaccard), with near-duplicates de-duplicated so a
  single passage copied several times does not inflate confidence.
* ``language_match``   — whether the best chunk's language matches the query
  language (unknown languages are never penalized).

Decision policy::

    no usable chunks                 -> UNANSWERABLE_FROM_RAG  (reason no_evidence)
    answer_presence <  lexical_floor -> UNANSWERABLE_FROM_RAG  (reason no_answer_terms /
                                                                 weak_answer_overlap)
    confidence    >= high_confidence -> ANSWERABLE            (reason answer_present)
    confidence    >= low_confidence  -> UNCERTAIN             (reason partial_evidence)
    otherwise                         -> UNANSWERABLE_FROM_RAG (reason insufficient_confidence)
"""

from __future__ import annotations

import logging
from typing import Optional

from ..chunking.tokenizer import tokenize
from ..config import (
    ANSWERABILITY_HIGH_CONFIDENCE,
    ANSWERABILITY_LEXICAL_FLOOR,
    ANSWERABILITY_LOW_CONFIDENCE,
    ANSWERABILITY_TOP_K,
)
from ..retrieval.models import RetrievalResult
from .models import AnswerabilityDecision, AnswerabilityStatus

logger = logging.getLogger("rag.answerability")

# Stopword sets are deliberately local to the gate (the LLM layer keeps its own
# for the post-hoc grounding check). Content terms are what the query is asking
# about; removing them here would silently weaken the answer-presence signal.
_EN_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "at", "by", "for", "and", "or", "but", "it",
        "its", "this", "that", "these", "those", "what", "which", "who", "whom",
        "whose", "how", "why", "does", "do", "did", "done", "not", "no", "yes",
        "with", "from", "as", "about", "up", "out", "into", "over", "under",
        "between", "during", "through", "than", "then", "there", "their",
        "they", "them", "he", "she", "his", "her", "him", "we", "us", "our",
        "ours", "you", "your", "yours", "my", "me", "i", "am", "tell", "say",
        "said", "know", "please", "can", "could", "would", "should", "may",
        "might", "has", "have", "had", "also", "very", "really", "just", "so",
    }
)

_UR_STOPWORDS = frozenset(
    {
        "کا", "کی", "کے", "کو", "سے", "میں", "پر", "اور", "ہے", "ہیں",
        "تھا", "تھی", "تھے", "ایک", "اس", "یہ", "وہ", "بھی", "نہیں",
        "کیا", "کون", "کہاں", "کیوں", "جو", "نے", "لیے", "کہ", "تم",
        "آپ", "میرے", "میری", "کا", "سکتا", "سکتی", "سکتے", "کرتا",
        "کرتی", "کرتے", "رہا", "رہی", "رہے",
    }
)

# Default signal weights (sum to 1). answer_presence dominates by design.
_WEIGHTS = {
    "answer_presence": 0.45,
    "hybrid_rank": 0.15,
    "retrieval_agreement": 0.15,
    "related_chunks": 0.15,
    "language_match": 0.10,
}

# Agreement proxy when there is no second result to compare against (a single
# chunk is weak corroboration, never a strong agreement signal).
_SINGLE_RESULT_AGREEMENT = 0.2

# Content-token Jaccard thresholds.
_RELATED_MIN = 0.3
_NEAR_DUPLICATE_MIN = 0.8

_EPS = 1e-9


def _content_tokens(text: str) -> list[str]:
    """Query/chunk content terms: Unicode word tokens minus stopwords."""
    lowered = (tokenize(text or "") or [])[:]
    return [tok.casefold() for tok in lowered if tok.casefold() not in _EN_STOPWORDS and tok.casefold() not in _UR_STOPWORDS]


def _overlap_ratio(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    """Fraction of query content terms present in the chunk (0..1)."""
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    chunk_set = set(chunk_tokens)
    return len(query_set & chunk_set) / len(query_set)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class AnswerabilityEvaluator:
    """Deterministic, model-free answerability gate.

    ``evaluate`` is a pure function of ``(query, language, results)`` — it
    never touches the embedding model, the indexes or any provider.
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        lexical_floor: Optional[float] = None,
        high_confidence: Optional[float] = None,
        low_confidence: Optional[float] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.top_k = top_k if top_k is not None else ANSWERABILITY_TOP_K
        self.lexical_floor = float(lexical_floor if lexical_floor is not None else ANSWERABILITY_LEXICAL_FLOOR)
        self.high_confidence = float(high_confidence if high_confidence is not None else ANSWERABILITY_HIGH_CONFIDENCE)
        self.low_confidence = float(low_confidence if low_confidence is not None else ANSWERABILITY_LOW_CONFIDENCE)
        self.weights = dict(weights) if weights else dict(_WEIGHTS)

    # -- signal computation ----------------------------------------------
    @staticmethod
    def _agreement(results: list[RetrievalResult]) -> float:
        """Normalized dense<->BM25 agreement proxy from RRF score fall-off."""
        if len(results) < 2:
            return _SINGLE_RESULT_AGREEMENT
        top = float(results[0].score)
        second = float(results[1].score)
        gap = (top - second) / max(abs(top), _EPS)
        return min(max(gap / 0.5, 0.0), 1.0)

    @staticmethod
    def _related_chunks(results: list[RetrievalResult], best_chunk: RetrievalResult) -> float:
        """On-topic corroboration count (near-duplicates de-duplicated), in [0,1]."""
        best_tokens = set(_content_tokens(best_chunk.text))
        # The best chunk is the baseline: near-copies of it carry no new
        # evidence, so they are filtered like any other duplicate.
        counted: list[set[str]] = [best_tokens] if best_tokens else []
        for result in results:
            if result.chunk_id == best_chunk.chunk_id:
                continue
            tokens = set(_content_tokens(result.text))
            if not best_tokens or not tokens:
                continue
            if _jaccard(best_tokens, tokens) < _RELATED_MIN:
                continue
            if any(_jaccard(tokens, previous) >= _NEAR_DUPLICATE_MIN for previous in counted):
                continue
            counted.append(tokens)
        return min(len(counted) - 1, 4) / 4.0

    @staticmethod
    def _language_match(language: Optional[str], chunk: RetrievalResult) -> float:
        if not language:
            return 1.0
        chunk_language = str((chunk.metadata or {}).get("language") or "").strip() or None
        if not chunk_language:
            return 1.0
        return 1.0 if chunk_language == language else 0.0

    # -- main entry ------------------------------------------------------
    def evaluate(
        self,
        query: str,
        language: Optional[str] = None,
        results: Optional[list[RetrievalResult]] = None,
        top_k: Optional[int] = None,
    ) -> AnswerabilityDecision:
        """Decide whether ``results`` can answer ``query``.

        Returns an :class:`AnswerabilityDecision` with the three-way status,
        a normalized confidence, a reason, and the supporting chunk ids.
        """
        results = [r for r in (results or []) if r and (r.text or "").strip()]
        if not results:
            return AnswerabilityDecision(
                status=AnswerabilityStatus.UNANSWERABLE_FROM_RAG,
                confidence=0.0,
                reason="no_evidence",
                evidence_count=0,
                best_score=0.0,
                supporting_chunk_ids=[],
            )

        top = results[: top_k or self.top_k]
        best_score = float(max(float(r.score) for r in results))

        query_tokens = _content_tokens(query)

        scored = [
            (_overlap_ratio(query_tokens, _content_tokens(r.text)), r) for r in top
        ]
        answer_presence, best_chunk = max(scored, key=lambda pair: pair[0])

        evidence = [
            r for r in top
            if _overlap_ratio(query_tokens, _content_tokens(r.text)) > 0.0
        ]

        best_index = top.index(best_chunk) if best_chunk in top else 0
        best_rank = int(getattr(best_chunk, "rank", 0) or 0) or (best_index + 1)
        rank_signal = 1.0 - (max(best_rank, 1) - 1) / max(len(top), 1)

        agreement = self._agreement(top)
        related = self._related_chunks(top, best_chunk)
        language_match = self._language_match(language, best_chunk)

        confidence = (
            self.weights.get("answer_presence", 0.0) * answer_presence
            + self.weights.get("hybrid_rank", 0.0) * rank_signal
            + self.weights.get("retrieval_agreement", 0.0) * agreement
            + self.weights.get("related_chunks", 0.0) * related
            + self.weights.get("language_match", 0.0) * language_match
        )

        supporting_chunk_ids = [r.chunk_id for r in evidence]

        if answer_presence < self.lexical_floor:
            # The answer's content words are missing — the gate must NOT let
            # irrelevant chunks be used as authoritative evidence regardless of
            # how confident retrieval looks. Confidence is attenuated so a low
            # value reliably accompanies the abstention.
            confidence = confidence * answer_presence
            return AnswerabilityDecision(
                status=AnswerabilityStatus.UNANSWERABLE_FROM_RAG,
                confidence=round(confidence, 4),
                reason="no_answer_terms" if answer_presence == 0.0 else "weak_answer_overlap",
                evidence_count=len(evidence),
                best_score=round(best_score, 6),
                supporting_chunk_ids=[],
            )

        if confidence >= self.high_confidence:
            status = AnswerabilityStatus.ANSWERABLE
            reason = "answer_present"
        elif confidence >= self.low_confidence:
            status = AnswerabilityStatus.UNCERTAIN
            reason = "partial_evidence"
        else:
            status = AnswerabilityStatus.UNANSWERABLE_FROM_RAG
            reason = "insufficient_confidence"
            supporting_chunk_ids = []

        return AnswerabilityDecision(
            status=status,
            confidence=round(confidence, 4),
            reason=reason,
            evidence_count=len(evidence),
            best_score=round(best_score, 6),
            supporting_chunk_ids=supporting_chunk_ids,
        )
