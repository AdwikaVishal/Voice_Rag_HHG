"""Voice/text query pipeline (Segments 4B + 4C + 4D + 4 + 5 + 5.1): input
-> guardrail -> retrieval -> answerability gate -> answer-support verification
-> router -> answer -> TTS.

Reuses the existing lazy singletons (:class:`STTService`,
:class:`ProductionRetriever`, :class:`LLMService`, :class:`TTSService`, the
Segment 4 answerability evaluator, the Segment 5 fallback router /
general-knowledge provider, and the Segment 5.1 answer-support verifier) so the
Whisper model, embedding model, FAISS index, BM25 index, the LLM backend and
the TTS client are each used once per process. Components can be injected for
tests; nothing is injected in production, so every expensive component stays
cached and reused.

Segment 4: the answerability gate sits between retrieval and the LLM and
decides whether the retrieved context can answer the query at all.

Segment 5: the router picks the answer route from that verdict:

* RAG_GROUNDED / RAG_UNCERTAIN -> grounded RAG answer from the supporting
  chunks (source ``rag`` / ``clarification``).
* GENERAL_KNOWLEDGE           -> the user query is answered from general
  knowledge. Irrelevant retrieved chunks are NEVER mixed into the fallback —
  the general provider receives only the user question.
* ABSTAIN                     -> the Segment 4 pure abstention (when the
  general-knowledge fallback is disabled).

Segment 5.1: answer-support verification. Retrieval relevance is not answer
support — a strongly similar passage may not actually answer the exact
question. When the gate verdict is ANSWERABLE (below the verify ceiling), the
supporting evidence is re-checked by the verifier before RAG generation; if the
verifier rejects it, the route is downgraded to general knowledge so the RAG
LLM never generates from evidence that does not support the answer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ..config import ANSWER_SUPPORT_VERIFY_CEILING, DEFAULT_TOP_K, GROUNDING_VERIFY_ENABLED
from ..guardrails import GuardrailResult, InputGuardrail, stt_language_code
from ..llm import ABSTENTION_EN, ABSTENTION_UR, LLMProviderError, LLMService, get_llm_service
from ..rag import (
    AnswerSupportVerifier,
    AnswerabilityEvaluator,
    AnswerabilityStatus,
    FallbackRouter,
    GeneralKnowledgeProvider,
    Route,
    api_status,
    get_answer_support_verifier,
    get_answerability_evaluator,
    get_fallback_router,
    get_general_knowledge_provider,
    source_for_route,
)
from ..rag.grounding import GroundingVerifier, get_grounding_verifier

_UNSET = object()  # sentinel for "not explicitly provided"
from ..retrieval.production import ProductionRetriever, get_production_retriever
from ..schemas import RetrievedChunk, SearchResponse
from ..stt import STTService, TranscriptionResult, get_stt_service
from ..tts import TTSResponse, TTSError, TTSService, get_tts_service
from .models import GenerationInfo, GuardrailInfo, SourceInfo, Timings, VoiceQueryResponse

logger = logging.getLogger("pipeline")

SOURCE_EXCERPT_MAX = 140


class PipelineStageError(RuntimeError):
    """A named pipeline stage failed (``stage`` in ``stt`` | ``retrieval`` | ``llm``).

    Subclasses :class:`RuntimeError` so callers that catch generic runtime
    failures keep working; the FastAPI layer uses ``stage`` to produce a
    stage-specific, user-safe error response.
    """

    def __init__(self, stage: str, message: str = "") -> None:
        super().__init__(message or stage)
        self.stage = stage


def _source_excerpt(text: str, max_chars: int = SOURCE_EXCERPT_MAX) -> str:
    """Short, stable excerpt for the source-transparency projection."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _model_name(llm) -> str:
    """Name of the LLM service's model, or an empty string for injected mocks."""
    model = getattr(llm, "model", "")
    return model if isinstance(model, str) else ""


def _build_sources(results, excerpt_max: int = SOURCE_EXCERPT_MAX) -> list[SourceInfo]:
    """Source-transparency projection for a list of results.

    Only stable ids, relevance scores, language and a short excerpt — never the
    internal metadata object.
    """
    return [
        SourceInfo(
            id=r.chunk_id,
            score=round(float(r.score), 5),
            language=str((r.metadata or {}).get("language") or "").strip() or None,
            excerpt=_source_excerpt(r.text, excerpt_max),
        )
        for r in results
        if (r.text or "").strip()
    ]


class QueryPipeline:
    """Input -> guardrail -> hybrid retrieval -> answerability gate -> router
    -> (grounded RAG LLM | general knowledge | abstention) -> optional TTS."""

    def __init__(
        self,
        stt: Optional[STTService] = None,
        guardrail: Optional[InputGuardrail] = None,
        retriever: Optional[ProductionRetriever] = None,
        llm: Optional[LLMService] = None,
        tts: Optional[TTSService] = None,
        evaluator: Optional[AnswerabilityEvaluator] = None,
        router: Optional[FallbackRouter] = None,
        general: Optional[GeneralKnowledgeProvider] = None,
        verifier: Optional[AnswerSupportVerifier] = None,
        grounding_verifier=_UNSET,
    ) -> None:
        # Lazily resolve to the process-wide singletons when not injected.
        self.stt = stt if stt is not None else get_stt_service()
        self.guardrail = guardrail if guardrail is not None else InputGuardrail()
        self.retriever = retriever if retriever is not None else get_production_retriever()
        self.llm = llm if llm is not None else get_llm_service()
        # Optional by design: unit tests run without TTS (no network/audio).
        self.tts = tts
        # Deterministic answerability gate (Segment 4), between retrieval and LLM.
        self.evaluator = evaluator if evaluator is not None else get_answerability_evaluator()
        # Segment 5 routing: verdict -> RAG / general-knowledge / abstain.
        self.router = router if router is not None else get_fallback_router()
        self.general = general if general is not None else get_general_knowledge_provider()
        # Segment 5.1 answer-support verification (enforced by default in
        # production; tests inject fakes).
        self.verifier = verifier if verifier is not None else get_answer_support_verifier()
        # Segment 6 post-generation grounding verifier.
        # When _UNSET (not explicitly provided), use the singleton in production.
        # When None (explicitly disabled), skip grounding verification.
        if grounding_verifier is _UNSET:
            self.grounding_verifier: Optional[GroundingVerifier] = get_grounding_verifier()
        else:
            self.grounding_verifier = grounding_verifier

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #
    def process_audio(
        self,
        audio_path: str | Path,
        language_hint: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        with_tts: bool = True,
    ) -> VoiceQueryResponse:
        """Run the full audio -> text -> guardrail -> retrieval -> gate ->
        router -> answer -> TTS flow.

        1. Transcribe with the singleton STT service (language hint used only
           when it resolves to a supported code; otherwise auto-detection).
        2. Guardrail the transcript.
        3. On rejection return immediately — retrieval and generation NEVER run.
        4. Otherwise the shared text/answer flow (retrieval -> gate -> route)
           takes over (see :meth:`_answer_query`).
        """
        total_start = time.perf_counter()

        # 1. STT
        stt_start = time.perf_counter()
        stt_language = stt_language_code(language_hint)
        try:
            transcription: TranscriptionResult = self.stt.transcribe(
                audio_path, language=stt_language
            )
        except ValueError:
            raise  # decode failures stay ValueError (maps to 422 upstream)
        except Exception as exc:
            logger.error("STT stage failed for %r: %s", audio_path, exc)
            raise PipelineStageError("stt", "speech-to-text failed") from exc
        stt_ms = (time.perf_counter() - stt_start) * 1000.0
        transcript = transcription.text

        # 2. Resolve the language asserted to the guardrail:
        #    - an explicit hint is asserted verbatim (unsupported -> rejected)
        #    - otherwise the STT language is asserted only when it maps onto
        #      the corpus (en/ur); any other detected code (e.g. `hi`) falls
        #      back to script-based detection and is never blocked.
        if language_hint:
            guardrail_language: Optional[str] = language_hint
        elif transcription.language in ("en", "ur"):
            guardrail_language = transcription.language
        else:
            guardrail_language = None

        # 3. Guardrail
        guardrail_start = time.perf_counter()
        guardrail = self.guardrail.check(transcript, language=guardrail_language)
        guardrail_ms = (time.perf_counter() - guardrail_start) * 1000.0

        if not guardrail.allowed:
            return self._rejection_response(transcript, guardrail, stt_ms, guardrail_ms, total_start)

        return self._answer_query(
            guardrail.normalized_text,
            guardrail.language,
            top_k,
            transcript,
            stt_ms,
            guardrail_ms,
            total_start,
            with_tts,
        )

    def process_text(
        self,
        text: str,
        language_hint: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        with_tts: bool = True,
    ) -> VoiceQueryResponse:
        """Run the full text query flow (no STT): guardrail -> retrieval ->
        answerability gate -> router -> answer -> optional TTS.

        Mirrors :meth:`process_audio` minus speech-to-text, so the same
        routing / fallback behavior is available from a plain-text request.
        """
        total_start = time.perf_counter()

        guardrail_start = time.perf_counter()
        guardrail = self.guardrail.check(text, language=language_hint or None)
        guardrail_ms = (time.perf_counter() - guardrail_start) * 1000.0

        if not guardrail.allowed:
            return self._rejection_response(text, guardrail, 0.0, guardrail_ms, total_start)

        return self._answer_query(
            guardrail.normalized_text,
            guardrail.language,
            top_k,
            text,
            0.0,
            guardrail_ms,
            total_start,
            with_tts,
        )

    # ------------------------------------------------------------------ #
    # Shared core
    # ------------------------------------------------------------------ #
    def _rejection_response(
        self,
        transcript: str,
        guardrail: GuardrailResult,
        stt_ms: float,
        guardrail_ms: float,
        total_start: float,
    ) -> VoiceQueryResponse:
        logger.info("Guardrail rejected query (reason=%s)", guardrail.reason)
        return VoiceQueryResponse(
            transcript=transcript,
            language=guardrail.language,
            guardrail=GuardrailInfo(allowed=guardrail.allowed, reason=guardrail.reason),
            retrieval=None,
            generation=None,
            tts=None,
            timings=Timings(
                stt_ms=round(stt_ms, 2),
                guardrail_ms=round(guardrail_ms, 2),
                retrieval_ms=0.0,
                llm_ms=0.0,
                tts_ms=0.0,
                total_ms=round((time.perf_counter() - total_start) * 1000.0, 2),
            ),
        )

    def _answer_query(
        self,
        query: str,
        language: Optional[str],
        top_k: int,
        transcript: str,
        stt_ms: float,
        guardrail_ms: float,
        total_start: float,
        with_tts: bool,
    ) -> VoiceQueryResponse:
        """Shared post-guardrail flow: retrieval -> answerability gate ->
        router -> (grounded RAG LLM | general knowledge | abstention) -> TTS.
        """
        # 4. Retrieval (language is already in script form: eng_Latn / urd_Arab).
        retrieval_start = time.perf_counter()
        try:
            results = self.retriever.search(
                query, top_k=top_k, language=language
            )
        except Exception as exc:
            logger.error("Retrieval stage failed for %r: %s", query, exc)
            raise PipelineStageError("retrieval", "retrieval failed") from exc
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        retrieval = SearchResponse(
            query=query,
            detected_language=language,
            top_k=top_k,
            strategy=self.retriever.strategy,
            index_chunks=self.retriever.chunk_count,
            latency_ms=round(retrieval_ms, 2),
            results=[RetrievedChunk.from_result(r) for r in results],
        )

        # 4b. Answerability gate (Segment 4): decide whether the retrieved
        # context can answer the query at all. The verdict (not the raw
        # retrieval) determines the route below.
        decision = self.evaluator.evaluate(query, language=language, results=results)
        logger.info(
            "Answerability gate: status=%s confidence=%.4f reason=%s evidence=%d "
            "supporting=%d for query %r",
            decision.status,
            decision.confidence,
            decision.reason,
            decision.evidence_count,
            len(decision.supporting_chunk_ids),
            query,
        )

        # 5. Route (Segment 5): the gate verdict picks the answer source.
        selected_route = self.router.route(decision)
        source = source_for_route(selected_route)

        # The evidence the RAG generator WOULD use (only chunks the gate marked
        # as supporting the query's content terms). Hoisted here so both the
        # verifier and the RAG branch see the same list.
        supporting = [
            r for r in results if r.chunk_id in set(decision.supporting_chunk_ids)
        ]

        # 5a. Answer-support verification (Segment 5.1). Retrieval relevance is
        # NOT answer support: a strongly similar passage may not actually
        # answer the exact question. When the gate says ANSWERABLE (but below
        # the verify ceiling) the evidence is re-checked before RAG generation;
        # a rejected verdict downgrades the route to general knowledge so the
        # RAG LLM never generates from unsupported evidence.
        rejected_by_verifier = False
        verify_ms = 0.0
        if (
            selected_route == Route.RAG_GROUNDED
            and self.verifier is not None
            and getattr(self.verifier, "enabled", True)
            and decision.confidence < ANSWER_SUPPORT_VERIFY_CEILING
        ):
            verify_start = time.perf_counter()
            try:
                verdict = self.verifier.verify(
                    query, [r.text for r in supporting], language=language
                )
            except LLMProviderError:
                raise  # typed provider failure maps to 502 upstream
            except Exception as exc:
                logger.error("Answer-support verification failed for %r: %s", query, exc)
                raise PipelineStageError("llm", "answer-support verification failed") from exc
            verify_ms = (time.perf_counter() - verify_start) * 1000.0
            logger.info(
                "Answer-support verification: supports=%s confidence=%.3f reason=%r "
                "verify_ms=%.1f evidence=%d for query %r",
                verdict.supports_answer,
                verdict.confidence,
                verdict.reason,
                verify_ms,
                len(supporting),
                query,
            )
            if not verdict.supports_answer:
                selected_route = Route.GENERAL_KNOWLEDGE
                source = source_for_route(selected_route)
                rejected_by_verifier = True

        grounding_ms = 0.0

        if selected_route in (Route.RAG_GROUNDED, Route.RAG_UNCERTAIN):
            # Source transparency: only the supporting chunks are cited.
            sources = _build_sources(supporting)
            evidence_texts = [r.text for r in supporting]

            llm_start = time.perf_counter()
            try:
                llm_response = self.llm.generate(query, evidence_texts, language=language)
            except LLMProviderError:
                raise
            except Exception as exc:
                logger.error("LLM stage failed for %r: %s", query, exc)
                raise PipelineStageError("llm", "llm generation failed") from exc
            llm_ms = (time.perf_counter() - llm_start) * 1000.0

            # Segment 6: post-generation grounding verification.
            final_answer = llm_response.answer
            final_status = api_status(decision.status)
            final_source = source
            final_grounded = llm_response.grounded
            final_abstained = llm_response.abstained

            if (
                GROUNDING_VERIFY_ENABLED
                and self.grounding_verifier is not None
                and not llm_response.abstained
                and (llm_response.answer or "").strip()
            ):
                grounding_start = time.perf_counter()
                try:
                    gv_result = self.grounding_verifier.verify(
                        query, llm_response.answer, evidence_texts, language=language
                    )
                except Exception as exc:
                    logger.error("GroundingVerifier failed for %r: %s", query, exc)
                    # Safe fallback: treat as ungrounded.
                    from ..rag.grounding import GroundingResult
                    gv_result = GroundingResult(
                        grounded=False, confidence=0.0, reason="verifier_error"
                    )
                grounding_ms = (time.perf_counter() - grounding_start) * 1000.0

                logger.info(
                    "GroundingVerifier: grounded=%s confidence=%.3f reason=%r "
                    "unsupported=%d grounding_ms=%.1f for query %r",
                    gv_result.grounded,
                    gv_result.confidence,
                    gv_result.reason,
                    len(gv_result.unsupported_claims),
                    grounding_ms,
                    query,
                )

                if not gv_result.grounded:
                    # Attempt one constrained regeneration.
                    logger.info("Grounding failed — attempting one regeneration for %r", query)
                    llm_start2 = time.perf_counter()
                    try:
                        llm_response2 = self.llm.generate(
                            query, evidence_texts, language=language
                        )
                    except Exception as exc:
                        logger.error("Regeneration failed for %r: %s", query, exc)
                        llm_response2 = None
                    llm_ms += (time.perf_counter() - llm_start2) * 1000.0

                    if llm_response2 and not llm_response2.abstained and (llm_response2.answer or "").strip():
                        grounding_start2 = time.perf_counter()
                        try:
                            gv_result2 = self.grounding_verifier.verify(
                                query, llm_response2.answer, evidence_texts, language=language
                            )
                        except Exception:
                            from ..rag.grounding import GroundingResult
                            gv_result2 = GroundingResult(
                                grounded=False, confidence=0.0, reason="verifier_error"
                            )
                        grounding_ms += (time.perf_counter() - grounding_start2) * 1000.0

                        if gv_result2.grounded:
                            final_answer = llm_response2.answer
                            final_grounded = llm_response2.grounded
                            final_abstained = llm_response2.abstained
                        else:
                            # Both attempts failed — controlled abstention.
                            abstention_msg = (
                                ABSTENTION_UR if language == "urd_Arab"
                                else "I don't have enough reliable information to answer that."
                            )
                            final_answer = abstention_msg
                            final_status = "insufficient_evidence"
                            final_source = "abstained"
                            final_grounded = False
                            final_abstained = True
                    else:
                        abstention_msg = (
                            ABSTENTION_UR if language == "urd_Arab"
                            else "I don't have enough reliable information to answer that."
                        )
                        final_answer = abstention_msg
                        final_status = "insufficient_evidence"
                        final_source = "abstained"
                        final_grounded = False
                        final_abstained = True

            generation = GenerationInfo(
                answer=final_answer,
                model=llm_response.model,
                language=language,
                grounded=final_grounded,
                context_count=llm_response.context_count,
                latency_ms=round(llm_ms, 2),
                abstained=final_abstained,
                sources=sources,
                usage=llm_response.usage,
                status=final_status,
                source=final_source,
                confidence=round(decision.confidence, 4),
                reason=decision.reason,
                evidence_count=decision.evidence_count,
                best_score=round(decision.best_score, 6),
                supporting_chunk_ids=list(decision.supporting_chunk_ids),
            )
        elif selected_route == Route.GENERAL_KNOWLEDGE:
            # The fallback answers the USER QUERY only — no retrieved chunk is
            # ever passed to the general-knowledge provider.
            llm_start = time.perf_counter()
            try:
                gk_response = self.general.answer(query, language=language)
            except LLMProviderError:
                raise  # typed provider failure maps to 502 upstream
            except Exception as exc:
                logger.error("General-knowledge stage failed for %r: %s", query, exc)
                raise PipelineStageError("llm", "general-knowledge generation failed") from exc
            llm_ms = (time.perf_counter() - llm_start) * 1000.0

            generation = GenerationInfo(
                answer=gk_response.answer,
                model=gk_response.model,
                language=language,
                grounded=False,
                context_count=0,
                latency_ms=round(llm_ms, 2),
                abstained=gk_response.abstained,
                sources=[],
                usage=gk_response.usage,
                status="answered",
                source=source,
                confidence=None,
                reason="answer_support_rejected" if rejected_by_verifier else decision.reason,
                evidence_count=decision.evidence_count,
                best_score=round(decision.best_score, 6),
                supporting_chunk_ids=[],
            )
        else:  # Route.ABSTAIN — Segment 4 pure abstention (fallback disabled).
            llm_ms = 0.0
            abstention = ABSTENTION_UR if language == "urd_Arab" else ABSTENTION_EN
            generation = GenerationInfo(
                answer=abstention,
                model=_model_name(self.llm),
                language=language,
                grounded=False,
                context_count=0,
                latency_ms=0.0,
                abstained=True,
                sources=[],
                usage=None,
                status=api_status(decision.status),
                source=source,
                confidence=round(decision.confidence, 4),
                reason=decision.reason,
                evidence_count=decision.evidence_count,
                best_score=round(decision.best_score, 6),
                supporting_chunk_ids=[],
            )

        logger.info(
            "route query=%r language=%s gate=%s route=%s source=%s "
            "llm_ms=%.2f verify_ms=%.2f grounding_ms=%.2f rejected_by_verifier=%s "
            "evidence=%d supporting=%d",
            query,
            language,
            decision.status,
            selected_route,
            source,
            llm_ms,
            verify_ms,
            grounding_ms,
            rejected_by_verifier,
            decision.evidence_count,
            len(decision.supporting_chunk_ids),
        )

        # 6. TTS — speak the final validated answer verbatim (a grounded RAG
        # answer, a general-knowledge answer, or the explicit abstention;
        # never an empty answer).
        tts_info: Optional[TTSResponse] = None
        tts_ms = 0.0
        if (
            with_tts
            and self.tts is not None
            and generation is not None
            and generation.answer.strip()
        ):
            tts_start = time.perf_counter()
            try:
                result = self.tts.synthesize(
                    generation.answer, language=generation.language
                )
            except TTSError as exc:
                logger.error("TTS failed for query %r: %s", query, exc)
            else:
                tts_ms = (time.perf_counter() - tts_start) * 1000.0
                tts_info = TTSResponse.from_result(result)
                # The temp WAV is owned by the caller; the JSON endpoint only
                # reports metadata, so the file is removed immediately.
                Path(result.audio_path).unlink(missing_ok=True)

        return VoiceQueryResponse(
            transcript=transcript,
            language=language,
            guardrail=GuardrailInfo(allowed=True, reason=None),
            retrieval=retrieval,
            generation=generation,
            tts=tts_info,
            timings=Timings(
                stt_ms=round(stt_ms, 2),
                guardrail_ms=round(guardrail_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                llm_ms=round(llm_ms, 2),
                grounding_ms=round(grounding_ms, 2),
                tts_ms=round(tts_ms, 2),
                total_ms=round((time.perf_counter() - total_start) * 1000.0, 2),
            ),
        )
