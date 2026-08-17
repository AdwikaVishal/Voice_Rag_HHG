"""Segment 4 — tests for the RAG answerability gate.

Covers the full decision space (ANSWERABLE / UNCERTAIN / UNANSWERABLE_FROM_RAG),
the API status mapping, the deterministic signal calibration, and pipeline
integration (gate runs between retrieval and the LLM; the LLM is never called
for an UNANSWERABLE_FROM_RAG verdict). Everything is deterministic — no real
embedding model, index or LLM provider is ever loaded.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.guardrails import InputGuardrail  # noqa: E402
from app.llm import ABSTENTION_EN, ABSTENTION_UR, LLMResponse  # noqa: E402
from app.pipeline import QueryPipeline  # noqa: E402
from app.rag import (  # noqa: E402
    AnswerabilityDecision,
    AnswerabilityEvaluator,
    AnswerabilityStatus,
    FallbackRouter,
    SupportVerdict,
    api_status,
)
from app.retrieval.models import RetrievalResult  # noqa: E402
from app.stt.models import TranscriptionResult  # noqa: E402

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

CDG_QUERY = "Which airport serves Paris and is abbreviated as CDG?"
CDG_PARAPHRASE = "What airport near Paris has the code CDG?"
CDG_URDU_QUERY = "سی ڈی جی ہوائی اڈا کیا ہے؟"
CAPITAL_FRANCE = "What is the capital of France?"
TELEPHONE_QUERY = "Who invented the telephone?"

CDG_CHUNKS = [
    ("cdg1", "Charles de Gaulle Airport (CDG) serves Paris, France as the "
              "main international airport.", "eng_Latn", 0.0320),
    ("cdg2", "CDG is located about 23 km north-east of Paris.", "eng_Latn", 0.0164),
    ("cdg3", "Roissy Charles de Gaulle airport is abbreviated as CDG.", "eng_Latn", 0.0161),
]

CDG_URDU_CHUNKS = [
    ("urd1", "سی ڈی جی ہوائی اڈا، جسے رائسی ہوائی اڈا بھی کہا جاتا ہے، پیرس کی خدمت کرتا ہے۔",
     "urd_Arab", 0.0320),
    ("urd2", "سی ڈی جی پیرس کے قریب ایک بڑا بین الاقوامی ہوائی اڈا ہے۔",
     "urd_Arab", 0.0164),
]

# Telephone-network chunks: they mention "telephone" but never "invented",
# mirroring the real corpus where the inventor question is a coverage gap.
TELEPHONE_CHUNKS = [
    ("t1", "The telephone network routes calls through the CDG switching centre.",
     "eng_Latn", 0.0300),
    ("t2", "Telephone cables carry voice signals across the country.",
     "eng_Latn", 0.0150),
]

IRRELEVANT_CHUNKS = [
    ("i1", "Bears hibernate during the winter months.", "eng_Latn", 0.0300),
    ("i2", "Cars need fuel to run.", "eng_Latn", 0.0290),
    ("i3", "The moon orbits the Earth.", "eng_Latn", 0.0280),
]

EXACT_ANSWER_CHUNKS = [
    ("e1", "Charles de Gaulle Airport (CDG), also known as Roissy airport, "
           "serves Paris, France as the main international airport.",
     "eng_Latn", 0.0320),
    ("e2", "CDG is the airport code for Charles de Gaulle Airport near Paris.",
     "eng_Latn", 0.0164),
    ("e3", "The abbreviation CDG stands for Roissy Charles de Gaulle Airport.",
     "eng_Latn", 0.0161),
]


def make_result(chunk_id, text, language="eng_Latn", score=0.95, rank=1):
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={
            "record_id": f"rec-{chunk_id}",
            "query_id": 1,
            "passage_index": 0,
            "language": language,
            "chunk_position": 0,
            "total_chunks": 1,
        },
        rank=rank,
    )


def chunks_to_results(spec, start_rank=1):
    return [
        make_result(cid, text, language=lang, score=score, rank=start_rank + i)
        for i, (cid, text, lang, score) in enumerate(spec)
    ]


# --------------------------------------------------------------------------
# Evaluator: the three decision states
# --------------------------------------------------------------------------


class TestAnswerabilityDecisions(unittest.TestCase):
    def setUp(self):
        self.gate = AnswerabilityEvaluator()

    # -- Test 1: CDG English -> ANSWERABLE --------------------------------
    def test_cdg_english_answerable(self):
        decision = self.gate.evaluate(CDG_QUERY, language="eng_Latn", results=chunks_to_results(CDG_CHUNKS))
        self.assertEqual(decision.status, AnswerabilityStatus.ANSWERABLE)
        self.assertEqual(api_status(decision.status), "grounded")
        self.assertGreaterEqual(decision.confidence, 0.62)
        self.assertEqual(decision.reason, "answer_present")
        self.assertGreaterEqual(decision.evidence_count, 1)
        self.assertTrue(decision.supporting_chunk_ids)
        self.assertIn("cdg1", decision.supporting_chunk_ids)

    # -- Test 2: CDG paraphrase -> ANSWERABLE -----------------------------
    def test_cdg_paraphrase_answerable(self):
        decision = self.gate.evaluate(
            CDG_PARAPHRASE, language="eng_Latn", results=chunks_to_results(CDG_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.ANSWERABLE)
        self.assertGreaterEqual(decision.confidence, 0.62)

    # -- Test 3: CDG Urdu -> ANSWERABLE -----------------------------------
    def test_cdg_urdu_answerable(self):
        decision = self.gate.evaluate(
            CDG_URDU_QUERY, language="urd_Arab", results=chunks_to_results(CDG_URDU_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.ANSWERABLE)
        self.assertGreaterEqual(decision.confidence, 0.62)
        self.assertIn("urd1", decision.supporting_chunk_ids)

    # -- Test 4: capital of France over CDG chunks -> NOT answerable -------
    def test_capital_of_france_not_answerable(self):
        decision = self.gate.evaluate(
            CAPITAL_FRANCE, language="eng_Latn", results=chunks_to_results(CDG_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(api_status(decision.status), "insufficient_evidence")
        self.assertEqual(decision.supporting_chunk_ids, [])
        self.assertLess(decision.confidence, 0.6)
        self.assertEqual(decision.reason, "weak_answer_overlap")

    # -- Test 5: telephone inventor -> NOT answerable ----------------------
    def test_telephone_inventor_not_answerable(self):
        decision = self.gate.evaluate(
            TELEPHONE_QUERY, language="eng_Latn", results=chunks_to_results(TELEPHONE_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(decision.supporting_chunk_ids, [])
        self.assertLess(decision.confidence, 0.6)

    # -- Test 6: exact answer context -> ANSWERABLE high confidence --------
    def test_exact_answer_high_confidence(self):
        decision = self.gate.evaluate(
            CDG_QUERY, language="eng_Latn", results=chunks_to_results(EXACT_ANSWER_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.ANSWERABLE)
        self.assertGreaterEqual(decision.confidence, 0.7)
        self.assertGreaterEqual(len(decision.supporting_chunk_ids), 2)

    # -- Test 7: partial / contextual evidence -> UNCERTAIN ----------------
    def test_partial_evidence_uncertain(self):
        results = [make_result("p1", "CDG is located about 23 km north-east of Paris.")]
        decision = self.gate.evaluate("Is CDG far from Paris?", language="eng_Latn", results=results)
        self.assertEqual(decision.status, AnswerabilityStatus.UNCERTAIN)
        self.assertEqual(api_status(decision.status), "uncertain")
        self.assertGreaterEqual(decision.confidence, 0.42)
        self.assertLess(decision.confidence, 0.62)
        self.assertEqual(decision.supporting_chunk_ids, ["p1"])

    # -- Test 8: synthetic irrelevant context -> UNANSWERABLE_FROM_RAG -----
    def test_synthetic_irrelevant_context_unanswerable(self):
        decision = self.gate.evaluate(
            CAPITAL_FRANCE, language="eng_Latn", results=chunks_to_results(IRRELEVANT_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.reason, "no_answer_terms")
        self.assertEqual(decision.supporting_chunk_ids, [])


# --------------------------------------------------------------------------
# Evaluator edge cases and calibration
# --------------------------------------------------------------------------


class TestAnswerabilityEdges(unittest.TestCase):
    def setUp(self):
        self.gate = AnswerabilityEvaluator()

    def test_no_results_unanswerable(self):
        decision = self.gate.evaluate(CAPITAL_FRANCE, language="eng_Latn", results=[])
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.reason, "no_evidence")
        self.assertEqual(decision.evidence_count, 0)

    def test_blank_texts_unanswerable(self):
        decision = self.gate.evaluate(
            CAPITAL_FRANCE, language="eng_Latn", results=[make_result("x", "   ")]
        )
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)
        self.assertEqual(decision.reason, "no_evidence")

    def test_empty_query_is_never_answerable(self):
        decision = self.gate.evaluate("", language="eng_Latn", results=chunks_to_results(CDG_CHUNKS))
        self.assertEqual(decision.status, AnswerabilityStatus.UNANSWERABLE_FROM_RAG)

    def test_language_mismatch_lowers_confidence(self):
        match = self.gate.evaluate(
            CDG_URDU_QUERY, language="urd_Arab", results=chunks_to_results(CDG_URDU_CHUNKS)
        )
        mismatch = self.gate.evaluate(
            CDG_URDU_QUERY, language="urd_Arab",
            results=[make_result("m1", text, language="eng_Latn", score=score, rank=i + 1)
                     for i, (_, text, _, score) in enumerate(CDG_URDU_CHUNKS)],
        )
        self.assertEqual(match.status, AnswerabilityStatus.ANSWERABLE)
        self.assertGreater(match.confidence, mismatch.confidence)

    def test_unknown_language_is_not_penalized(self):
        decision = self.gate.evaluate(
            CDG_QUERY, language=None, results=chunks_to_results(CDG_CHUNKS)
        )
        self.assertEqual(decision.status, AnswerabilityStatus.ANSWERABLE)

    def test_near_duplicates_are_deduplicated(self):
        base = "Charles de Gaulle Airport (CDG) serves Paris as the main international airport."
        chunks = [
            make_result("d1", base, score=0.032, rank=1),
            make_result("d2", base + " It is in France.", score=0.016, rank=2),
            make_result("d3", base, score=0.016, rank=3),
        ]
        related = AnswerabilityEvaluator._related_chunks(chunks, chunks[0])
        self.assertEqual(related, 0.0)

    def test_distinct_related_chunks_count_toward_confidence(self):
        results = chunks_to_results(EXACT_ANSWER_CHUNKS)
        related = AnswerabilityEvaluator._related_chunks(results, results[0])
        self.assertGreaterEqual(related, 0.25)

    def test_urdu_question_words_are_stopwords(self):
        from app.rag.answerability import _content_tokens

        tokens = _content_tokens(CDG_URDU_QUERY)
        self.assertIn("سی", tokens)
        self.assertIn("اڈا", tokens)
        self.assertNotIn("کیا", tokens)
        self.assertNotIn("ہے", tokens)

    def test_decision_schema(self):
        decision = self.gate.evaluate(CDG_QUERY, language="eng_Latn", results=chunks_to_results(CDG_CHUNKS))
        self.assertIsInstance(decision, AnswerabilityDecision)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)

    def test_api_status_mapping(self):
        self.assertEqual(api_status(AnswerabilityStatus.ANSWERABLE), "grounded")
        self.assertEqual(api_status(AnswerabilityStatus.UNCERTAIN), "uncertain")
        self.assertEqual(api_status(AnswerabilityStatus.UNANSWERABLE_FROM_RAG), "insufficient_evidence")
        self.assertEqual(api_status("bogus"), "insufficient_evidence")


# --------------------------------------------------------------------------
# Pipeline integration: retrieval -> gate -> LLM decision
# --------------------------------------------------------------------------


def make_stt(text):
    stt = mock.Mock()
    stt.transcribe.return_value = TranscriptionResult(
        text=text, language="en" if text.isascii() else "ur",
        duration_seconds=1.5, processing_time_ms=120.0,
    )
    return stt


def make_retriever(results):
    retriever = mock.Mock()
    retriever.strategy = "recursive"
    retriever.chunk_count = 9964
    retriever.search.return_value = results
    return retriever


def make_llm():
    llm = mock.Mock()
    llm.model = "qwen2.5:3b"
    llm.generate.return_value = LLMResponse(
        answer="CDG is Roissy-Charles de Gaulle Airport serving Paris, France.",
        model="qwen2.5:3b",
        grounded=True,
        context_count=3,
        language="eng_Latn",
        latency_ms=500.0,
        abstained=False,
        usage=None,
    )
    return llm


def make_verifier():
    verifier = mock.Mock()
    verifier.enabled = True
    verifier.verify.return_value = SupportVerdict(
        supports_answer=True, confidence=1.0, reason="supports answer"
    )
    return verifier


class TestPipelineGateIntegration(unittest.TestCase):
    def test_answerable_query_reaches_the_llm(self):
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt(CDG_QUERY),
            guardrail=InputGuardrail(),
            retriever=make_retriever(chunks_to_results(CDG_CHUNKS)),
            llm=llm,
            verifier=make_verifier(),
        )
        resp = pipeline.process_audio("q.wav")
        self.assertEqual(resp.generation.status, "grounded")
        self.assertGreaterEqual(resp.generation.confidence, 0.62)
        self.assertEqual(resp.generation.reason, "answer_present")
        self.assertTrue(resp.generation.supporting_chunk_ids)
        self.assertGreater(len(resp.generation.sources), 0)
        self.assertEqual(llm.generate.call_count, 1)
        self.assertEqual(
            resp.generation.answer,
            "CDG is Roissy-Charles de Gaulle Airport serving Paris, France.",
        )

    def test_unanswerable_query_skips_the_llm(self):
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt(CAPITAL_FRANCE),
            guardrail=InputGuardrail(),
            retriever=make_retriever(chunks_to_results(CDG_CHUNKS)),
            llm=llm,
            router=FallbackRouter(enable_general_knowledge=False),
        )
        resp = pipeline.process_audio("q.wav")
        self.assertEqual(resp.generation.status, "insufficient_evidence")
        self.assertEqual(resp.generation.answer, ABSTENTION_EN)
        self.assertTrue(resp.generation.abstained)
        self.assertFalse(resp.generation.grounded)
        self.assertEqual(resp.generation.sources, [])
        self.assertEqual(resp.generation.supporting_chunk_ids, [])
        self.assertLess(resp.generation.confidence, 0.6)
        self.assertEqual(resp.timings.llm_ms, 0.0)
        llm.generate.assert_not_called()

    def test_uncertain_query_reaches_the_llm_with_qualified_status(self):
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt("Is CDG far from Paris?"),
            guardrail=InputGuardrail(),
            retriever=make_retriever(
                [make_result("p1", "CDG is located about 23 km north-east of Paris.")]
            ),
            llm=llm,
        )
        resp = pipeline.process_audio("q.wav")
        self.assertEqual(resp.generation.status, "uncertain")
        self.assertGreaterEqual(resp.generation.confidence, 0.42)
        self.assertLess(resp.generation.confidence, 0.62)
        self.assertEqual(llm.generate.call_count, 1)

    def test_unanswerable_urdu_query_uses_urdu_abstention(self):
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt(CDG_URDU_QUERY),
            guardrail=InputGuardrail(),
            retriever=make_retriever([make_result("en1", "The capital of France is Paris.")]),
            llm=llm,
            router=FallbackRouter(enable_general_knowledge=False),
        )
        resp = pipeline.process_audio("q.wav")
        self.assertEqual(resp.generation.status, "insufficient_evidence")
        self.assertEqual(resp.generation.answer, ABSTENTION_UR)
        llm.generate.assert_not_called()

    def test_supporting_chunk_context_forwarded_to_llm(self):
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt(CDG_QUERY),
            guardrail=InputGuardrail(),
            retriever=make_retriever(chunks_to_results(CDG_CHUNKS)),
            llm=llm,
            verifier=make_verifier(),
        )
        pipeline.process_audio("q.wav")
        context = llm.generate.call_args.args[1]
        self.assertGreater(len(context), 0)
        self.assertEqual(context[0], CDG_CHUNKS[0][1])

    def test_evaluator_is_injectable(self):
        evaluator = mock.Mock()
        evaluator.evaluate.return_value = AnswerabilityDecision(
            status=AnswerabilityStatus.ANSWERABLE,
            confidence=0.9,
            reason="answer_present",
            evidence_count=2,
            best_score=0.03,
            supporting_chunk_ids=["cdg1", "cdg2"],
        )
        llm = make_llm()
        pipeline = QueryPipeline(
            stt=make_stt(CDG_QUERY),
            guardrail=InputGuardrail(),
            retriever=make_retriever(chunks_to_results(CDG_CHUNKS)),
            llm=llm,
            evaluator=evaluator,
            verifier=make_verifier(),
        )
        resp = pipeline.process_audio("q.wav")
        self.assertEqual(resp.generation.status, "grounded")
        self.assertEqual(resp.generation.confidence, 0.9)
        self.assertEqual(resp.generation.supporting_chunk_ids, ["cdg1", "cdg2"])
        llm.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
