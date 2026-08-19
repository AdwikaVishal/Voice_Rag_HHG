# Voice-Enabled RAG System

A multilingual (English + Urdu) voice-based Retrieval-Augmented Generation system. Speak a question, and the system transcribes it, retrieves relevant evidence from a document corpus, decides whether it can actually answer from that evidence, generates a grounded response with an LLM, verifies the answer is supported, and speaks the answer back — all with full transparency into sources, confidence, and timings.

> **HH Goa 2026 — Task 2**

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Debugging](#debugging)

---

## What It Does

**Input:** 🎤 Audio (WAV, MP3, M4A, WebM, Ogg) in English or Urdu
**Output:** 📝 Answer text + 🔊 Synthesized audio + 📋 Cited sources + ✓/✗ Grounding verdict

```
Audio → STT → Language Detect → Guardrail → Hybrid Retrieval → Answerability Gate
      → Answer Verification → Router → LLM Generation → Grounding Check → TTS → Audio
```

**Design principle:** the system never hallucinates by omission — it abstains or falls back to general knowledge (clearly labeled) rather than fabricating answers from irrelevant context. Every response includes the retrieved sources, a grounding verdict, and a per-stage latency breakdown.

**Total latency:** ~11–22 seconds end-to-end (warm state, CPU).

---

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Input | Speech Recognition | Faster-Whisper (small, CPU, int8) |
| — | Language Detection | Script-based (Unicode ranges) |
| Validation | Input Guardrails | Deterministic rules |
| Retrieval | Dense + Sparse | FAISS (multilingual-e5-small) + BM25 + RRF |
| Analysis | Answerability Gate | 5-signal weighted confidence model |
| Verification | Answer-Support Check | Token matching + Q&A structure |
| Routing | Route Selection | Gate verdict → answer source |
| Generation | Grounded LLM | Ollama `qwen2.5:3b` (local) |
| Validation | Grounding Check | Lexical token-overlap |
| Output | Text-to-Speech | Edge-TTS (Microsoft) |

### End-to-end flow

```
VOICE INPUT (mic or file)
   │
   ▼
STT Service ─────────► Faster-Whisper decodes .webm/.wav/.mp3/.m4a/.ogg via PyAV
   │
   ▼
Language Detection ──► Unicode script check → eng_Latn / urd_Arab
   │
   ▼
Input Guardrail ─────► reject empty / >512 chars / injection patterns / unsupported language
   │
   ▼
Hybrid Retrieval ────► FAISS (dense) + BM25 (sparse) → Reciprocal Rank Fusion (k=60)
   │
   ▼
Answerability Gate ──► 5 weighted signals → ANSWERABLE / UNCERTAIN / UNANSWERABLE_FROM_RAG
   │
   ▼
Answer-Support Verify ► confirms retrieved text actually answers the question
   │
   ▼
Routing Decision ────► RAG_GROUNDED / RAG_UNCERTAIN / GENERAL_KNOWLEDGE / ABSTAIN
   │
   ▼
LLM Generation ──────► Ollama qwen2.5:3b, grounded or general-knowledge prompt
   │
   ▼
Grounding Validation ► lexical overlap, tri-state true/false/null
   │
   ▼
TTS Synthesis ───────► Edge-TTS → WAV
   │
   ▼
VOICE + TEXT OUTPUT (audio/wav + X-Voice-RAG-Meta header)
```

For fully detailed component diagrams (retrieval internals, gate decision tree, error-handling flowchart, and a real example timeline), see `SYSTEM_ARCHITECTURE_GUIDE.md`.

---

## Quick Start

```bash
cd backend
python -m venv venv && source venv/bin/activate    # venv\Scripts\activate on Windows
pip install -r requirements.txt

python scripts/prepare_dataset.py          # first time only — download + prepare corpus
python scripts/build_retrieval_index.py    # first time only — build FAISS + BM25 indexes

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the UI at **http://127.0.0.1:8000/ui/**

Or serve the frontend separately:
```bash
python -m http.server 8001 -d frontend
# → http://localhost:8001/?api=http://127.0.0.1:8000
```

### Run tests

```bash
python -m unittest discover -s tests -v   # 389 tests total
python scripts/final_smoke_test.py        # comprehensive PRD-compliance check
```

---

## Pipeline Stages

### 1. Speech-to-Text
Faster-Whisper (small model, CPU, int8) transcribes audio. Supports WAV, MP3, M4A, WebM, Ogg. Latency ~3–6s for short clips.

### 2. Language Detection
Pure Unicode script matching — no ML model. Latin script → `eng_Latn`, Arabic script → `urd_Arab`.

### 3. Input Guardrail
Deterministic rejection rules:

| Rule | Condition |
|---|---|
| Empty | Whitespace only |
| Too long | > 512 characters |
| Injection | SQL/code-like patterns |
| Unsupported language | Not `eng_Latn` or `urd_Arab` |

### 4. Chunking (corpus preparation)
Four interchangeable strategies, all producing typed `Chunk` models with preserved metadata:

| Strategy | Approach | Notes |
|---|---|---|
| Fixed | ~256 tokens, 20% overlap | Baseline |
| Sentence | Natural sentence boundaries | — |
| **Recursive** | Paragraph → sentence → token hierarchy | **Selected for production** |
| Metadata | Sentence-aware + language/source enrichment | Multilingual filtering |

Production build: 100,783 chunks from the full corpus, median 54 tokens/chunk.

### 5. Hybrid Retrieval
```
Query ─┬─► Dense (FAISS, multilingual-e5-small, 384-dim, cosine similarity)
       └─► Sparse (BM25Okapi, Unicode tokenization)
             │
             ▼
       Reciprocal Rank Fusion (k=60) — chunks in both lists get boosted
```
Retrieval latency: 22.3ms (P50) / 30.5ms (P95) / 34.2ms (P99). Throughput ~45 queries/sec.

### 6. Answerability Gate
Five deterministic, weighted signals decide whether retrieved evidence actually supports an answer:

| Signal | Weight | Checks |
|---|---|---|
| `answer_presence` | 45% | Query terms present in best chunk |
| `hybrid_rank` | 15% | Top-result rank position |
| `retrieval_agreement` | 15% | Dense/BM25 consensus (RRF score fall-off) |
| `related_chunks` | 15% | Topic coherence (Jaccard similarity) |
| `language_match` | 10% | Chunk language matches query language |

```
if answer_presence < 0.25          → UNANSWERABLE_FROM_RAG
elif confidence >= 0.90            → ANSWERABLE
elif confidence >= 0.60            → UNCERTAIN
else                                → UNANSWERABLE_FROM_RAG
```

### 7. Answer-Support Verification
Confirms the retrieved passage genuinely answers the question (token coverage + Q&A structure). Rejection downgrades the route to general knowledge rather than forcing a bad grounding.

### 8. Routing

| Gate Verdict | Route | `source` | Behavior |
|---|---|---|---|
| ANSWERABLE | RAG_GROUNDED | `rag` | Generate from retrieved context |
| UNCERTAIN | RAG_UNCERTAIN | `clarification` | Cautious answer with caveats |
| UNANSWERABLE (fallback on) | GENERAL_KNOWLEDGE | `general_knowledge` | LLM answers from its own knowledge, no chunks mixed in |
| UNANSWERABLE (fallback off) | ABSTAIN | `abstained` | Refuse to answer |

### 9. LLM Generation
Ollama `qwen2.5:3b`, local CPU inference, language-specific prompts. Never called with empty context by design — no hallucination path. Latency: EN 6–15s, UR 25–40s.

### 10. Grounding Validation
Lexical token-overlap between generated answer and source chunk. Threshold 40%. Tri-state: `true` / `false` / `null` (null when abstained). **Limitation:** cannot detect fabricated statistics layered onto a true claim.

### 11. Text-to-Speech
Edge-TTS (Microsoft), voices `en-US-AriaNeural` / `ur-PK-UzmaNeural`. Output transcoded via PyAV. Latency 1–2s; requires network.

---

## API Reference

```
GET  /                          Health check
GET  /health                    Status + component health
GET  /retriever/info            Index configuration

POST /search                    Text → retrieved chunks only
POST /query                     Text → answer (JSON)
POST /stt                       Audio → transcription
POST /voice/query                Audio → answer (JSON)
POST /voice/query/audio          Audio → answer (WAV + X-Voice-RAG-Meta header)
```

### Text query example

```python
import requests

response = requests.post("http://127.0.0.1:8000/query", json={
    "query": "What is CDG airport?",
    "top_k": 5,
    "language": "en"
})

result = response.json()
print(result["answer"])     # generated answer
print(result["source"])     # rag / clarification / general_knowledge / abstained
print(result["grounded"])   # True / False / None
print(result["sources"])    # retrieved chunks with scores
```

### Voice query example

```bash
curl -X POST http://127.0.0.1:8000/voice/query/audio \
  -F "audio=@question.wav" \
  -F "language=en" \
  --output answer.wav \
  -H "Accept: audio/wav"
# X-Voice-RAG-Meta response header carries the JSON metadata
```

### Response shape (`VoiceQueryResponse`)

```
transcript          str
answer              str
source               "rag" | "clarification" | "general_knowledge" | "abstained"
grounded             bool | null
language             str
sources[]            { id, score, language, excerpt }
generation_info      { model, prompt_tokens, completion_tokens, temperature }
guardrail_info       { passed, reason, normalized_query }
retrieval_info       { strategy, chunks_searched, language_filter }
timings              { stt_ms, retrieval_ms, gate_ms, llm_ms, tts_ms, total_ms }
```

### Error format

```json
{
  "detail": {
    "code": "stt_failed" | "retrieval_failed" | "llm_failed",
    "message": "User-friendly error message"
  }
}
```

| Stage | Failure → | Status |
|---|---|---|
| Audio upload | invalid/empty/too large | 400 |
| Decode | PyAV/ffmpeg error | 422 |
| STT | Whisper error | 500 |
| Guardrail | rejected input | 422 (no LLM call made) |
| Retrieval | FAISS/index error | 500 |
| LLM | Ollama unavailable | 502 |
| TTS | synthesis failure | 200 with text-only answer, or 500 depending on config |

---

## Configuration

Set via environment variables (`.env`):

```bash
# Speech-to-text
STT_MODEL=small              # tiny/base/small/medium
STT_DEVICE=cpu                # cpu/cuda/mps
STT_PRECISION=int8            # int8/float16/float32

# Retrieval
CHUNKING_STRATEGY=recursive
RETRIEVAL_INDEX_DIR=data/indexes/recursive/
RETRIEVAL_RRF_K=60

# Answerability gate
ANSWERABILITY_HIGH_CONFIDENCE=0.90
ANSWERABILITY_LOW_CONFIDENCE=0.60
ANSWERABILITY_LEXICAL_FLOOR=0.25

# LLM
LLM_PROVIDER=ollama            # ollama/azure/openai
LLM_MODEL=qwen2.5:3b
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=256

# Text-to-speech
TTS_PROVIDER=edge_tts          # edge_tts/azure
TTS_VOICE_EN=en-US-AriaNeural
TTS_VOICE_UR=ur-PK-UzmaNeural

# General
GENERAL_KNOWLEDGE_FALLBACK=true    # answer from LLM knowledge when RAG can't
GROUNDING_VERIFY_ENABLED=false     # optional LLM-based verifier (off by default)
```

---

## Project Structure

```
Voice_Rag_HHG/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + endpoints
│   │   ├── config.py                # Env-driven configuration
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── chunking/                # 4 chunking strategies
│   │   ├── retrieval/               # FAISS, BM25, RRF fusion
│   │   ├── guardrails/              # Input validation
│   │   ├── stt/                     # Faster-Whisper wrapper
│   │   ├── tts/                     # Edge-TTS wrapper
│   │   ├── llm/                     # Ollama wrapper + prompts
│   │   ├── rag/                     # Gate, verifier, router, grounding
│   │   └── pipeline/                # QueryPipeline orchestration
│   ├── data/
│   │   ├── processed/               # msmarco_xi_sample.jsonl, chunked corpora
│   │   └── indexes/recursive/       # faiss.index, bm25_index.pkl, metadata.json
│   ├── scripts/                     # dataset prep, index build, benchmarks, smoke tests
│   ├── tests/                       # 389 tests across 15+ modules
│   ├── requirements.txt
│   └── PRD_COMPLIANCE_REPORT.md
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── SYSTEM_ARCHITECTURE_GUIDE.md
```

---

## Performance

**Latency (Apple M2 CPU, warm state)**

| Stage | Typical Time |
|---|---|
| STT | 3–6 s |
| Language detection | < 1 ms |
| Guardrail | < 1 ms |
| Hybrid retrieval | 22–35 ms |
| Answerability gate | < 1 ms |
| Answer-support verification | < 1 ms |
| LLM generation (EN) | 6–15 s |
| LLM generation (UR) | 25–40 s |
| TTS synthesis | 1–2 s |
| **Total (English, warm)** | **~11–22 s** |
| First request (cold start) | +12–15 s for model/index loading |

**Retrieval accuracy** (benchmark index, 318 queries)

| Strategy | Chunks | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Fixed | 9,974 | 0.261 | 0.491 | 0.635 | 0.777 |
| **Recursive ✓** | 9,964 | 0.258 | 0.500 | 0.629 | **0.786** |
| Sentence | 9,945 | 0.258 | 0.500 | 0.629 | 0.783 |
| Metadata | 9,945 | 0.258 | 0.500 | 0.629 | 0.783 |

Hybrid retrieval improves recall +2–5% over dense- or sparse-only search.

**Throughput:** ~45 retrieval queries/sec; 1–2 audio files/sec (sequential STT).

---

## Testing

```
Total: 389 tests | Passed: 383 | Failed: 3 (pre-existing, Hindi unsupported) | Skipped: 3 (opt-in)

test_chunking.py .................... 30
test_retrieval.py ................... 35
test_production_retriever.py ........ 18
test_stt.py ......................... 25
test_input_guardrail.py ............. 23
test_query_pipeline.py .............. 60+
test_llm.py .......................... 31
test_tts.py .......................... 19
(+ answerability, router, verifier, grounding)
```

```bash
python -m unittest discover -s tests -v
python scripts/final_smoke_test.py
```

---

## Known Limitations

| Issue | Impact | Workaround |
|---|---|---|
| Urdu STT auto-detection | Whisper labels Urdu audio as `hi` | Pass `language=ur` explicitly |
| Grounding verifier disabled by default | Can't catch fabricated statistics layered on true claims | Enable `GROUNDING_VERIFY_ENABLED` with a larger model |
| Hindi unsupported | Script detection returns `None` | Use English or Urdu only |
| CPU inference | 6–40s LLM latency | Use GPU for production |
| TTS needs network | Fails offline | Text answer still returned; add fallback synthesizer |
| Benchmark-scale index by default | 9,964 chunks, not full 100,783 | `python scripts/build_retrieval_index.py --full` |

---

## Debugging

```bash
curl http://127.0.0.1:8000/health          # component health
curl http://127.0.0.1:8000/retriever/info  # index config

python scripts/test_stt.py                 # STT only
python scripts/evaluate_retrieval.py       # retrieval only
python scripts/test_llm.py                 # LLM only
python scripts/test_voice_pipeline.py      # full pipeline
python scripts/final_smoke_test.py         # PRD compliance
```

---

## Key Takeaways

1. **End-to-end voice QA** — microphone to speaker, no manual steps.
2. **Multilingual** — English and Urdu with automatic language detection.
3. **Deterministic, never-hallucinate design** — abstains or clearly labels fallback rather than fabricating from irrelevant context.
4. **Fully transparent** — every response surfaces sources, grounding verdict, and per-stage timings.
5. **Modular** — each pipeline stage is independently testable and configurable via env vars.
6. **Production-ready** — 389 tests, hybrid retrieval benchmarked against 4 chunking strategies.

Further reading: `SYSTEM_ARCHITECTURE_GUIDE.md` (full diagrams) · `backend/PRD_COMPLIANCE_REPORT.md` (requirements traceability) · `backend/README.md` (setup details)