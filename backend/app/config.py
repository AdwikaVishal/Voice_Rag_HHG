"""Application configuration.

Settings are read from environment variables so the same code can serve any
chunking strategy or index location without code changes. The production
defaults follow the Segment 4 benchmark verdict: ``recursive`` chunking served
by a hybrid (FAISS dense + BM25 + RRF) retriever.

Environment variables:

* ``CHUNKING_STRATEGY``        — production chunking strategy (default ``recursive``)
* ``VOICE_RAG_DATA_DIR``       — data root (default ``<backend>/data``)
* ``VOICE_RAG_CHUNKS_DIR``     — chunk JSONL directory (default ``data/processed/chunks``)
* ``VOICE_RAG_INDEXES_DIR``    — index parent (default ``data/indexes``; production loads
  ``<parent>/<strategy>``, e.g. ``data/indexes/recursive`` for the full-corpus recursive index)
* ``VOICE_RAG_MODEL_NAME``     — embedding model id
* ``VOICE_RAG_EMBEDDING_BATCH_SIZE`` — encode batch size (default 32)
* ``VOICE_RAG_RRF_K``          — Reciprocal Rank Fusion constant (default 60)
* ``VOICE_RAG_DEFAULT_TOP_K``  — default result count (default 10)
* ``VOICE_RAG_MAX_TOP_K``      — upper bound accepted by the API (default 50)

Speech-to-text (Segment 4A):

* ``STT_PROVIDER``             — STT backend: ``faster_whisper`` (default, local) or ``sarvam`` (cloud)
* ``SARVAM_API_KEY``           — Sarvam API key (required when provider is ``sarvam``)
* ``SARVAM_MODEL``             — Sarvam model id (default ``sarvam_v1``)
* ``SARVAM_LANGUAGE``          — Sarvam language hint (default ``auto``; e.g. ``en``, ``ur``)
* ``STT_MODEL_SIZE``           — faster-whisper model size (default ``small``)
* ``STT_DEVICE``               — device (default ``cpu``)
* ``STT_COMPUTE_TYPE``         — CTranslate2 compute type (default ``int8``)
* ``STT_ALLOWED_EXTENSIONS``   — comma-separated accepted suffixes (default
  ``.wav,.mp3,.m4a,.webm,.ogg`` — the browser MediaRecorder formats are
  decoded by faster-whisper's bundled PyAV/ffmpeg)
* ``STT_MAX_UPLOAD_BYTES``     — max accepted upload size (default 25 MB)

Input guardrail (Segment 4B):

* ``GR_MAX_INPUT_CHARS``       — max normalized query length (default 2000)

LLM answer generation (Segment 4C):

* ``LLM_PROVIDER``             — provider backend: ``ollama`` (default) or ``openai``
* ``LLM_MODEL``                — model id (default ``qwen2.5:3b``)
* ``LLM_API_KEY``              — API key for ``openai`` (env only, read at construction, never logged)
* ``LLM_BASE_URL``             — provider base URL (default per provider)
* ``LLM_TIMEOUT_S``            — HTTP timeout for generation (default 120)
* ``LLM_TEMPERATURE``          — sampling temperature (default 0.2)
* ``LLM_MAX_TOKENS``           — max generated tokens (default 300)
 * ``LLM_MAX_CONTEXT_CHARS``    — context block size cap for the prompt (default 8000)

RAG answerability gate (Segment 4):

* ``ANSWERABILITY_TOP_K``         — results considered by the gate (default 5)
* ``ANSWERABILITY_LEXICAL_FLOOR`` — min fraction of query content terms that must
  appear in a chunk for the query to be answerable at all (default 0.6)
* ``ANSWERABILITY_HIGH_CONFIDENCE`` — confidence at/above which a query is
  ANSWERABLE (default 0.62)
* ``ANSWERABILITY_LOW_CONFIDENCE``  — confidence at/above which a query is
  UNCERTAIN (default 0.42)

Answer-support verification (Segment 5.1):

* ``ANSWER_SUPPORT_VERIFY_ENABLED`` — after the gate says ANSWERABLE, run a
  deterministic evidence-support check that confirms the supporting evidence
  DIRECTLY answers the exact question before RAG generation; a rejected verdict
  routes to the general-knowledge fallback instead (default ``true``)
* ``ANSWER_SUPPORT_VERIFY_CEILING`` — gate confidence at/above which the
  verifier is skipped and the gate verdict is trusted outright, so clearly
  grounded queries do not pay the extra check (default ``0.85``)

General-knowledge fallback (Segment 5):

* ``GENERAL_KNOWLEDGE_FALLBACK`` — when the answerability gate finds no usable
  RAG evidence, route to the general-knowledge provider instead of abstaining
  (default ``true``; set ``false`` to restore the Segment 4 pure-abstention
  behavior). The general provider reuses the existing ``LLM_*`` configuration —
  no second model, provider or client is created.

Text-to-speech (Segment 4D):

* ``TTS_PROVIDER``             — TTS backend: ``edge`` (default) — Microsoft Edge
  neural voices via ``edge-tts`` (no API key; requires network)
* ``TTS_VOICE_EN``             — English voice (default ``en-US-AriaNeural``)
* ``TTS_VOICE_UR``             — Urdu voice (default ``ur-PK-UzmaNeural``)
* ``TTS_RATE``                 — speaking-rate override (default ``+0%``)
* ``TTS_FORMAT``               — output container (default ``wav``)

Web interface (Segment 5):

* ``VOICE_RAG_ALLOWED_ORIGINS`` — comma-separated CORS-allowed frontend origins
  (default ``http://localhost:8001`` — only the development frontend is allowed)
* ``VOICE_RAG_FRONTEND_DIR``    — static frontend directory served at ``/ui``
  (default ``<repo>/frontend``)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from backend folder
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Chunking strategy selected for production from the Segment 4 benchmark.
CHUNKING_STRATEGY = _env("CHUNKING_STRATEGY", "recursive")

DATA_DIR = Path(_env("VOICE_RAG_DATA_DIR", str(BASE_DIR / "data")))
CHUNKS_DIR = Path(_env("VOICE_RAG_CHUNKS_DIR", str(DATA_DIR / "processed" / "chunks")))
# Production index parent. The full-corpus recursive index lives at
# ``data/indexes/recursive``; the Segment 3 benchmark sample indexes stay at
# ``data/indexes/benchmark/{strategy}`` (reproducible, but not what production
# loads). Benchmark scripts pass explicit --indexes-dir paths.
INDEXES_DIR = Path(_env("VOICE_RAG_INDEXES_DIR", str(DATA_DIR / "indexes")))

MODEL_NAME = _env("VOICE_RAG_MODEL_NAME", "intfloat/multilingual-e5-small")
EMBEDDING_BATCH_SIZE = int(_env("VOICE_RAG_EMBEDDING_BATCH_SIZE", "32"))
RRF_K = float(_env("VOICE_RAG_RRF_K", "60"))
DEFAULT_TOP_K = int(_env("VOICE_RAG_DEFAULT_TOP_K", "10"))
MAX_TOP_K = int(_env("VOICE_RAG_MAX_TOP_K", "50"))

# Speech-to-text (Segment 4A) — provider-based (faster-whisper or Sarvam).
STT_PROVIDER = _env("STT_PROVIDER", "faster_whisper").strip().lower()

# Sarvam cloud API settings (used only when STT_PROVIDER == "sarvam").
SARVAM_API_KEY = _env("SARVAM_API_KEY", "")
SARVAM_MODEL = _env("SARVAM_MODEL", "sarvam_v1")
SARVAM_LANGUAGE = _env("SARVAM_LANGUAGE", "auto")

# Faster-Whisper local model settings (used only when STT_PROVIDER == "faster_whisper").
STT_MODEL_SIZE = _env("STT_MODEL_SIZE", "small")
STT_DEVICE = _env("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = _env("STT_COMPUTE_TYPE", "int8")
STT_ALLOWED_EXTENSIONS = tuple(
    ext.strip().lower()
    for ext in _env(
        "STT_ALLOWED_EXTENSIONS", ".wav,.mp3,.m4a,.webm,.ogg"
    ).split(",")
    if ext.strip()
)
STT_MAX_UPLOAD_BYTES = int(_env("STT_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Input guardrail (Segment 4B) — deterministic query pre-checks.
GR_MAX_INPUT_CHARS = int(_env("GR_MAX_INPUT_CHARS", "2000"))

# LLM answer generation (Segment 4C). The API key is read from the
# environment at service-construction time and is never logged or exposed.
LLM_PROVIDER = _env("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:3b")
LLM_BASE_URL = _env("LLM_BASE_URL", "").rstrip("/")
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "120"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "300"))
LLM_MAX_CONTEXT_CHARS = int(_env("LLM_MAX_CONTEXT_CHARS", "8000"))

# RAG answerability gate (Segment 4) — deterministic pre-LLM verdict.
ANSWERABILITY_TOP_K = int(_env("ANSWERABILITY_TOP_K", "5"))
ANSWERABILITY_LEXICAL_FLOOR = float(_env("ANSWERABILITY_LEXICAL_FLOOR", "0.6"))
ANSWERABILITY_HIGH_CONFIDENCE = float(_env("ANSWERABILITY_HIGH_CONFIDENCE", "0.62"))
ANSWERABILITY_LOW_CONFIDENCE = float(_env("ANSWERABILITY_LOW_CONFIDENCE", "0.42"))

# Answer-support verification (Segment 5.1). A retrieved passage can be strongly
# similar to a query without actually answering it ("What is the capital of
# France?" over a passage about Versailles). When the gate verdict is ANSWERABLE,
# the verifier confirms the supporting evidence directly answers the exact
# question before the RAG LLM is allowed to generate from it. The check is
# deterministic (full query-content coverage + question-answer structure), so it
# adds no model latency.
ANSWER_SUPPORT_VERIFY_ENABLED = _env("ANSWER_SUPPORT_VERIFY_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Confidence ceiling: above this the gate verdict is trusted without the extra
# verifier check. Current gate scores rarely exceed ~0.75, so in practice the
# verifier runs on essentially all ANSWERABLE verdicts.
ANSWER_SUPPORT_VERIFY_CEILING = float(_env("ANSWER_SUPPORT_VERIFY_CEILING", "0.85"))

GROUNDING_VERIFY_ENABLED = _env("GROUNDING_VERIFY_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# General-knowledge fallback (Segment 5). When true, an UNANSWERABLE_FROM_RAG
# verdict routes to the general-knowledge provider (the user query only, never
# irrelevant RAG chunks); when false the Segment 4 pure abstention is kept.
GENERAL_KNOWLEDGE_FALLBACK = _env("GENERAL_KNOWLEDGE_FALLBACK", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Text-to-speech (Segment 4D). `edge` uses edge-tts (Microsoft Edge neural
# voices) — no API key, network required, and genuine Urdu support via
# ur-PK-UzmaNeural. The Urdu voice is never silently replaced by a Hindi one.
TTS_PROVIDER = _env("TTS_PROVIDER", "edge").strip().lower()
TTS_VOICE_EN = _env("TTS_VOICE_EN", "en-US-AriaNeural")
TTS_VOICE_UR = _env("TTS_VOICE_UR", "ur-PK-UzmaNeural")
TTS_RATE = _env("TTS_RATE", "+0%")
TTS_FORMAT = _env("TTS_FORMAT", "wav").strip().lower()

# Web interface (Segment 5) — CORS is locked to the development frontend
# origin(s); the wildcard is never used without a documented reason.
FRONTEND_DIR = Path(_env("VOICE_RAG_FRONTEND_DIR", str(BASE_DIR.parent / "frontend")))
ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in _env("VOICE_RAG_ALLOWED_ORIGINS", "http://localhost:8001").split(",")
    if origin.strip()
)


def strategy_index_dir(strategy: str) -> Path:
    """Index directory for a chunking strategy under :data:`INDEXES_DIR`."""
    return Path(INDEXES_DIR) / strategy
