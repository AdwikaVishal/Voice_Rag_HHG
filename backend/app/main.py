"""Voice RAG API — FastAPI application.

Endpoints:

* ``GET /``              — health check
* ``GET /retriever/info`` — production retrieval configuration / load state
* ``POST /search``       — hybrid (FAISS + BM25 + RRF) retrieval
* ``POST /query``        — text query: guardrail -> retrieval -> answerability
                           gate -> answer-support verification -> router ->
                           answer (Segments 5 + 5.1)
* ``POST /stt``          — speech-to-text (multipart audio upload, Segment 4A)
* ``POST /voice/query``  — audio -> STT -> input guardrail -> retrieval ->
                           grounded LLM answer -> TTS metadata (Segment 4D)
* ``POST /voice/query/audio`` — same pipeline, returning the synthesized
                           answer as ``audio/wav`` (Segment 4D)

Heavy components (embedding model, FAISS/BM25 indexes, Whisper model) are
lazily loaded on first use and reused for the process lifetime. The TTS client
(edge-tts) is stateless — it needs a network connection but loads no model.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .config import (
    ALLOWED_ORIGINS,
    CHUNKING_STRATEGY,
    DEFAULT_TOP_K,
    FRONTEND_DIR,
    MAX_TOP_K,
    STT_ALLOWED_EXTENSIONS,
    STT_MAX_UPLOAD_BYTES,
)
from .llm import LLMProviderError
from .pipeline import PipelineStageError, VoiceQueryResponse, get_query_pipeline
from .retrieval.filters import detect_script_language
from .retrieval.production import get_production_retriever
from .schemas import RetrievedChunk, SearchRequest, SearchResponse
from .stt import TranscriptionResult, get_stt_service
from .tts import SynthesisError, UnsupportedLanguageError

logger = logging.getLogger("api")

# Surface structured per-request routing logs from the pipeline (gate verdict,
# route, source, timings) together with uvicorn's own output. Loggers below the
# root default to WARNING and would otherwise silently drop the pipeline INFO
# lines that record every routing decision.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Stage-coded error responses for the audio endpoint. The frontend maps these
# codes to clean, user-facing messages (Segment 6 error handling). Internal
# implementation details are never included in the response bodies.
_STAGE_ERRORS = {
    "stt": (500, "stt_failed", "Could not understand the audio."),
    "retrieval": (500, "retrieval_failed", "Could not retrieve relevant information."),
    "llm": (502, "llm_failed", "Could not generate an answer."),
}


def _stage_error_response(exc: PipelineStageError) -> JSONResponse:
    status, code, message = _STAGE_ERRORS.get(
        exc.stage, (500, "pipeline_failed", "Voice query pipeline failed.")
    )
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}})

app = FastAPI(title="Voice RAG API", version="0.1.0")

# CORS is locked to the configured development frontend origin(s) only — never
# the wildcard. The frontend reads a metadata header on the audio response, so
# it must be exposed. When the UI is served from /ui (same origin) CORS is not
# exercised at all; the middleware only matters for a separately-served UI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Voice-RAG-Meta"],
)


@app.get("/")
def root():
    return {"message": "Voice RAG API is running"}


@app.get("/health")
def health():
    """Cheap liveness probe — never loads models or runs inference.

    Reports whether the expensive singletons have been initialized yet, which
    is useful for debugging but costs nothing to compute.
    """
    from .stt import get_stt_service
    from .tts import get_tts_service

    retriever = get_production_retriever()
    return {
        "status": "ok",
        "components": {
            "retriever_loaded": retriever.is_loaded,
            "retriever_chunks": retriever.chunk_count,
            "stt_loaded": get_stt_service().is_loaded,
            "tts_provider": get_tts_service().provider_name,
        },
    }


@app.get("/retriever/info")
def retriever_info():
    retriever = get_production_retriever()
    info = {
        "strategy": retriever.strategy,
        "index_dir": str(retriever.index_dir),
        "model": retriever.model_name,
        "rrf_k": retriever.rrf_k,
        "loaded": retriever.is_loaded,
    }
    if retriever.is_loaded:
        info["chunks"] = retriever.chunk_count
    return info


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    start = time.perf_counter()
    retriever = get_production_retriever()
    results = retriever.search(request.query, top_k=request.top_k, language=request.language)
    latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

    detected = request.language or detect_script_language(request.query)
    return SearchResponse(
        query=request.query,
        detected_language=detected,
        top_k=request.top_k,
        strategy=retriever.strategy,
        index_chunks=retriever.chunk_count,
        latency_ms=latency_ms,
        results=[RetrievedChunk.from_result(r) for r in results],
    )


@app.post(
    "/query",
    response_model=VoiceQueryResponse,
    summary="Text query: guardrail -> retrieval -> answerability gate -> router -> answer",
)
def text_query(request: SearchRequest):
    """Full text query path (Segment 5): the same guardrail -> retrieval ->
    answerability gate -> router flow as the voice pipeline, without STT/TTS.

    The body is identical to ``POST /search`` (``query`` / ``top_k`` /
    ``language``). The response includes the answer, its ``source``
    (``rag`` | ``clarification`` | ``general_knowledge`` | ``abstained``), and
    the gate verdict/confidence. RAG evidence always wins; only queries with no
    usable RAG evidence fall back to general knowledge.
    """
    return get_query_pipeline().process_text(
        request.query, language_hint=request.language, top_k=request.top_k, with_tts=False
    )


@app.exception_handler(FileNotFoundError)
def _handle_missing_index(_request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Speech-to-text (Segment 4A) + voice query pipeline (Segment 4B)
# --------------------------------------------------------------------------


def save_upload_to_temp(audio: UploadFile) -> Path:
    """Validate an upload and stream it to a temp file (never stored).

    Returns the temp path (caller must unlink it). Raises ``HTTPException``:
    unsupported extension -> 415, oversize -> 413, zero bytes -> 400.
    """
    filename = audio.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in STT_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported audio type {ext!r}. "
                f"Allowed extensions: {', '.join(STT_ALLOWED_EXTENSIONS)}."
            ),
        )

    total = 0
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as out_file:
        tmp_path = Path(out_file.name)
        while True:
            chunk = audio.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > STT_MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large (max {STT_MAX_UPLOAD_BYTES} bytes).",
                )
            out_file.write(chunk)
    if total == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    return tmp_path


def transcribe_upload(audio: UploadFile, language: Optional[str] = None) -> TranscriptionResult:
    """Validate an upload, transcribe it, and clean up the temporary file.

    ``language`` is an optional language hint (e.g. ``"ur"``). When supplied it
    is forwarded to the transcription layer verbatim; when ``None`` the model
    auto-detects the spoken language (and the detected code is reported back).
    """
    tmp_path: Optional[Path] = None
    try:
        tmp_path = save_upload_to_temp(audio)
        return get_stt_service().transcribe(tmp_path, language=language or None)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Could not decode audio %r: %s", audio.filename, exc)
        raise HTTPException(
            status_code=422, detail="Audio file could not be decoded (corrupt or invalid)."
        ) from exc
    except Exception as exc:
        logger.error("Transcription failed for %r: %s", audio.filename, exc)
        raise HTTPException(status_code=500, detail="Speech-to-text transcription failed.") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post(
    "/stt",
    response_model=TranscriptionResult,
    summary="Transcribe an audio file (speech-to-text)",
)
def stt(
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a)"),
    language: Optional[str] = Form(
        None,
        description=(
            "Optional language hint (ISO-639-1, e.g. 'ur'). When omitted, "
            "faster-whisper auto-detects the spoken language."
        ),
    ),
):
    return transcribe_upload(audio, language=language)


@app.post(
    "/voice/query",
    response_model=VoiceQueryResponse,
    summary="Voice query: audio -> STT -> guardrail -> hybrid retrieval -> LLM answer -> TTS",
)
def voice_query(
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a)"),
    language: Optional[str] = Form(
        None,
        description=(
            "Optional language hint (ISO-639-1, e.g. 'ur'). When omitted, "
            "the STT language / query script is used."
        ),
    ),
    top_k: int = Form(
        DEFAULT_TOP_K,
        description=f"Number of chunks to retrieve (1..{MAX_TOP_K}).",
    ),
) -> VoiceQueryResponse:
    """STT + guardrail + retrieval + grounded LLM answer + TTS metadata.
    Never fabricates an answer: it abstains when the retrieved context is
    insufficient. The ``tts`` block reports synthesis metadata only — the
    audio itself is streamed by ``POST /voice/query/audio``."""
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be between 1 and {MAX_TOP_K}.",
        )
    tmp_path: Optional[Path] = None
    try:
        tmp_path = save_upload_to_temp(audio)
        return get_query_pipeline().process_audio(
            tmp_path, language_hint=language, top_k=top_k
        )
    except HTTPException:
        raise
    except LLMProviderError as exc:
        logger.error("LLM provider failed for %r: %s", audio.filename, exc)
        raise HTTPException(
            status_code=502,
            detail="LLM answer generation is unavailable (provider error).",
        ) from exc
    except ValueError as exc:
        logger.warning("Could not decode audio %r: %s", audio.filename, exc)
        raise HTTPException(
            status_code=422, detail="Audio file could not be decoded (corrupt or invalid)."
        ) from exc
    except Exception as exc:
        logger.error("Voice query failed for %r: %s", audio.filename, exc)
        raise HTTPException(status_code=500, detail="Voice query pipeline failed.") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post(
    "/voice/query/audio",
    summary="Voice query returning the synthesized answer audio (audio/wav)",
)
def voice_query_audio(
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a)"),
    language: Optional[str] = Form(
        None,
        description=(
            "Optional language hint (ISO-639-1, e.g. 'ur'). When omitted, "
            "the STT language / query script is used."
        ),
    ),
    top_k: int = Form(
        DEFAULT_TOP_K,
        description=f"Number of chunks to retrieve (1..{MAX_TOP_K}).",
    ),
) -> FileResponse:
    """Full pipeline: STT -> guardrail -> retrieval -> grounded LLM answer ->
    TTS -> audio/wav. Never fabricates an answer; abstentions are spoken as
    their explicit uncertainty message. The generated WAV is a temporary file
    that is deleted as soon as the response has been sent."""
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=422,
            detail=f"top_k must be between 1 and {MAX_TOP_K}.",
        )
    tmp_path: Optional[Path] = None
    try:
        tmp_path = save_upload_to_temp(audio)
        pipeline = get_query_pipeline()
        response = pipeline.process_audio(
            tmp_path, language_hint=language, top_k=top_k, with_tts=False
        )
    except HTTPException:
        raise
    except PipelineStageError as exc:
        logger.error("Voice pipeline stage '%s' failed for %r: %s", exc.stage, audio.filename, exc)
        return _stage_error_response(exc)
    except LLMProviderError as exc:
        logger.error("LLM provider failed for %r: %s", audio.filename, exc)
        return JSONResponse(
            status_code=502,
            content={"detail": {"code": "llm_failed", "message": "Could not generate an answer."}},
        )
    except ValueError as exc:
        logger.warning("Could not decode audio %r: %s", audio.filename, exc)
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "stt_decode", "message": "Could not understand the audio."}},
        )
    except Exception as exc:
        logger.error("Voice query failed for %r: %s", audio.filename, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": {"code": "pipeline_failed", "message": "Voice query pipeline failed."}},
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if not response.guardrail.allowed:
        # Guardrail rejection — no answer, no audio. The user still sees their
        # transcript; internal guardrail rules are never surfaced.
        return JSONResponse(
            status_code=400,
            content={
                "detail": "I can't process that request.",
                "transcript": response.transcript,
            },
        )
    if response.generation is None or not response.generation.answer.strip():
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "pipeline_failed", "message": "No answer was generated."}},
        )

    def _tts_error_response(status: int, exc: Exception) -> JSONResponse:
        # TTS failure must not lose the generated answer: the frontend falls
        # back to showing the text answer when audio cannot be synthesized.
        generation = response.generation
        detail = {
            "code": "tts_failed",
            "message": "Audio synthesis failed.",
            "answer": generation.answer,
            "transcript": response.transcript,
            "language": audio_language,
            "status": generation.status,
            "source": generation.source,
            "confidence": generation.confidence,
            "grounded": generation.grounded,
            "abstained": generation.abstained,
            "sources": [s.model_dump() for s in generation.sources],
            "model": generation.model,
        }
        return JSONResponse(status_code=status, content={"detail": detail})

    answer = response.generation.answer
    audio_language = response.language
    if response.tts is not None:
        audio_language = response.tts.language
    try:
        result = pipeline.tts.synthesize(answer, language=audio_language)
    except UnsupportedLanguageError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "tts_language", "message": str(exc)}},
        )
    except SynthesisError as exc:
        logger.error("TTS synthesis failed for %r: %s", audio.filename, exc)
        return _tts_error_response(502, exc)
    except Exception as exc:
        logger.error("TTS synthesis failed for %r: %s", audio.filename, exc)
        return _tts_error_response(500, exc)

    audio_path = Path(result.audio_path)
    meta = {
        "transcript": response.transcript,
        "answer": answer,
        "language": audio_language,
        "status": response.generation.status,
        "source": response.generation.source,
        "confidence": response.generation.confidence,
        "grounded": response.generation.grounded,
        "abstained": response.generation.abstained,
        "guardrail_allowed": response.guardrail.allowed,
        "model": response.generation.model,
        "sources": [s.model_dump() for s in response.generation.sources],
        "timings": {
            "stt_ms": response.timings.stt_ms,
            "retrieval_ms": response.timings.retrieval_ms,
            "llm_ms": response.timings.llm_ms,
            "grounding_ms": response.timings.grounding_ms,
            "tts_ms": result.processing_time_ms,
            "total_ms": response.timings.total_ms + result.processing_time_ms,
        },
    }
    encoded = base64.urlsafe_b64encode(json.dumps(meta).encode("utf-8")).decode("ascii")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename="answer.wav",
        headers={"X-Voice-RAG-Meta": encoded},
        background=BackgroundTask(audio_path.unlink, missing_ok=True),
    )


if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")
