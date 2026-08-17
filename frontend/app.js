"use strict";

// Voice RAG frontend (Segment 5).
// Mic -> POST /voice/query/audio -> audio/wav response; metadata rides in the
// X-Voice-RAG-Meta response header (base64url JSON) so one request yields both
// the generated audio and the transcript / answer / grounding / timings.

const API_BASE = new URLSearchParams(window.location.search).get("api") || "";

// Browser recording formats, best first. The extension is derived from the
// format actually chosen so the backend can route it correctly (webm/ogg/m4a).
const PREFERRED_MIMES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

const byId = (id) => document.getElementById(id);
const micBtn = byId("mic-btn");
const playBtn = byId("play-btn");
const statusEl = byId("status");
const errorEl = byId("error");
const resultEl = byId("result");
const transcriptEl = byId("transcript");
const answerEl = byId("answer");
const groundingEl = byId("grounding");
const noticeEl = byId("notice");
const sourcesBlockEl = byId("sources-block");
const sourcesEl = byId("sources");
const timingsEl = byId("timings");
const langEl = byId("lang");
const fileInput = byId("file-input");

const STATE = {
  IDLE: "idle",
  RECORDING: "recording",
  PROCESSING: "processing",
  PLAYING: "playing",
  ERROR: "error",
};

let state = STATE.IDLE;
let mediaRecorder = null;
let chunks = [];
let lastAudioBlob = null;
let lastAudioUrl = null;
let lastMeta = null;
let requestInFlight = false;
let audioEl = new Audio();

function setState(next) {
  state = next;
  statusEl.className = "status " + next;
  const labels = {
    idle: "Status: Idle",
    recording: "🔴 Recording… speak now",
    processing: "⏳ Processing…",
    playing: "🔊 Playing response…",
    error: "⚠ Something went wrong",
  };
  statusEl.textContent = labels[next] || "Status: " + next;
  micBtn.textContent = next === STATE.RECORDING ? "🔴 Stop Recording" : "🎤 Start Recording";
  // Only one request at a time: the mic, upload fallback and language selector
  // are locked while a request is in flight (recording / processing / playing).
  micBtn.disabled = next === STATE.PROCESSING || next === STATE.PLAYING;
  playBtn.disabled = !lastAudioBlob || next === STATE.RECORDING || next === STATE.PROCESSING;
  fileInput.disabled = next !== STATE.IDLE;
  langEl.disabled = next !== STATE.IDLE;
  if (next !== STATE.ERROR) errorEl.hidden = true;
  if (next !== STATE.IDLE && next !== STATE.ERROR) noticeEl.hidden = true;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  setState(STATE.ERROR);
}

function pickMimeType() {
  if (!window.MediaRecorder) return null;
  for (const mime of PREFERRED_MIMES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  return null;
}

function extensionFor(mime) {
  if (!mime) return "webm";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4")) return "m4a";
  return "webm";
}

async function startRecording() {
  if (state !== STATE.IDLE || requestInFlight) return;
  errorEl.hidden = true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("This browser doesn't support microphone access.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err && err.name === "NotAllowedError") {
      showError("Microphone permission was denied.");
    } else if (err && err.name === "NotFoundError") {
      showError("No microphone was found.");
    } else {
      showError("Couldn't access the microphone: " + (err && err.name ? err.name : "unknown error"));
    }
    return;
  }

  const mime = pickMimeType();
  if (!mime && !window.MediaRecorder) {
    stream.getTracks().forEach((t) => t.stop());
    showError("Recording is not supported in this browser.");
    return;
  }

  try {
    mediaRecorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    showError("Couldn't start the recorder in this browser.");
    return;
  }

  chunks = [];
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const type = mediaRecorder.mimeType || "audio/webm";
    const blob = new Blob(chunks, { type });
    sendAudio(blob, extensionFor(type));
  };
  mediaRecorder.onerror = () => {
    stream.getTracks().forEach((t) => t.stop());
    showError("Recording failed.");
  };

  mediaRecorder.start();
  setState(STATE.RECORDING);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    setState(STATE.PROCESSING);
  }
}

function decodeMeta(header) {
  if (!header) return null;
  try {
    const json = atob(header.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch (err) {
    return null;
  }
}

function renderResult(meta) {
  resultEl.hidden = false;
  transcriptEl.textContent = meta.transcript || "(no transcript)";
  answerEl.textContent = meta.answer || "(no answer)";

  // Use the backend `source` field to pick the grounding label.  Falls back
  // to the legacy `grounded` boolean when `source` is missing (e.g. TTS-error
  // fallback).
  const src = meta.source;
  groundingEl.className = "grounding";
  groundingEl.classList.remove("ok", "info", "unverified", "not-reliable");
  if (src === "rag") {
    groundingEl.textContent = "✓ Answer grounded in retrieved context.";
    groundingEl.classList.add("ok");
  } else if (src === "general_knowledge") {
    groundingEl.textContent = "ℹ Answer generated from general knowledge.";
    groundingEl.classList.add("info");
  } else if (src === "clarification") {
    groundingEl.textContent = "↳ Clarification needed to answer reliably.";
    groundingEl.classList.add("unverified");
  } else if (src === "abstained") {
    groundingEl.textContent = "⚠ I couldn't answer reliably from the available information.";
    groundingEl.classList.add("not-reliable");
  } else {
    // Fallback: no source field — use legacy grounded boolean.
    const g = meta.grounded;
    if (g === true) {
      groundingEl.textContent = "✓ Answer grounded in retrieved context.";
      groundingEl.classList.add("ok");
    } else if (g === false) {
      groundingEl.textContent = "⚠ Answer could not be verified against retrieved context.";
      groundingEl.classList.add("not-reliable");
    } else {
      groundingEl.textContent = "⚠ Grounding could not be fully verified.";
      groundingEl.classList.add("unverified");
    }
  }

  // Only show RAG sources when the answer was actually grounded in retrieval.
  if (src === "rag") {
    renderSources(meta.sources);
  } else {
    sourcesBlockEl.hidden = true;
  }
  noticeEl.hidden = true;

  const t = (meta.timings || {});
  timingsEl.textContent =
    "Pipeline timings (measured):\nSTT " + ms(t.stt_ms) +
    " · Retrieval " + ms(t.retrieval_ms) +
    " · LLM " + ms(t.llm_ms) +
    " · TTS " + ms(t.tts_ms) +
    " · Total " + ms(t.total_ms);
}

function renderSources(list) {
  if (!Array.isArray(list) || list.length === 0) {
    sourcesBlockEl.hidden = true;
    return;
  }
  sourcesBlockEl.hidden = false;
  sourcesEl.textContent = "";
  list.forEach((s, i) => {
    const li = document.createElement("li");
    const excerpt = s.excerpt || s.id || "";
    li.textContent = (i + 1) + ". " + excerpt + " — relevance " + Number(s.score).toFixed(4);
    sourcesEl.appendChild(li);
  });
}

function showNotice(message) {
  noticeEl.textContent = message;
  noticeEl.hidden = false;
}

function ms(v) {
  return v == null ? "—" : (v / 1000).toFixed(1) + "s";
}

function playAudio(blob) {
  if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
  lastAudioBlob = blob;
  lastAudioUrl = URL.createObjectURL(blob);
  audioEl.src = lastAudioUrl;
  setState(STATE.PLAYING);
  audioEl.play().then(() => {
    audioEl.onended = () => setState(STATE.IDLE);
  }).catch(() => {
    audioEl.onended = null;
    setState(STATE.IDLE);
  });
}

async function sendAudio(blob, extension) {
  // One request at a time: ignore attempts to start a second request while the
  // recorder is still capturing (onstop) or another request is in flight.
  if (state === STATE.RECORDING || requestInFlight) return;
  requestInFlight = true;
  setState(STATE.PROCESSING);
  const form = new FormData();
  form.append("audio", blob, "recording." + extension);
  const language = langEl.value;
  if (language) form.append("language", language);
  form.append("top_k", "5");

  let resp;
  try {
    resp = await fetch(API_BASE + "/voice/query/audio", { method: "POST", body: form });
  } catch (err) {
    requestInFlight = false;
    showError("Backend is unavailable. Is it running?");
    return;
  }

  if (!resp.ok) {
    requestInFlight = false;
    let body = null;
    try { body = await resp.json(); } catch (err) { /* keep generic */ }
    const detail = (body && body.detail) || {};
    const code = typeof detail === "string" ? null : detail.code;
    const detailMessage = typeof detail === "string" ? detail : detail.message;

    // Guardrail rejection is user-friendly; internal rules are never shown.
    if (resp.status === 400) {
      showError("I can't process that request.");
      return;
    }

    // TTS failure with a generated answer: show the text answer instead of an
    // error, so a synthesis outage never loses the answer.
    if (code === "tts_failed" && detail.answer) {
      renderResult({
        transcript: detail.transcript || "",
        answer: detail.answer,
        grounded: detail.grounded,
        abstained: detail.abstained,
        sources: detail.sources || [],
        timings: detail.timings || {},
      });
      showNotice("⚠ Audio synthesis failed — showing the text answer.");
      setState(STATE.IDLE);
      return;
    }

    if (resp.status === 413) {
      showError("The audio file is too large.");
      return;
    }

    const stageMessages = {
      stt_decode: "Could not understand the audio.",
      stt_failed: "Could not understand the audio.",
      retrieval_failed: "Could not retrieve relevant information.",
      llm_failed: "Could not generate an answer.",
      tts_failed: "Audio synthesis failed.",
      pipeline_failed: "Something went wrong while processing the audio.",
    };
    showError(stageMessages[code] || detailMessage || "Request failed (HTTP " + resp.status + ").");
    return;
  }

  const contentType = resp.headers.get("content-type") || "";
  if (!contentType.startsWith("audio/")) {
    requestInFlight = false;
    showError("The backend returned an invalid audio response.");
    return;
  }

  const meta = decodeMeta(resp.headers.get("X-Voice-RAG-Meta")) || {
    transcript: "",
    answer: "",
    grounded: null,
    timings: {},
  };
  lastMeta = meta;
  renderResult(meta);
  const audio = await resp.blob();
  requestInFlight = false;
  playAudio(audio);
}

micBtn.addEventListener("click", () => {
  if (state === STATE.RECORDING) stopRecording();
  else if (state === STATE.IDLE) startRecording();
});

playBtn.addEventListener("click", () => {
  if (lastAudioBlob) playAudio(lastAudioBlob);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const ext = (file.name.split(".").pop() || "webm").toLowerCase();
  sendAudio(file, ext);
  fileInput.value = "";
});

setState(STATE.IDLE);
