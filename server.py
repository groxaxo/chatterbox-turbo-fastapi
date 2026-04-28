import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import anyio
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from chatterbox.tts_turbo import ChatterboxTurboTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("chatterbox_api")


# -----------------------------
# Runtime config
# -----------------------------

DEVICE = os.getenv("DEVICE", "cuda")
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1") == "1"
EXPECTED_GPU_NAME = os.getenv("EXPECTED_GPU_NAME", "").strip()

VOICE_DIR = Path(os.getenv("VOICE_DIR", "./voices")).resolve()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "").strip()
TMP_DIR = Path(os.getenv("TMP_DIR", tempfile.gettempdir())).resolve()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", os.getenv("MAX_CHUNK_CHARS", "360")))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
VOICE_CACHE_SIZE = int(os.getenv("VOICE_CACHE_SIZE", "8"))

AUTO_CHUNK_ENABLED = os.getenv("AUTO_CHUNK_ENABLED", "1") == "1"
AUTO_CHUNK_TARGET_CHARS = int(os.getenv("AUTO_CHUNK_TARGET_CHARS", str(MAX_TEXT_CHARS)))
AUTO_CHUNK_HARD_LIMIT = int(os.getenv("AUTO_CHUNK_HARD_LIMIT", str(max(MAX_TEXT_CHARS, 420))))
# Sentences shorter than this are merged forward rather than emitted as standalone chunks
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))
CHUNK_PAUSE_MS = int(os.getenv("CHUNK_PAUSE_MS", "140"))
DEFAULT_RESPONSE_FORMAT = os.getenv("DEFAULT_RESPONSE_FORMAT", "mp3").strip().lower()

WARMUP_TEXT = os.getenv("WARMUP_TEXT", "Warmup complete. [chuckle]")
STARTUP_WARMUP = os.getenv("STARTUP_WARMUP", "0") == "1"
LAZY_LOAD_MODEL = os.getenv("LAZY_LOAD_MODEL", "1") == "1"
MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "900"))
MODEL_IDLE_CHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_IDLE_CHECK_INTERVAL_SECONDS", "30"))

ENABLE_CELERY = os.getenv("ENABLE_CELERY", "0") == "1"
CELERY_TASK_TIMEOUT_SECONDS = int(os.getenv("CELERY_TASK_TIMEOUT_SECONDS", "300"))
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/14")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_QUEUE = os.getenv("CELERY_QUEUE", "chatterbox_tts")

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "1000"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.2"))
DEFAULT_NORM_LOUDNESS = os.getenv("DEFAULT_NORM_LOUDNESS", "1") == "1"

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SUPPORTED_TTS_MODELS = {"tts-1", "tts-1-hd"}
DEFAULT_VOICE_ALIASES = {"default", "alloy"}
SUPPORTED_RESPONSE_FORMATS = {"wav", "mp3", "json_base64"}
WORKER_GPU_INDICES = os.getenv("WORKER_GPU_INDICES", "2,3").strip()


# -----------------------------
# Global state
# -----------------------------

model: Optional[ChatterboxTurboTTS] = None
model_lock = threading.Lock()

# key -> Conditionals object from chatterbox.tts_turbo
conditionals_cache: "OrderedDict[str, Any]" = OrderedDict()
voice_digest_cache: "OrderedDict[str, tuple[tuple[int, int, int, int], str]]" = OrderedDict()

device_configured = False
model_last_used_monotonic = 0.0
idle_monitor_started = False


# -----------------------------
# Auth
# -----------------------------

async def require_api_key(request: Request) -> None:
    return


# -----------------------------
# Utility
# -----------------------------

def ensure_dirs() -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def configure_torch() -> str:
    global DEVICE
    global device_configured

    if device_configured:
        return DEVICE

    torch.set_grad_enabled(False)

    if DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            if REQUIRE_CUDA:
                raise RuntimeError("DEVICE=cuda requested, but torch.cuda.is_available() is false.")
            DEVICE = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            if EXPECTED_GPU_NAME and EXPECTED_GPU_NAME.lower() not in gpu_name.lower():
                raise RuntimeError(
                    f"Selected GPU '{gpu_name}' does not match EXPECTED_GPU_NAME='{EXPECTED_GPU_NAME}'."
                )

            # RTX 3060 / 3090 are Ampere. TF32 improves matmul/conv speed with negligible TTS impact.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    device_configured = True
    return DEVICE


def touch_model_usage() -> None:
    global model_last_used_monotonic
    model_last_used_monotonic = time.monotonic()


def resolve_default_voice_path() -> Optional[Path]:
    candidates: list[Path] = []
    if DEFAULT_VOICE:
        candidates.append(Path(DEFAULT_VOICE).resolve())
    candidates.append((VOICE_DIR / "default.wav").resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def unload_model_locked(reason: str) -> None:
    global model

    if model is None:
        return

    logger.info("Unloading Chatterbox model (%s).", reason)
    model = None
    conditionals_cache.clear()
    voice_digest_cache.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def idle_unload_loop() -> None:
    while True:
        time.sleep(max(5, MODEL_IDLE_CHECK_INTERVAL_SECONDS))
        if MODEL_IDLE_UNLOAD_SECONDS <= 0:
            continue

        idle_for = time.monotonic() - model_last_used_monotonic
        if model is None or idle_for < MODEL_IDLE_UNLOAD_SECONDS:
            continue

        with model_lock:
            idle_for = time.monotonic() - model_last_used_monotonic
            if model is not None and idle_for >= MODEL_IDLE_UNLOAD_SECONDS:
                unload_model_locked(f"idle for {int(idle_for)}s")


def ensure_idle_monitor_started() -> None:
    global idle_monitor_started

    if idle_monitor_started or MODEL_IDLE_UNLOAD_SECONDS <= 0:
        return

    threading.Thread(target=idle_unload_loop, name="chatterbox-idle-unloader", daemon=True).start()
    idle_monitor_started = True


def ensure_model_loaded_locked() -> ChatterboxTurboTTS:
    global model

    if model is not None:
        touch_model_usage()
        return model

    final_device = configure_torch()
    logger.info("Loading Chatterbox Turbo on %s...", final_device)
    model = ChatterboxTurboTTS.from_pretrained(device=final_device)

    default_path = resolve_default_voice_path()
    if default_path is not None:
        logger.info("Preparing default voice conditionals: %s", default_path)
        default_conds, _ = get_or_prepare_conditionals(default_path, norm_loudness=DEFAULT_NORM_LOUDNESS)
        model.conds = default_conds
    else:
        logger.warning("No default voice reference file found. Requests must provide a voice file.")

    touch_model_usage()
    ensure_idle_monitor_started()
    return model


def set_seed(seed: int) -> None:
    if seed <= 0:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_generation_args(
    *,
    text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Max input size is {MAX_INPUT_CHARS} chars.",
        )

    if not 0.05 <= temperature <= 2.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0.05 and 2.0.")

    if not 0.0 <= top_p <= 1.0:
        raise HTTPException(status_code=400, detail="top_p must be between 0.0 and 1.0.")

    if not 0 <= int(top_k) <= 1000:
        raise HTTPException(status_code=400, detail="top_k must be between 0 and 1000.")

    if not 1.0 <= repetition_penalty <= 2.0:
        raise HTTPException(status_code=400, detail="repetition_penalty must be between 1.0 and 2.0.")


def normalize_voice_path(voice: Optional[str]) -> Optional[Path]:
    """
    JSON endpoint voice resolution.
    - empty/default/alloy -> DEFAULT_VOICE (or voices/default.wav fallback)
    - plain filename -> VOICE_DIR / filename
    - absolute path -> only when ALLOW_ABSOLUTE_VOICE_PATHS=1
    """
    if not voice or voice in DEFAULT_VOICE_ALIASES:
        return resolve_default_voice_path()

    raw = voice.strip()
    if raw in DEFAULT_VOICE_ALIASES:
        return resolve_default_voice_path()

    if os.path.isabs(raw):
        if os.getenv("ALLOW_ABSOLUTE_VOICE_PATHS", "0") != "1":
            raise HTTPException(status_code=400, detail="Absolute voice paths are disabled.")
        p = Path(raw).resolve()
    else:
        # Reject anything containing a directory separator (all platforms).
        bad_seps = {"/", "\\", os.sep, os.altsep}
        if any(sep and sep in raw for sep in bad_seps):
            raise HTTPException(status_code=400, detail="Voice name must be a plain filename, not a path.")
        p = (VOICE_DIR / raw).resolve()
        if not p.is_relative_to(VOICE_DIR.resolve()):
            raise HTTPException(status_code=404, detail=f"Voice file not found: {voice}")

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"Voice file not found: {voice}")

    if p.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported voice file extension: {p.suffix}")

    return p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_file_sha256(path: Path) -> str:
    st = path.stat()
    signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    cache_key = str(path)
    cached = voice_digest_cache.get(cache_key)

    if cached is not None:
        cached_signature, digest = cached
        if cached_signature == signature:
            voice_digest_cache.move_to_end(cache_key)
            return digest

    digest = file_sha256(path)
    voice_digest_cache[cache_key] = (signature, digest)
    voice_digest_cache.move_to_end(cache_key)

    while len(voice_digest_cache) > max(32, VOICE_CACHE_SIZE * 4):
        voice_digest_cache.popitem(last=False)

    return digest


async def save_upload(upload: UploadFile) -> Path:
    data = await upload.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max is {MAX_UPLOAD_MB} MB.")

    suffix = Path(upload.filename or "voice.wav").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice file extension '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}",
        )

    ensure_dirs()
    out_path = TMP_DIR / f"chatterbox_voice_{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(data)
    return out_path


def cache_key_for_voice(path: Path, norm_loudness: bool) -> str:
    return f"{cached_file_sha256(path)}:norm={int(norm_loudness)}"


def get_or_prepare_conditionals(voice_path: Path, norm_loudness: bool) -> tuple[Any, bool]:
    """
    ChatterboxTurboTTS.generate(audio_prompt_path=...) calls prepare_conditionals every time.
    We instead prepare once, cache model.conds, then call generate(..., audio_prompt_path=None).
    """
    if model is None:
        raise RuntimeError("Model is not loaded.")

    key = cache_key_for_voice(voice_path, norm_loudness)
    cached = conditionals_cache.get(key)

    if cached is not None:
        conditionals_cache.move_to_end(key)
        return cached, True

    model.prepare_conditionals(str(voice_path), norm_loudness=norm_loudness)
    conds = model.conds

    conditionals_cache[key] = conds
    conditionals_cache.move_to_end(key)

    while len(conditionals_cache) > max(1, VOICE_CACHE_SIZE):
        conditionals_cache.popitem(last=False)

    return conds, False


def tensor_to_float_array(wav: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach()
        if wav.device.type != "cpu":
            wav = wav.cpu()
        if not torch.is_floating_point(wav):
            wav = wav.float()
    else:
        wav = torch.as_tensor(wav, dtype=torch.float32)

    if wav.ndim == 2 and wav.shape[0] == 1:
        arr = wav.squeeze(0).numpy()
    elif wav.ndim == 1:
        arr = wav.numpy()
    else:
        arr = wav.squeeze().numpy()

    return np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)


def wav_bytes_from_array(arr: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def mp3_bytes_from_array(arr: np.ndarray, sample_rate: int) -> bytes:
    wav_bytes = wav_bytes_from_array(arr, sample_rate)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "mp3",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, input=wav_bytes, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for MP3 output but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to encode MP3 output: {exc.stderr.decode('utf-8', errors='ignore')}") from exc
    return result.stdout


def encode_audio_bytes(arr: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str, str]:
    if output_format == "wav":
        return wav_bytes_from_array(arr, sample_rate), "audio/wav", "wav"
    if output_format == "mp3":
        return mp3_bytes_from_array(arr, sample_rate), "audio/mpeg", "mp3"
    raise RuntimeError(f"Unsupported output format: {output_format}")


def split_oversized_unit(text: str, hard_limit: int) -> list[str]:
    text = text.strip()
    if len(text) <= hard_limit:
        return [text]

    pieces: list[str] = []
    current = ""
    for clause in re.split(r"(?<=[,;:])\s+", text):
        clause = clause.strip()
        if not clause:
            continue
        candidate = clause if not current else f"{current} {clause}"
        if current and len(candidate) > hard_limit:
            pieces.extend(split_oversized_unit(current, hard_limit))
            current = clause
        else:
            current = candidate

    if current:
        if len(current) <= hard_limit:
            pieces.append(current)
        else:
            words = current.split()
            bucket = ""
            for word in words:
                candidate = word if not bucket else f"{bucket} {word}"
                if bucket and len(candidate) > hard_limit:
                    pieces.append(bucket)
                    bucket = word
                else:
                    bucket = candidate
            if bucket:
                pieces.append(bucket)

    return [piece for piece in pieces if piece]


def chunk_text_for_tts(text: str) -> list[str]:
    """
    Split text into synthesis chunks at sentence boundaries (. ! ? and newlines).

    Strategy:
    - Every sentence ending is a natural chunk boundary.
    - Sentences shorter than MIN_CHUNK_CHARS are merged forward with the next sentence
      so we avoid submitting very short standalone clips.
    - Sentences longer than AUTO_CHUNK_HARD_LIMIT are sub-split via split_oversized_unit.
    - A backward-merge post-pass ensures a short trailing chunk is folded into the
      previous chunk rather than being emitted as a tiny standalone clip.

    With Celery, each chunk is dispatched as a separate task so they run in parallel
    across available GPU workers.
    """
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    if not cleaned:
        return []

    if not AUTO_CHUNK_ENABLED:
        return [cleaned]

    # Split on sentence-ending punctuation or paragraph breaks
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    if not raw_sentences:
        return [cleaned]

    # Sub-split any sentence that exceeds the hard limit
    units: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) > AUTO_CHUNK_HARD_LIMIT:
            units.extend(split_oversized_unit(sentence, AUTO_CHUNK_HARD_LIMIT))
        else:
            units.append(sentence)

    # Emit one chunk per sentence; merge short leading fragments forward
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        merged = f"{current} {unit}"
        if len(merged) > AUTO_CHUNK_HARD_LIMIT:
            # Merging would overflow — emit current, start fresh
            chunks.append(current)
            current = unit
        elif len(current) < MIN_CHUNK_CHARS:
            # Too short to stand alone — merge with next
            current = merged
        else:
            # Long enough to be its own chunk
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    # Backward-merge pass: fold a short trailing chunk into the previous one
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if len(merged) <= AUTO_CHUNK_HARD_LIMIT:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def concatenate_audio_arrays(chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    pause_samples = max(0, int(sample_rate * (CHUNK_PAUSE_MS / 1000.0)))
    silence = np.zeros(pause_samples, dtype=np.float32)

    merged: list[np.ndarray] = []
    for idx, chunk in enumerate(chunks):
        if idx > 0 and pause_samples > 0:
            merged.append(silence)
        merged.append(chunk.astype(np.float32, copy=False))

    return np.concatenate(merged) if len(merged) > 1 else merged[0]


def generate_chunk_locked(
    *,
    text: str,
    voice_path: Optional[Path],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
    seed: int,
) -> tuple[np.ndarray, int, bool]:
    with model_lock:
        loaded_model = ensure_model_loaded_locked()
        set_seed(seed)

        voice_cache_hit = False
        if voice_path is not None:
            conds, voice_cache_hit = get_or_prepare_conditionals(voice_path, norm_loudness=norm_loudness)
            loaded_model.conds = conds
        elif loaded_model.conds is None:
            raise RuntimeError("No default voice conditionals are loaded. Provide a voice file or set DEFAULT_VOICE.")

        with torch.inference_mode():
            wav = loaded_model.generate(
                text.strip(),
                audio_prompt_path=None,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                norm_loudness=bool(norm_loudness),
            )

        sample_rate = int(loaded_model.sr)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()

        touch_model_usage()
        return tensor_to_float_array(wav), sample_rate, voice_cache_hit


def synthesize_request_sync(
    *,
    text: str,
    voice_path: Optional[Path],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
    seed: int,
    output_format: str,
) -> tuple[bytes, dict[str, Any]]:
    started = time.perf_counter()
    chunks = chunk_text_for_tts(text)

    if not chunks:
        raise RuntimeError("No text chunks were produced.")

    sample_rate: Optional[int] = None
    arrays: list[np.ndarray] = []
    voice_cache_hits: list[bool] = []

    for idx, chunk in enumerate(chunks):
        chunk_seed = (seed + idx) if seed > 0 else 0
        arr, chunk_sample_rate, voice_cache_hit = generate_chunk_locked(
            text=chunk,
            voice_path=voice_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=chunk_seed,
        )
        if sample_rate is None:
            sample_rate = chunk_sample_rate
        elif sample_rate != chunk_sample_rate:
            raise RuntimeError("Mismatched sample rates while concatenating synthesized chunks.")

        arrays.append(arr)
        voice_cache_hits.append(voice_cache_hit)

    assert sample_rate is not None
    combined = concatenate_audio_arrays(arrays, sample_rate)
    audio_bytes, content_type, extension = encode_audio_bytes(combined, sample_rate, output_format)

    elapsed = time.perf_counter() - started
    audio_seconds = float(combined.shape[-1] / sample_rate) if sample_rate > 0 else 0.0

    meta = {
        "sample_rate": sample_rate,
        "wall_seconds": round(elapsed, 4),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(elapsed / audio_seconds, 4) if audio_seconds > 0 else None,
        "x_realtime": round(audio_seconds / elapsed, 4) if elapsed > 0 else None,
        "device": DEVICE,
        "voice_cached": any(voice_cache_hits),
        "cache_entries": len(conditionals_cache),
        "chunk_count": len(chunks),
        "chunk_target_chars": AUTO_CHUNK_TARGET_CHARS,
        "chunk_hard_limit": AUTO_CHUNK_HARD_LIMIT,
        "lazy_loaded": LAZY_LOAD_MODEL,
        "model_loaded": model is not None,
        "output_format": output_format,
        "content_type": content_type,
        "filename": f"speech.{extension}",
    }

    return audio_bytes, meta


def synthesize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    voice_path = Path(payload["voice_path"]).resolve() if payload.get("voice_path") else None
    wav_bytes, meta = synthesize_request_sync(
        text=payload["text"],
        voice_path=voice_path,
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
        repetition_penalty=float(payload["repetition_penalty"]),
        norm_loudness=bool(payload["norm_loudness"]),
        seed=int(payload["seed"]),
        output_format=str(payload.get("output_format", DEFAULT_RESPONSE_FORMAT)),
    )
    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "metadata": meta,
    }


def synthesize_single_chunk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Synthesize a single pre-split text chunk.  No further chunking is applied.
    Always returns lossless WAV as base64 so the API server can stitch chunks
    before encoding to the client-requested format.
    """
    voice_path = Path(payload["voice_path"]).resolve() if payload.get("voice_path") else None
    arr, sample_rate, voice_cache_hit = generate_chunk_locked(
        text=payload["text"],
        voice_path=voice_path,
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
        repetition_penalty=float(payload["repetition_penalty"]),
        norm_loudness=bool(payload["norm_loudness"]),
        seed=int(payload["seed"]),
    )
    wav_bytes = wav_bytes_from_array(arr, sample_rate)
    return {
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "metadata": {
            "sample_rate": sample_rate,
            "voice_cached": voice_cache_hit,
        },
    }


def response_with_audio(audio_bytes: bytes, meta: dict[str, Any]) -> Response:
    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{meta["filename"]}"',
        "X-Sample-Rate": str(meta["sample_rate"]),
        "X-Wall-Seconds": str(meta["wall_seconds"]),
        "X-Audio-Seconds": str(meta["audio_seconds"]),
        "X-Voice-Cached": str(meta["voice_cached"]).lower(),
        "X-Voice-Cache-Entries": str(meta["cache_entries"]),
        "X-Chunk-Count": str(meta["chunk_count"]),
        "X-Chunk-Target-Chars": str(meta["chunk_target_chars"]),
        "X-Output-Format": str(meta["output_format"]),
    }
    if meta.get("rtf") is not None:
        headers["X-RTF"] = str(meta["rtf"])
    if meta.get("x_realtime") is not None:
        headers["X-Speed-X-Realtime"] = str(meta["x_realtime"])
    return Response(content=audio_bytes, media_type=meta["content_type"], headers=headers)


def available_models_payload() -> dict[str, list[dict[str, str]]]:
    return {
        "models": [
            {"id": "tts-1", "name": "Chatterbox Turbo (standard)"},
            {"id": "tts-1-hd", "name": "Chatterbox Turbo (hd alias)"},
        ]
    }


def available_voices_payload() -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    voices = [
        {"id": "alloy", "name": "alloy (default reference voice)"},
        {"id": "default", "name": "default"},
    ]
    for path in sorted(VOICE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO_SUFFIXES:
            voices.append({"id": path.name, "name": path.name})
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for voice in voices:
        if voice["id"] in seen:
            continue
        seen.add(voice["id"])
        deduped.append(voice)
    return {"voices": deduped}


def runtime_status(include_sensitive: bool = True) -> dict[str, Any]:
    default_voice_path = resolve_default_voice_path()
    gpu = None

    if include_sensitive and DEVICE.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            gpu = {
                "name": torch.cuda.get_device_name(idx),
                "index": idx,
                "total_vram_gb": round(props.total_memory / 1024**3, 2),
                "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
                "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
            }
        except Exception:
            gpu = None

    return {
        "ok": True,
        "backend_mode": "celery" if ENABLE_CELERY else "direct",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "expected_gpu_name": EXPECTED_GPU_NAME or ("RTX 3060" if ENABLE_CELERY else None),
        "model_loaded": model is not None,
        "lazy_load_model": LAZY_LOAD_MODEL,
        "model_idle_unload_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "sample_rate": int(model.sr) if model else 24000,
        "default_voice": str(default_voice_path) if default_voice_path else None,
        "voice_dir": str(VOICE_DIR),
        "max_input_chars": MAX_INPUT_CHARS,
        "chunk_target_chars": AUTO_CHUNK_TARGET_CHARS,
        "chunk_hard_limit": AUTO_CHUNK_HARD_LIMIT,
        "auto_chunk_enabled": AUTO_CHUNK_ENABLED,
        "default_response_format": DEFAULT_RESPONSE_FORMAT,
        "worker_gpu_indices": WORKER_GPU_INDICES or None,
        "voice_cache_entries": len(conditionals_cache),
        "voice_cache_size": VOICE_CACHE_SIZE,
        "celery_queue": CELERY_QUEUE if ENABLE_CELERY else None,
    }


def wait_for_celery_result(task) -> tuple[bytes, dict[str, Any]]:
    deadline = time.monotonic() + CELERY_TASK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if task.ready():
            break
        time.sleep(0.1)
    else:
        raise HTTPException(status_code=504, detail="Timed out waiting for Celery synthesis task.")

    if task.failed():
        raise RuntimeError(str(task.result))

    payload = task.result
    if not isinstance(payload, dict):
        raise RuntimeError("Celery synthesis task returned an unexpected payload type.")

    audio_base64 = payload.get("audio_base64")
    if not audio_base64:
        raise RuntimeError("Celery synthesis task returned no audio payload.")

    return base64.b64decode(audio_base64), payload["metadata"]


def wait_for_celery_chunks_parallel(
    tasks: list,
) -> list[tuple[np.ndarray, int]]:
    """
    Wait for all chunk tasks in parallel.  Returns a list of (array, sample_rate)
    tuples in original chunk order.

    Fail-fast: on any single task failure or timeout the remaining tasks are
    revoked and an HTTPException is raised.
    """
    deadline = time.monotonic() + CELERY_TASK_TIMEOUT_SECONDS
    n = len(tasks)
    done: dict[int, Any] = {}

    while time.monotonic() < deadline:
        pending = [i for i in range(n) if i not in done]
        if not pending:
            break
        for i in pending:
            if tasks[i].ready():
                done[i] = tasks[i]
        if len(done) < n:
            time.sleep(0.05)
    else:
        # Timeout — revoke remaining tasks
        for i in range(n):
            if i not in done:
                try:
                    tasks[i].revoke(terminate=True)
                except Exception:
                    pass
        raise HTTPException(
            status_code=504,
            detail=f"Timed out waiting for {n - len(done)}/{n} Celery chunk tasks.",
        )

    # Check for failures
    for i in range(n):
        if tasks[i].failed():
            err = str(tasks[i].result)
            for j in range(n):
                if j != i:
                    try:
                        tasks[j].revoke(terminate=True)
                    except Exception:
                        pass
            raise HTTPException(status_code=500, detail=f"Chunk {i} synthesis failed: {err}")

    # Decode results in order and verify sample-rate consistency
    results: list[tuple[np.ndarray, int]] = []
    expected_sr: Optional[int] = None
    for i in range(n):
        payload = tasks[i].result
        if not isinstance(payload, dict) or not payload.get("audio_base64"):
            raise HTTPException(status_code=500, detail=f"Chunk {i} returned an empty payload.")
        wav_bytes = base64.b64decode(payload["audio_base64"])
        arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
        if expected_sr is None:
            expected_sr = sr
        elif sr != expected_sr:
            logger.warning("Sample-rate mismatch across chunks: %d vs %d — resampling ignored.", sr, expected_sr)
        results.append((arr, sr))

    return results


# -----------------------------
# Lifespan
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    ensure_idle_monitor_started()

    if not LAZY_LOAD_MODEL and not ENABLE_CELERY:
        with model_lock:
            ensure_model_loaded_locked()

        if STARTUP_WARMUP:
            logger.info("Running startup warmup...")
            try:
                synthesize_request_sync(
                    text=WARMUP_TEXT,
                    voice_path=resolve_default_voice_path(),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    top_k=DEFAULT_TOP_K,
                    repetition_penalty=DEFAULT_REPETITION_PENALTY,
                    norm_loudness=DEFAULT_NORM_LOUDNESS,
                    seed=0,
                )
                logger.info("Warmup complete.")
            except Exception as exc:
                logger.warning("Warmup skipped/failed: %s: %s", type(exc).__name__, exc)

    yield

    with model_lock:
        unload_model_locked("shutdown")


app = FastAPI(
    title="Chatterbox Turbo FastAPI",
    version="3.0.0",
    lifespan=lifespan,
)


# -----------------------------
# API models
# -----------------------------

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    model: str = Field("tts-1", description="tts-1 or tts-1-hd")
    voice: str = Field("alloy", description="'alloy', 'default', or a filename inside VOICE_DIR")
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    norm_loudness: bool = DEFAULT_NORM_LOUDNESS
    seed: int = 0
    response_format: str = Field(DEFAULT_RESPONSE_FORMAT, description="mp3, wav, or json_base64")


async def run_generation(
    *,
    text: str,
    voice_path: Optional[Path],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
    seed: int,
    output_format: str,
) -> tuple[bytes, dict[str, Any]]:
    if ENABLE_CELERY:
        from celery_worker import celery_app

        chunks = chunk_text_for_tts(text)
        chunk_count = len(chunks)

        if chunk_count == 0:
            raise HTTPException(status_code=400, detail="No text to synthesize after cleaning.")

        base_payload = {
            "voice_path": str(voice_path) if voice_path else None,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "norm_loudness": norm_loudness,
            "seed": seed,
        }

        if chunk_count == 1:
            # Single chunk — use the original task so the worker handles format encoding
            payload = {**base_payload, "text": chunks[0], "output_format": output_format}
            task = celery_app.send_task(
                "chatterbox_turbo.synthesize",
                kwargs={"payload": payload},
                queue=CELERY_QUEUE,
            )
            return await anyio.to_thread.run_sync(
                lambda: wait_for_celery_result(task), abandon_on_cancel=False
            )

        # Multiple chunks — dispatch all in parallel, stitch on the API server
        logger.info("Parallel dispatch: %d chunks for %d chars", chunk_count, len(text))
        tasks = [
            celery_app.send_task(
                "chatterbox_turbo.synthesize_chunk",
                kwargs={"payload": {**base_payload, "text": chunk}},
                queue=CELERY_QUEUE,
            )
            for chunk in chunks
        ]

        chunk_results: list[tuple[np.ndarray, int]] = await anyio.to_thread.run_sync(
            lambda: wait_for_celery_chunks_parallel(tasks), abandon_on_cancel=False
        )

        t_start = time.monotonic()

        # Stitch chunks with a short silence between each
        sample_rate = chunk_results[0][1]
        arrays = [arr for arr, _sr in chunk_results]
        stitched = concatenate_audio_arrays(arrays, sample_rate)

        audio_seconds = round(len(stitched) / sample_rate, 4)
        wall_seconds = round(time.monotonic() - t_start, 4)
        extension = "mp3" if output_format == "mp3" else "wav"
        content_type = "audio/mpeg" if output_format == "mp3" else "audio/wav"

        if output_format == "mp3":
            wav_tmp = wav_bytes_from_array(stitched, sample_rate)
            final_bytes = encode_audio_bytes(wav_tmp, "mp3")
        elif output_format == "json_base64":
            wav_bytes_out = wav_bytes_from_array(stitched, sample_rate)
            final_bytes = json.dumps(
                {"audio_base64": base64.b64encode(wav_bytes_out).decode("ascii")}
            ).encode()
            extension = "json"
            content_type = "application/json"
        else:
            final_bytes = wav_bytes_from_array(stitched, sample_rate)

        meta = {
            "chunk_count": chunk_count,
            "sample_rate": sample_rate,
            "voice_cached": False,
            "cache_entries": len(conditionals_cache),
            "wall_seconds": wall_seconds,
            "audio_seconds": audio_seconds,
            "rtf": round(wall_seconds / audio_seconds, 4) if audio_seconds > 0 else None,
            "x_realtime": round(audio_seconds / wall_seconds, 4) if wall_seconds > 0 else None,
            "chunk_target_chars": AUTO_CHUNK_TARGET_CHARS,
            "output_format": output_format,
            "content_type": content_type,
            "filename": f"speech.{extension}",
        }

        return final_bytes, meta

    return await anyio.to_thread.run_sync(
        lambda: synthesize_request_sync(
            text=text,
            voice_path=voice_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
            output_format=output_format,
        ),
        abandon_on_cancel=False,
    )


# -----------------------------
# Routes
# -----------------------------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "backend_mode": "celery" if ENABLE_CELERY else "direct",
        "model_loaded": model is not None,
    }


@app.get("/status", dependencies=[Depends(require_api_key)])
def status() -> dict[str, Any]:
    return runtime_status(include_sensitive=True)


@app.get("/health")
def health() -> dict[str, Any]:
    return healthz()


@app.get("/voices", dependencies=[Depends(require_api_key)])
def voices() -> dict[str, list[dict[str, str]]]:
    return available_voices_payload()


@app.get("/audio/models", dependencies=[Depends(require_api_key)])
@app.get("/v1/audio/models", dependencies=[Depends(require_api_key)])
def audio_models() -> dict[str, list[dict[str, str]]]:
    return available_models_payload()


@app.get("/audio/voices", dependencies=[Depends(require_api_key)])
@app.get("/v1/audio/voices", dependencies=[Depends(require_api_key)])
def audio_voices() -> dict[str, list[dict[str, str]]]:
    return available_voices_payload()


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def openai_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "owned_by": "groxaxo"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "groxaxo"},
        ],
    }


@app.post("/tts", dependencies=[Depends(require_api_key)])
async def tts_multipart(
    text: str = Form(...),
    voice: UploadFile | None = File(default=None),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    top_p: float = Form(DEFAULT_TOP_P),
    top_k: int = Form(DEFAULT_TOP_K),
    repetition_penalty: float = Form(DEFAULT_REPETITION_PENALTY),
    norm_loudness: bool = Form(DEFAULT_NORM_LOUDNESS),
    seed: int = Form(0),
    response_format: str = Form(DEFAULT_RESPONSE_FORMAT),
):
    validate_generation_args(
        text=text,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    if response_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'mp3'.")

    uploaded_path: Optional[Path] = None

    try:
        if voice is not None:
            uploaded_path = await save_upload(voice)
            voice_path = uploaded_path
        else:
            voice_path = normalize_voice_path("alloy")

        wav_bytes, meta = await run_generation(
            text=text,
            voice_path=voice_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
            output_format=response_format,
        )

        return response_with_audio(wav_bytes, meta)

    except HTTPException:
        raise
    except AssertionError:
        logger.exception("TTS assertion failed (multipart)")
        raise HTTPException(status_code=400, detail="Invalid TTS request.")
    except Exception:
        logger.exception("TTS generation failed (multipart)")
        raise HTTPException(status_code=500, detail="TTS failed. Check server logs.")
    finally:
        if uploaded_path is not None:
            uploaded_path.unlink(missing_ok=True)


@app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
async def openai_style_speech(req: SpeechRequest):
    validate_generation_args(
        text=req.input,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
    )

    if req.model not in SUPPORTED_TTS_MODELS:
        raise HTTPException(status_code=400, detail=f"model must be one of: {', '.join(sorted(SUPPORTED_TTS_MODELS))}")

    if req.response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"response_format must be one of: {', '.join(sorted(SUPPORTED_RESPONSE_FORMATS))}.",
        )

    voice_path = normalize_voice_path(req.voice)

    try:
        wav_bytes, meta = await run_generation(
            text=req.input,
            voice_path=voice_path,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            norm_loudness=req.norm_loudness,
            seed=req.seed,
            output_format="mp3" if req.response_format == "json_base64" else req.response_format,
        )

        if req.response_format == "json_base64":
            return JSONResponse(
                {
                    "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                    "content_type": "audio/wav",
                    "metadata": meta,
                }
            )

        return response_with_audio(wav_bytes, meta)

    except HTTPException:
        raise
    except AssertionError:
        logger.exception("TTS assertion failed (OpenAI speech endpoint)")
        raise HTTPException(status_code=400, detail="Invalid TTS request.")
    except Exception:
        logger.exception("TTS generation failed (OpenAI speech endpoint)")
        raise HTTPException(status_code=500, detail="TTS failed. Check server logs.")


@app.post("/warmup", dependencies=[Depends(require_api_key)])
async def warmup():
    try:
        voice_path = normalize_voice_path("alloy")
        wav_bytes, meta = await run_generation(
            text=WARMUP_TEXT,
            voice_path=voice_path,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            norm_loudness=DEFAULT_NORM_LOUDNESS,
            seed=0,
            output_format="wav",
        )
        return {"ok": True, "metadata": meta, "bytes": len(wav_bytes)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Warmup failed")
        raise HTTPException(status_code=500, detail="Warmup failed. Check server logs.")
