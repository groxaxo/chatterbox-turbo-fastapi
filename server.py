import base64
import hashlib
import io
import logging
import os
import random
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

API_KEY = os.getenv("API_KEY", "").strip()

VOICE_DIR = Path(os.getenv("VOICE_DIR", "./voices")).resolve()
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "").strip()
TMP_DIR = Path(os.getenv("TMP_DIR", tempfile.gettempdir())).resolve()

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "700"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
VOICE_CACHE_SIZE = int(os.getenv("VOICE_CACHE_SIZE", "8"))

WARMUP_TEXT = os.getenv("WARMUP_TEXT", "Warmup complete. [chuckle]")
STARTUP_WARMUP = os.getenv("STARTUP_WARMUP", "1") == "1"

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "1000"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.2"))
DEFAULT_NORM_LOUDNESS = os.getenv("DEFAULT_NORM_LOUDNESS", "1") == "1"

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# -----------------------------
# Global state
# -----------------------------

model: Optional[ChatterboxTurboTTS] = None
model_lock = threading.Lock()

# key -> Conditionals object from chatterbox.tts_turbo
conditionals_cache: "OrderedDict[str, Any]" = OrderedDict()
voice_digest_cache: "OrderedDict[str, tuple[tuple[int, int, int, int], str]]" = OrderedDict()


# -----------------------------
# Auth
# -----------------------------

async def require_api_key(request: Request) -> None:
    if not API_KEY:
        return

    auth = request.headers.get("authorization", "")
    x_api_key = request.headers.get("x-api-key", "")

    if auth == f"Bearer {API_KEY}" or x_api_key == API_KEY:
        return

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# -----------------------------
# Utility
# -----------------------------

def configure_torch() -> str:
    global DEVICE

    torch.set_grad_enabled(False)

    if DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            if REQUIRE_CUDA:
                raise RuntimeError("DEVICE=cuda requested, but torch.cuda.is_available() is false.")
            DEVICE = "cpu"
        else:
            # RTX 3060 / 3090 are Ampere. TF32 improves matmul/conv speed with negligible TTS impact.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    return DEVICE


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
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Max is {MAX_TEXT_CHARS} chars. For realtime calls, chunk hard.",
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
    - empty/default -> DEFAULT_VOICE
    - plain filename -> VOICE_DIR / filename  (no path traversal allowed)
    - absolute path -> only when ALLOW_ABSOLUTE_VOICE_PATHS=1
    """
    if not voice or voice == "default":
        return Path(DEFAULT_VOICE).resolve() if DEFAULT_VOICE else None

    raw = voice.strip()

    if os.path.isabs(raw):
        if os.getenv("ALLOW_ABSOLUTE_VOICE_PATHS", "0") != "1":
            raise HTTPException(status_code=400, detail="Absolute voice paths are disabled.")
        p = Path(raw).resolve()
    else:
        # Only plain filenames are accepted; reject anything with a directory separator.
        if "/" in raw or os.sep in raw:
            raise HTTPException(status_code=400, detail="Voice name must be a plain filename, not a path.")
        p = (VOICE_DIR / raw).resolve()
        # Belt-and-suspenders: ensure the resolved path stays inside VOICE_DIR.
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

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_DIR / f"chatterbox_voice_{uuid.uuid4().hex}{suffix}"
    out_path.write_bytes(data)
    return out_path


def cache_key_for_voice(path: Path, norm_loudness: bool) -> str:
    return f"{cached_file_sha256(path)}:norm={int(norm_loudness)}"


def get_or_prepare_conditionals(voice_path: Path, norm_loudness: bool) -> tuple[Any, bool]:
    """
    Critical speed trick.
    ChatterboxTurboTTS.generate(audio_prompt_path=...) calls prepare_conditionals every time.
    We instead prepare once, cache model.conds, then call generate(..., audio_prompt_path=None).

    Must be called under model_lock because prepare_conditionals mutates model.conds.
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


def wav_tensor_to_wav_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    if isinstance(wav, torch.Tensor):
        wav = wav.detach()
        if wav.device.type != "cpu":
            wav = wav.cpu()
        if not torch.is_floating_point(wav):
            wav = wav.float()
    else:
        wav = torch.as_tensor(wav, dtype=torch.float32)

    # Chatterbox returns [1, samples]. SoundFile wants [samples] or [samples, channels].
    if wav.ndim == 2 and wav.shape[0] == 1:
        arr = wav.squeeze(0).numpy()
    elif wav.ndim == 1:
        arr = wav.numpy()
    else:
        arr = wav.squeeze().numpy()

    # Keep TTS output standard and API-friendly.
    arr = np.clip(arr, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def generate_locked(
    *,
    text: str,
    voice_path: Optional[Path],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    norm_loudness: bool,
    seed: int,
) -> tuple[bytes, dict[str, Any]]:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    started = time.perf_counter()
    voice_cache_hit = False

    with model_lock:
        # Seed inside the lock so concurrent requests don't clobber each other's RNG state.
        set_seed(seed)

        if voice_path is not None:
            conds, voice_cache_hit = get_or_prepare_conditionals(voice_path, norm_loudness=norm_loudness)
            model.conds = conds
        elif model.conds is None:
            raise RuntimeError("No default voice conditionals are loaded. Provide a voice file or set DEFAULT_VOICE.")

        with torch.inference_mode():
            # Turbo currently ignores min_p, cfg_weight, and exaggeration. Do not expose fake knobs.
            wav = model.generate(
                text.strip(),
                audio_prompt_path=None,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                norm_loudness=bool(norm_loudness),
            )

        sample_rate = int(model.sr)

        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    audio_seconds = float(wav.shape[-1] / sample_rate)
    wav_bytes = wav_tensor_to_wav_bytes(wav, sample_rate)

    meta = {
        "sample_rate": sample_rate,
        "wall_seconds": round(elapsed, 4),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(elapsed / audio_seconds, 4) if audio_seconds > 0 else None,
        "x_realtime": round(audio_seconds / elapsed, 4) if elapsed > 0 else None,
        "device": DEVICE,
        "voice_cached": voice_cache_hit,
        "cache_entries": len(conditionals_cache),
    }

    return wav_bytes, meta


def response_with_audio(wav_bytes: bytes, meta: dict[str, Any]) -> Response:
    headers: dict[str, str] = {
        "Content-Disposition": 'attachment; filename="speech.wav"',
        "X-Sample-Rate": str(meta["sample_rate"]),
        "X-Wall-Seconds": str(meta["wall_seconds"]),
        "X-Audio-Seconds": str(meta["audio_seconds"]),
        "X-Voice-Cached": str(meta["voice_cached"]).lower(),
        "X-Voice-Cache-Entries": str(meta["cache_entries"]),
    }
    # Only include performance-ratio headers when they were successfully computed.
    if meta.get("rtf") is not None:
        headers["X-RTF"] = str(meta["rtf"])
    if meta.get("x_realtime") is not None:
        headers["X-Speed-X-Realtime"] = str(meta["x_realtime"])
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


# -----------------------------
# Lifespan
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Warn loudly if the API key is still the placeholder value.
    _key = os.getenv("API_KEY", "")
    if not _key or _key == "change-this-key":
        logger.warning(
            "API_KEY is unset or still the placeholder 'change-this-key'. "
            "The server is accessible without authentication — set a strong API_KEY before exposing this service."
        )

    final_device = configure_torch()
    logger.info("Loading Chatterbox Turbo on %s...", final_device)

    model = ChatterboxTurboTTS.from_pretrained(device=final_device)

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    if DEFAULT_VOICE:
        default_path = Path(DEFAULT_VOICE).resolve()
        if not default_path.exists():
            raise RuntimeError(f"DEFAULT_VOICE does not exist: {default_path}")

        logger.info("Preparing default voice conditionals: %s", default_path)
        with model_lock:
            default_conds, _ = get_or_prepare_conditionals(default_path, norm_loudness=DEFAULT_NORM_LOUDNESS)
            model.conds = default_conds

    if STARTUP_WARMUP:
        logger.info("Running warmup...")
        try:
            # If DEFAULT_VOICE exists, this uses cached conditionals. If not, built-in conds may exist.
            generate_locked(
                text=WARMUP_TEXT,
                voice_path=Path(DEFAULT_VOICE).resolve() if DEFAULT_VOICE else None,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                norm_loudness=DEFAULT_NORM_LOUDNESS,
                seed=0,
            )
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning("Warmup skipped/failed: %s: %s", type(e).__name__, e)

    yield

    logger.info("Releasing Chatterbox model.")
    model = None
    conditionals_cache.clear()
    voice_digest_cache.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Chatterbox Turbo FastAPI",
    version="2.0.0",
    lifespan=lifespan,
)


# -----------------------------
# API models
# -----------------------------

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field("default", description="'default' or a filename inside VOICE_DIR")
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    norm_loudness: bool = DEFAULT_NORM_LOUDNESS
    seed: int = 0
    response_format: str = Field("wav", description="wav or json_base64")


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
) -> tuple[bytes, dict[str, Any]]:
    return await anyio.to_thread.run_sync(
        lambda: generate_locked(
            text=text,
            voice_path=voice_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
        ),
        abandon_on_cancel=False,
    )


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    gpu = None

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        gpu = {
            "name": torch.cuda.get_device_name(idx),
            "index": idx,
            "total_vram_gb": round(props.total_memory / 1024**3, 2),
            "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
        }

    return {
        "ok": model is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "sample_rate": int(model.sr) if model else None,
        "default_voice": DEFAULT_VOICE or None,
        "voice_dir": str(VOICE_DIR),
        "max_text_chars": MAX_TEXT_CHARS,
        "voice_cache_entries": len(conditionals_cache),
        "voice_cache_size": VOICE_CACHE_SIZE,
    }


@app.get("/voices", dependencies=[Depends(require_api_key)])
def voices() -> dict[str, list[str]]:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "voices": sorted(
            p.name for p in VOICE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_AUDIO_SUFFIXES
        )
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
):
    validate_generation_args(
        text=text,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    uploaded_path: Optional[Path] = None

    try:
        if voice is not None:
            uploaded_path = await save_upload(voice)
            voice_path = uploaded_path
        else:
            voice_path = normalize_voice_path("default")

        wav_bytes, meta = await run_generation(
            text=text,
            voice_path=voice_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
        )

        return response_with_audio(wav_bytes, meta)

    except HTTPException:
        raise
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
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

    if req.response_format not in {"wav", "json_base64"}:
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'json_base64'.")

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
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("TTS generation failed (OpenAI speech endpoint)")
        raise HTTPException(status_code=500, detail="TTS failed. Check server logs.")


@app.post("/warmup", dependencies=[Depends(require_api_key)])
async def warmup():
    try:
        voice_path = normalize_voice_path("default")
        wav_bytes, meta = await run_generation(
            text=WARMUP_TEXT,
            voice_path=voice_path,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            norm_loudness=DEFAULT_NORM_LOUDNESS,
            seed=0,
        )
        return {"ok": True, "metadata": meta, "bytes": len(wav_bytes)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Warmup failed")
        raise HTTPException(status_code=500, detail="Warmup failed. Check server logs.")
