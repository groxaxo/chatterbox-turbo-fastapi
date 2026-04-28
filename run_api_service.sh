#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY_ENV_PREFIX="${PY_ENV_PREFIX:-/home/op/miniconda3/envs/chatterbox-turbo-api/bin}"
UVICORN_BIN="${UVICORN_BIN:-$PY_ENV_PREFIX/uvicorn}"

export ALLOW_NO_AUTH="${ALLOW_NO_AUTH:-0}"
if [[ "$ALLOW_NO_AUTH" != "1" && -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is required." >&2
  echo "  Set a strong key:  API_KEY=\"\$(openssl rand -hex 32)\" ./run_api_service.sh" >&2
  echo "  Local dev only:    ALLOW_NO_AUTH=1 ./run_api_service.sh" >&2
  exit 1
fi

BRITISH_WOMAN_REF="/home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav"
DEFAULT_FALLBACK="$PWD/voices/default.wav"

export PORT="${PORT:-7766}"
export VOICE_DIR="${VOICE_DIR:-$PWD/voices}"
export DEFAULT_VOICE="${DEFAULT_VOICE:-}"
if [[ -z "$DEFAULT_VOICE" ]]; then
  if [[ -f "$BRITISH_WOMAN_REF" ]]; then
    export DEFAULT_VOICE="$BRITISH_WOMAN_REF"
  else
    export DEFAULT_VOICE="$DEFAULT_FALLBACK"
  fi
fi

export ENABLE_CELERY="${ENABLE_CELERY:-1}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/14}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$CELERY_BROKER_URL}"
export CELERY_QUEUE="${CELERY_QUEUE:-chatterbox_tts}"
export WORKER_GPU_INDICES="${WORKER_GPU_INDICES:-2,3}"
export LAZY_LOAD_MODEL="${LAZY_LOAD_MODEL:-1}"
export MODEL_IDLE_UNLOAD_SECONDS="${MODEL_IDLE_UNLOAD_SECONDS:-900}"
export MODEL_IDLE_CHECK_INTERVAL_SECONDS="${MODEL_IDLE_CHECK_INTERVAL_SECONDS:-30}"
export AUTO_CHUNK_ENABLED="${AUTO_CHUNK_ENABLED:-1}"
export AUTO_CHUNK_TARGET_CHARS="${AUTO_CHUNK_TARGET_CHARS:-520}"
export AUTO_CHUNK_HARD_LIMIT="${AUTO_CHUNK_HARD_LIMIT:-580}"
export CHUNK_PAUSE_MS="${CHUNK_PAUSE_MS:-140}"
export DEFAULT_RESPONSE_FORMAT="${DEFAULT_RESPONSE_FORMAT:-mp3}"
export MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-12000}"
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-520}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-25}"
export VOICE_CACHE_SIZE="${VOICE_CACHE_SIZE:-8}"
export STARTUP_WARMUP="${STARTUP_WARMUP:-0}"
export DEVICE="${DEVICE:-cpu}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$VOICE_DIR"

exec "$UVICORN_BIN" server:app --host 0.0.0.0 --port "$PORT" --workers 1
