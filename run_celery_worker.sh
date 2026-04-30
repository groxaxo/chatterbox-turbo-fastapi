#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_PHYSICAL_INDEX="${1:-${GPU_PHYSICAL_INDEX:-2}}"
PY_ENV_PREFIX="${PY_ENV_PREFIX:-/home/op/miniconda3/envs/chatterbox-turbo-api/bin}"
CELERY_BIN="${CELERY_BIN:-$PY_ENV_PREFIX/celery}"

BRITISH_WOMAN_REF="/home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav"
DEFAULT_FALLBACK="$PWD/voices/default.wav"

export VOICE_DIR="${VOICE_DIR:-$PWD/voices}"
export DEFAULT_VOICE="${DEFAULT_VOICE:-}"
if [[ -z "$DEFAULT_VOICE" ]]; then
  if [[ -f "$BRITISH_WOMAN_REF" ]]; then
    export DEFAULT_VOICE="$BRITISH_WOMAN_REF"
  else
    export DEFAULT_VOICE="$DEFAULT_FALLBACK"
  fi
fi

export ENABLE_CELERY=0
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-RTX 3060}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/14}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$CELERY_BROKER_URL}"
export CELERY_QUEUE="${CELERY_QUEUE:-chatterbox_tts}"
export WORKER_LAZY_LOAD_MODEL="${WORKER_LAZY_LOAD_MODEL:-0}"
export WORKER_MODEL_IDLE_UNLOAD_SECONDS="${WORKER_MODEL_IDLE_UNLOAD_SECONDS:-0}"
export WORKER_STARTUP_WARMUP="${WORKER_STARTUP_WARMUP:-1}"
export LAZY_LOAD_MODEL="$WORKER_LAZY_LOAD_MODEL"
export MODEL_IDLE_UNLOAD_SECONDS="$WORKER_MODEL_IDLE_UNLOAD_SECONDS"
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
export STARTUP_WARMUP="$WORKER_STARTUP_WARMUP"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$VOICE_DIR"

GPU_INFO="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | awk -F', ' -v idx="$GPU_PHYSICAL_INDEX" '$1 == idx {print $2 "|" $3; exit}')"
if [[ -z "$GPU_INFO" ]]; then
  echo "ERROR: Could not resolve physical GPU index $GPU_PHYSICAL_INDEX via nvidia-smi." >&2
  exit 1
fi

GPU_UUID="${GPU_INFO%%|*}"
GPU_NAME="${GPU_INFO#*|}"
if [[ -n "$EXPECTED_GPU_NAME" && "$GPU_NAME" != *"$EXPECTED_GPU_NAME"* ]]; then
  echo "ERROR: Physical GPU index $GPU_PHYSICAL_INDEX resolved to '$GPU_NAME', expected '$EXPECTED_GPU_NAME'." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID"

exec "$CELERY_BIN" -A celery_worker.celery_app worker \
  --loglevel="${CELERY_LOGLEVEL:-INFO}" \
  --pool=solo \
  --concurrency=1 \
  --queues="$CELERY_QUEUE" \
  --hostname="chatterbox-gpu${GPU_PHYSICAL_INDEX}@%h"
