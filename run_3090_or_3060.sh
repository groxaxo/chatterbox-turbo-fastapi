#!/usr/bin/env bash
set -euo pipefail

# Change to the directory that contains this script so the server can always
# find relative paths (voices/, server.py, etc.) regardless of the caller's CWD.
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VOICE_DIR="${VOICE_DIR:-$PWD/voices}"
export DEFAULT_VOICE="${DEFAULT_VOICE:-$VOICE_DIR/default.wav}"
export API_KEY="${API_KEY:-change-this-key}"
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-700}"
export VOICE_CACHE_SIZE="${VOICE_CACHE_SIZE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$VOICE_DIR"

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8011}" --workers 1
