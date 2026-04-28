#!/usr/bin/env bash
set -euo pipefail

# Change to the directory that contains this script so the server can always
# find relative paths (voices/, server.py, etc.) regardless of the caller's CWD.
cd "$(dirname "$0")"

export ALLOW_NO_AUTH="${ALLOW_NO_AUTH:-0}"

if [[ "$ALLOW_NO_AUTH" != "1" && -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY is required." >&2
  echo "  Set a strong key:  API_KEY=\"\$(openssl rand -hex 32)\" ./run_3090_or_3060.sh" >&2
  echo "  Local dev only:    ALLOW_NO_AUTH=1 ./run_3090_or_3060.sh" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export VOICE_DIR="${VOICE_DIR:-$PWD/voices}"
export DEFAULT_VOICE="${DEFAULT_VOICE:-$VOICE_DIR/default.wav}"
export API_KEY="${API_KEY:-}"
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-700}"
export VOICE_CACHE_SIZE="${VOICE_CACHE_SIZE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$VOICE_DIR"

exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8011}" --workers 1
