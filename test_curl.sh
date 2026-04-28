#!/usr/bin/env bash
set -euo pipefail

API="${API:-http://127.0.0.1:8011}"
KEY="${API_KEY:?API_KEY must be set}"
VOICE="${VOICE:-voices/default.wav}"

echo "==> /healthz (no auth)"
curl -sf "$API/healthz" | jq .

echo "==> /status (authenticated)"
curl -sf "$API/status" -H "authorization: Bearer $KEY" | jq .

echo "==> /v1/audio/speech (first request)"
curl -sf -X POST "$API/v1/audio/speech" \
  -H "content-type: application/json" \
  -H "authorization: Bearer $KEY" \
  -d '{
    "input": "Hi there, this is Chatterbox Turbo running behind the production-hardened FastAPI server.",
    "voice": "default",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
    "response_format": "wav"
  }' \
  --output speech.wav \
  -D - | grep -E "x-voice-cached|x-rtf|x-wall"

file speech.wav

echo "==> Path traversal must return 400"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/v1/audio/speech" \
  -H "authorization: Bearer $KEY" \
  -H "content-type: application/json" \
  -d '{"input":"test","voice":"../secret.wav"}')
[[ "$STATUS" == "400" ]] && echo "PASS (got 400)" || { echo "FAIL (got $STATUS)"; exit 1; }
