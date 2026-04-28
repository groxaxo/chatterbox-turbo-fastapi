#!/usr/bin/env bash
set -euo pipefail

API="${API:-http://127.0.0.1:8011}"
KEY="${API_KEY:-change-this-key}"
VOICE="${VOICE:-voices/default.wav}"

curl "$API/health" | jq .

curl -X POST "$API/v1/audio/speech" \
  -H "content-type: application/json" \
  -H "authorization: Bearer $KEY" \
  -d '{
    "input": "Hi there, this is Chatterbox Turbo running behind the improved FastAPI server. [chuckle]",
    "voice": "default",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
    "response_format": "wav"
  }' \
  --output speech.wav

file speech.wav
