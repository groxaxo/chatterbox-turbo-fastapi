#!/usr/bin/env bash
set -euo pipefail

API="${API:-http://127.0.0.1:7766}"
RUN_SPANISH_TESTS="${RUN_SPANISH_TESTS:-1}"

echo "==> /healthz (no auth)"
curl -sf "$API/healthz" | jq .

echo "==> /status"
curl -sf "$API/status" | jq .

echo "==> /v1/models"
curl -sf "$API/v1/models" | jq .

echo "==> /v1/audio/models"
curl -sf "$API/v1/audio/models" | jq .

echo "==> /v1/audio/voices"
curl -sf "$API/v1/audio/voices" | jq .

echo "==> /v1/profiles"
curl -sf "$API/v1/profiles" | jq .

echo "==> English /v1/audio/speech (default MP3 / Open WebUI path)"
curl -sf -X POST "$API/v1/audio/speech" \
  -H "content-type: application/json" \
  -d '{
    "input": "Hi there, this is the English Chatterbox Turbo voice.",
    "voice": "alloy",
    "model": "tts-1",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2
  }' \
  --output speech-en.mp3 \
  -D - | grep -Ei "x-chatterbox-profile|x-chatterbox-language|x-voice-cached|x-rtf|x-wall|x-output-format"
file speech-en.mp3

if [[ "$RUN_SPANISH_TESTS" == "1" ]]; then
  echo "==> Spanish Argentina via voice=lucia-ar"
  curl -sf -X POST "$API/v1/audio/speech" \
    -H "content-type: application/json" \
    -d '{
      "input": "¡Hola! Soy Lucía, y esta es una prueba de español argentino.",
      "voice": "lucia-ar",
      "model": "tts-1",
      "response_format": "wav",
      "temperature": 0.75,
      "top_p": 0.95,
      "top_k": 1000,
      "repetition_penalty": 1.2
    }' \
    --output speech-es-ar.wav \
    -D - | grep -Ei "x-chatterbox-profile|x-chatterbox-language|x-voice-cached|x-rtf|x-wall|x-output-format"
  file speech-es-ar.wav

  echo "==> Spanish LATAM via profile-specific endpoint"
  curl -sf -X POST "$API/v1/audio/speech/lucia-latam" \
    -H "content-type: application/json" \
    -d '{
      "input": "Hola, esta es una prueba de la voz latinoamericana equilibrada.",
      "voice": "alloy",
      "model": "tts-1",
      "response_format": "wav",
      "temperature": 0.75,
      "top_p": 0.95,
      "top_k": 1000,
      "repetition_penalty": 1.2
    }' \
    --output speech-es-latam.wav \
    -D - | grep -Ei "x-chatterbox-profile|x-chatterbox-language|x-voice-cached|x-rtf|x-wall|x-output-format"
  file speech-es-latam.wav
fi

echo "==> Path traversal must return 400"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/v1/audio/speech" \
  -H "content-type: application/json" \
  -d '{"input":"test","voice":"../secret.wav"}')
[[ "$STATUS" == "400" ]] && echo "PASS (got 400)" || { echo "FAIL (got $STATUS)"; exit 1; }
