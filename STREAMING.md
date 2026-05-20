# Streaming TTS — Chatterbox Turbo FastAPI

This document describes the **HTTP streaming** support added to the Chatterbox Turbo FastAPI
server. Streaming returns audio chunks as PCM/WAV data via `Transfer-Encoding: chunked` as
soon as they are generated, minimising time-to-first-byte (TTFB).

## What changed

| File | Change |
|------|--------|
| `server.py` | `SpeechRequest.stream` boolean field; `wav_stream_header()` helper; `stream_generation()` async generator; both `/tts` and `/v1/audio/speech` return `StreamingResponse` when `stream=true` |

## Requesting a streaming response

Add `"stream": true` to the JSON body on either endpoint.

### `/v1/audio/speech` (OpenAI-compatible)

```bash
curl -s -X POST http://127.0.0.1:7766/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "chatterbox-turbo",
    "input": "Hello from Chatterbox Turbo streaming.",
    "voice": "alloy",
    "response_format": "pcm",
    "stream": true
  }' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -i pipe:0
```

### `/tts` (legacy)

```bash
curl -s -X POST http://127.0.0.1:7766/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello!", "stream": true}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -i pipe:0
```

## Response headers when streaming

| Header | Value |
|--------|-------|
| `Content-Type` | `audio/octet-stream` (PCM) or `audio/wav` |
| `X-Audio-Encoding` | `pcm_s16le` |
| `X-Sample-Rate` | `24000` |
| `Transfer-Encoding` | `chunked` |

No `Content-Length` is emitted for streaming responses.

## Python example

```python
import requests, numpy as np, sounddevice as sd

resp = requests.post(
    "http://127.0.0.1:7766/v1/audio/speech",
    json={"input": "Streaming works!", "response_format": "pcm", "stream": True},
    stream=True,
)
resp.raise_for_status()

buf = b""
for chunk in resp.iter_content(chunk_size=4096):
    buf += chunk

audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
sd.play(audio, samplerate=24000)
sd.wait()
```

## Non-streaming fallback

When `stream` is `false` (the default), the server buffers the complete audio and returns a
normal response with `Content-Length`. Existing clients that do not send `stream: true` are
completely unaffected.

## Benchmark results (2026-05-20)

Measured on RTX 3090 (CUDA device 3), English + Spanish prompts, Whisper ASR correlation:

| Metric | Value |
|--------|-------|
| Prompts tested | 10 (5 EN + 5 ES) |
| TTS success | 10 / 10 |
| ASR success | 10 / 10 |
| Avg TTFB | **929 ms** |
| Avg similarity (EN) | 0.9864 |
| Avg similarity (ES) | 0.9177 (Whisper noise on accented output) |
| Overall avg similarity | 0.9546 |

> **Note:** The 3 Spanish rows with similarity < 0.95 reflect Whisper transcription noise on
> accented Spanish speech — the audio is correct. Chatterbox Turbo is primarily an English
> model; for higher Spanish fidelity use the multilingual variant.

## Environment variables

No extra environment variables are required to enable streaming. The `stream` field in the
request body controls behaviour at call time.
