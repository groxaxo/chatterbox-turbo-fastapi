# Chatterbox Turbo English + Spanish FastAPI

A production-oriented FastAPI and Celery wrapper for **official English Chatterbox Turbo** plus the **Lucía Latin-American Spanish fine-tunes** from `groxaxo/chaturbo-espanol`.

The same API process and worker pool can synthesize both languages without restarting:

- English uses the untouched `ResembleAI/chatterbox-turbo` T3-Turbo model and normal reference-voice cloning.
- Spanish uses the Lucía merged checkpoint or LoRA adapters from `groxaxo/chaturbo-espanol`.
- The official base model is **not** included in the Spanish repository and is always loaded separately.
- English and Spanish share the decoder, voice encoder, tokenizer, chunking, streaming, response encoding, and Celery infrastructure.

## Supported voices and profiles

| OpenAI `voice` value | Language | Spanish artifact | Conditioning |
|---|---|---|---|
| `alloy` / `default` | English | Official base model | `DEFAULT_VOICE` or `voices/default.wav` |
| Any local filename in `VOICE_DIR` | English | Official base model | Uploaded/local reference voice |
| `lucia-ar` | es-AR | Merged T3 checkpoint | Bundled Lucía AR persona |
| `lucia-latam` | es-419 | LoRA adapter on the official base | Bundled Lucía LATAM persona |
| `lucia-cl-pilot` | es-CL | LoRA adapter on the official base | Lucía LATAM persona fallback |
| `lucia-co-pilot` | es-CO | LoRA adapter on the official base | Lucía LATAM persona fallback |

Convenience aliases include `lucia`, `spanish`, `es`, `es-AR`, `es-419`, `es-CL`, and `es-CO`.

The Chilean and Colombian pilot artifact directories do not include their own `conditioning.pt` and `reference.wav`, so the runtime deliberately reuses the Lucía LATAM persona conditioning while applying the regional pilot adapter.

## Architecture

`server.py` remains the stable synthesis, chunking, streaming, voice-cache, and Celery implementation. The multilingual entrypoints install a profile-aware runtime layer:

- `multilingual_server.py` is the ASGI entrypoint.
- `multilingual_celery_worker.py` is the Celery entrypoint.
- `multilingual_runtime.py` manages model downloads, profile selection, strict checkpoint/adapter loading, persona conditioning, LRU eviction, and status reporting.

The English engine stays as the shared backbone. Spanish engines retain their own T3 model while sharing the English engine's S3Gen decoder, voice encoder, and tokenizer. This avoids loading a complete second Chatterbox stack for every Spanish profile.

A worker can keep English and Spanish available together. `SPANISH_PROFILE_CACHE_SIZE` controls how many Spanish T3 variants remain resident in addition to English.

## Install

Requirements:

- Python 3.11 recommended
- FFmpeg
- Redis when Celery mode is enabled
- CUDA for normal production use; CPU also works but is slower
- MPS is not recommended for this model because attention can exhaust unified memory

```bash
sudo apt update
sudo apt install -y ffmpeg redis-server

conda activate base
cd chatterbox-turbo-fastapi
./install_cuda124.sh
```

The installer adds the runtime dependencies required for merged checkpoints and adapters: `huggingface-hub`, `safetensors`, and `peft`.

## Model download

### Automatic Hugging Face cache

By default, the first worker request downloads:

- `ResembleAI/chatterbox-turbo`
- `groxaxo/chaturbo-espanol`

Set `HF_TOKEN` when the repository or deployment requires authentication.

### Explicit offline/local download

```bash
conda activate chatterbox-turbo-api
python download_models.py \
  --output-dir ./models \
  --profiles lucia-ar,lucia-latam
```

For every profile:

```bash
python download_models.py --output-dir ./models --profiles all
```

Then configure:

```bash
export BASE_MODEL_DIR="$PWD/models/chatterbox-turbo"
export SPANISH_MODEL_DIR="$PWD/models/chaturbo-espanol"
```

The runtime does not trust or require the absolute paths stored in the training host's `training_config.json`. It resolves the official base directory and each profile artifact directly from these environment variables or the Hugging Face cache.

## Run

Prepare an English reference voice:

```bash
mkdir -p voices
cp /path/to/english-reference.wav voices/default.wav
```

Start the production API and configured Celery workers:

```bash
./run_api_service.sh
```

Manual worker launch on a physical GPU:

```bash
./run_celery_worker.sh 2
./run_celery_worker.sh 3
```

The API remains on port `7766` by default.

## OpenAI-compatible API

### English

```bash
curl -X POST http://127.0.0.1:7766/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "alloy",
    "input": "Hello. This request uses the official English Turbo model.",
    "response_format": "mp3"
  }' \
  --output english.mp3
```

A local English reference file can be selected by filename:

```json
{
  "model": "tts-1",
  "voice": "my-speaker.wav",
  "input": "This uses a file from VOICE_DIR."
}
```

### Spanish Argentina — merged checkpoint

```bash
curl -X POST http://127.0.0.1:7766/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "lucia-ar",
    "input": "¡Hola! Soy Lucía, y esta es una prueba de español argentino.",
    "response_format": "wav",
    "temperature": 0.75,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2
  }' \
  --output lucia-ar.wav
```

### Balanced LATAM — LoRA adapter

```bash
curl -X POST http://127.0.0.1:7766/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "lucia-latam",
    "input": "Hola, esta es la voz latinoamericana equilibrada.",
    "response_format": "mp3"
  }' \
  --output lucia-latam.mp3
```

### Profile-specific endpoint

The profile may also be placed in the URL. The request's `voice` value is overridden by the path profile:

```bash
curl -X POST http://127.0.0.1:7766/v1/audio/speech/lucia-latam \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "alloy",
    "input": "Este endpoint selecciona el perfil de forma explícita.",
    "response_format": "wav"
  }' \
  --output explicit-profile.wav
```

Responses include:

- `X-Chatterbox-Profile: english`, `lucia-ar`, `lucia-latam`, etc.
- `X-Chatterbox-Language: en`, `es-AR`, `es-419`, etc.
- Existing timing, chunking, cache, and output-format headers.

## Discovery and status

| Endpoint | Description |
|---|---|
| `GET /healthz` | Lightweight liveness probe |
| `GET /status` | Runtime, GPU, Celery, and multilingual cache status |
| `GET /profiles` | Spanish profile registry |
| `GET /v1/profiles` | Versioned alias of `/profiles` |
| `GET /v1/audio/voices` | English voices plus Lucía profiles |
| `GET /v1/models` | OpenAI-compatible TTS model aliases |
| `POST /v1/audio/speech` | OpenAI-compatible English or Spanish synthesis |
| `POST /v1/audio/speech/{profile}` | Explicit Spanish profile synthesis |
| `POST /tts` | Existing multipart English voice-upload endpoint |
| `POST /warmup` | Existing English warmup endpoint |

```bash
curl -s http://127.0.0.1:7766/v1/profiles | jq .
curl -s http://127.0.0.1:7766/status | jq .multilingual
```

## Open WebUI

Use:

- TTS engine: `OpenAI`
- Base URL: `http://127.0.0.1:7766` or `http://127.0.0.1:7766/v1`
- API key: any value; authentication is currently disabled by the launcher
- Model: `tts-1`
- Voice: `alloy`, `lucia-ar`, or `lucia-latam`

The language is selected by the voice, so one Open WebUI endpoint can serve both English and Spanish.

## Streaming

The existing `stream: true` behavior remains available for English and Spanish:

```bash
curl -N -X POST http://127.0.0.1:7766/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "voice": "lucia-ar",
    "input": "Esta respuesta se genera por fragmentos.",
    "response_format": "pcm",
    "stream": true
  }' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -i pipe:0
```

Celery still synthesizes chunks in parallel and returns them in sentence order. Direct mode emits chunks as each one completes.

## Configuration

Copy `multilingual.env.example` into your deployment environment or add the settings to `/etc/chatterbox-turbo-fastapi.env`.

Important variables:

| Variable | Default | Meaning |
|---|---|---|
| `BASE_MODEL_REPO` | `ResembleAI/chatterbox-turbo` | Official English/base checkpoint |
| `BASE_MODEL_REVISION` | `main` | Base revision |
| `BASE_MODEL_DIR` | empty | Offline/local base model directory |
| `SPANISH_ENABLED` | `1` | Enable Lucía profile routing |
| `SPANISH_MODEL_REPO` | `groxaxo/chaturbo-espanol` | Spanish artifact repository |
| `SPANISH_MODEL_REVISION` | `main` | Spanish revision |
| `SPANISH_MODEL_DIR` | empty | Offline/local Spanish artifact directory |
| `DEFAULT_SPANISH_PROFILE` | `lucia-ar` | Target for aliases such as `lucia` and `es` |
| `SPANISH_PROFILE_CACHE_SIZE` | `1` | Spanish T3 variants retained per worker |
| `PRELOAD_PROFILES` | empty | Comma-separated Spanish profiles to load at worker startup |
| `STRICT_SPANISH_TAGS` | `1` | Reject unknown bracketed Turbo tags |
| `HF_TOKEN` | empty | Hugging Face token when needed |

Optional persona overrides:

- `LUCIA_REFERENCE_WAV`
- `LUCIA_CONDITIONING_PT`
- Per-profile forms such as `LUCIA_AR_REFERENCE_WAV` and `LUCIA_AR_CONDITIONING_PT`

Normally the bundled `reference.wav` and `conditioning.pt` should be used unchanged.

## VRAM and cache policy

English always supplies the shared backbone. Spanish profiles add only a separate T3/LoRA model and their conditionals.

For 12 GB workers, start with:

```bash
SPANISH_PROFILE_CACHE_SIZE=1
PRELOAD_PROFILES=
```

This keeps English plus the most recently used Spanish profile resident. Switching from `lucia-ar` to `lucia-latam` evicts the older Spanish T3 but does not unload English or the shared decoder.

On larger GPUs, keep two Spanish profiles hot:

```bash
SPANISH_PROFILE_CACHE_SIZE=2
PRELOAD_PROFILES=lucia-ar,lucia-latam
```

`PRELOAD_PROFILES` must not exceed the cache size unless intentional LRU eviction is acceptable.

## Systemd

```bash
sudo ./install_systemd_services.sh
```

This installs:

- `chatterbox-turbo-fastapi.service`
- `chatterbox-turbo-celery@.service`
- `chatterbox-turbo.target`

Configuration lives at:

```text
/etc/chatterbox-turbo-fastapi.env
```

Existing environment files are preserved. Add the multilingual variables manually when upgrading an already installed service.

## Test

```bash
./test_curl.sh
```

This checks discovery, English synthesis, Lucía AR synthesis, Lucía LATAM synthesis, response headers, and the existing path-traversal guard.

Skip the Spanish generation calls while testing only API wiring:

```bash
RUN_SPANISH_TESTS=0 ./test_curl.sh
```

Unit tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Important model notes

- The Spanish Hugging Face repository contains only fine-tuned artifacts. It does not contain the official base model.
- `lucia-ar/t3_turbo_finetuned_merged.safetensors` is loaded with `strict=True` after constructing the native Turbo T3.
- The merged checkpoint legitimately has 298 tensors because the live Turbo model removes `tfmr.wte.weight` after loading the native tokenizer-compatible embedding structure.
- LoRA profiles use `PeftModel.from_pretrained(..., is_trainable=False)` on a clean official Turbo T3.
- Spanish text is NFC-normalized while preserving `ñ`, accents, `ü`, inverted punctuation, and recognized Turbo tags.
- The API does not rewrite numbers, dates, currencies, abbreviations, or phone numbers because automatic expansion can change what should be spoken.
