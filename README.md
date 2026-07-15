# Chatterbox Turbo English + Spanish FastAPI

Production-oriented FastAPI and Celery serving for:

- the official English `ResembleAI/chatterbox-turbo` model;
- the Lucía Latin-American Spanish artifacts from `groxaxo/chaturbo-espanol`;
- one OpenAI-compatible endpoint that can switch languages per request without restarting.

The Spanish repository does **not** include the official base model. This service downloads or loads both repositories and reconstructs each profile with the correct training ancestry.

## Profiles

Select the language/profile through the OpenAI `voice` field.

| `voice` | Language | Exact T3 chain | Persona conditioning |
|---|---|---|---|
| `alloy` / `default` | English | Official Turbo | `DEFAULT_VOICE` or `voices/default.wav` |
| filename in `VOICE_DIR` | English | Official Turbo | Selected reference file |
| `lucia-ar` | es-AR | Official Turbo architecture → Lucía AR merged T3 | `lucia-ar/conditioning.pt` + `reference.wav` |
| `lucia-latam` | es-419 | Official Turbo architecture → Lucía AR merged T3 → LATAM LoRA | `lucia-latam/conditioning.pt` + `reference.wav` |
| `lucia-cl-pilot` | es-CL | Official Turbo architecture → Lucía AR merged T3 → Chile pilot LoRA | Lucía LATAM persona fallback |
| `lucia-co-pilot` | es-CO | Official Turbo architecture → Lucía AR merged T3 → Colombia pilot LoRA | Lucía LATAM persona fallback |

Aliases include `lucia`, `spanish`, `es`, `es-AR`, `es-419`, `es-CL`, `es-CO`, `argentina`, `chile`, and `colombia`.

The LATAM and pilot adapters are **continual LoRAs**. Applying them directly to the untouched official T3 is incorrect. The runtime first strict-loads `lucia-ar/t3_turbo_finetuned_merged.safetensors`, then applies the requested PEFT adapter.

## Architecture

The existing `server.py` remains responsible for validation, sentence chunking, HTTP streaming, MP3/WAV/PCM encoding, voice caching, and Celery dispatch.

The multilingual layer consists of:

- `multilingual_runtime.py` — shared routing, model/cache lifecycle, Spanish normalization, persona conditioning, discovery, and response metadata;
- `chaturbo_espanol_runtime.py` — exact artifact-chain reconstruction and provenance verification;
- `multilingual_server.py` — ASGI entrypoint;
- `multilingual_celery_worker.py` — Celery entrypoint.

The English engine remains the resident backbone. Spanish profiles keep their own T3 or PEFT-wrapped T3 while sharing the official engine's:

- S3Gen decoder;
- voice encoder;
- tokenizer.

This lets a worker serve English and Spanish simultaneously without loading a complete duplicate Chatterbox stack for every profile.

All generation on one worker remains protected by the existing model lock because the decoder and voice encoder are shared. Parallelism comes from separate Celery workers/GPU processes.

## Install

Requirements:

- Python 3.11 recommended;
- FFmpeg;
- Redis when Celery mode is enabled;
- CUDA for production performance; CPU is supported but slower;
- MPS is not recommended because SDPA attention can exhaust unified memory.

```bash
sudo apt update
sudo apt install -y ffmpeg redis-server

conda activate base
cd chatterbox-turbo-fastapi
./install_cuda124.sh
```

The installer pins the Spanish runtime stack used by the fine-tuning repository, including `transformers==5.2.0`, PEFT, safetensors, and `setuptools<81`.

## Download models

### Automatic Hugging Face cache

With `BASE_MODEL_DIR` and `SPANISH_MODEL_DIR` empty, the worker resolves:

- `ResembleAI/chatterbox-turbo`;
- `groxaxo/chaturbo-espanol`.

Set `HF_TOKEN` when authentication is required.

### Explicit/offline download

```bash
conda activate chatterbox-turbo-api
python download_models.py \
  --output-dir ./models \
  --profiles lucia-ar,lucia-latam
```

Or download every profile:

```bash
python download_models.py --output-dir ./models --profiles all
```

The downloader expands dependencies automatically:

- requesting `lucia-latam` also downloads `lucia-ar`, because the AR merged checkpoint is its frozen base;
- requesting a pilot also downloads `lucia-ar` and `lucia-latam`, because pilots require the AR warm base and LATAM persona files.

Configure local paths:

```bash
export BASE_MODEL_DIR="$PWD/models/chatterbox-turbo"
export SPANISH_MODEL_DIR="$PWD/models/chaturbo-espanol"
```

Absolute paths embedded in training-host JSON are not trusted. Runtime paths are resolved from these environment variables or the Hugging Face cache.

## Run

Prepare an English reference voice:

```bash
mkdir -p voices
cp /path/to/english-reference.wav voices/default.wav
```

Production launcher:

```bash
./run_api_service.sh
```

Manual workers on physical GPUs:

```bash
./run_celery_worker.sh 2
./run_celery_worker.sh 3
```

Default API address: `http://127.0.0.1:7766`.

## API examples

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

### Lucía Argentina

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

### Lucía balanced LATAM

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

### Explicit profile route

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

The path profile overrides the body `voice` value.

Buffered audio responses include:

- `X-Chatterbox-Profile`;
- `X-Chatterbox-Language`;
- the existing timing, cache, chunk, and output-format headers.

For `json_base64`, profile and language are included inside the returned metadata. The explicit profile route also adds profile headers to streaming responses.

## Streaming

English and Spanish both support `stream: true`:

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

Celery preserves sentence order after parallel chunk generation. Direct mode yields chunks sequentially as they complete.

## Discovery

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Lightweight liveness |
| `GET /health` | Liveness alias |
| `GET /status` | Device, cache, Celery, and artifact-chain status |
| `GET /profiles` | Spanish profile registry and base chain |
| `GET /v1/profiles` | Versioned profile registry |
| `GET /v1/audio/voices` | English voices plus Spanish profiles |
| `GET /v1/models` | OpenAI-style model aliases |
| `POST /v1/audio/speech` | English or Spanish synthesis |
| `POST /v1/audio/speech/{profile}` | Explicit Spanish profile synthesis |
| `POST /tts` | Multipart English reference upload |
| `POST /warmup` | English warmup |

```bash
curl -s http://127.0.0.1:7766/v1/profiles | jq .
curl -s http://127.0.0.1:7766/status | jq .multilingual
```

## Open WebUI

Use:

- TTS engine: `OpenAI`;
- base URL: `http://127.0.0.1:7766` or `http://127.0.0.1:7766/v1`;
- API key: any value with the default no-auth launcher;
- model: `tts-1`;
- voice: `alloy`, `lucia-ar`, `lucia-latam`, or another exposed profile.

One endpoint can therefore serve both English and Spanish; the selected voice determines the profile.

## Configuration

See `multilingual.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `BASE_MODEL_REPO` | `ResembleAI/chatterbox-turbo` | Official model repository |
| `BASE_MODEL_REVISION` | `main` | Official model revision |
| `BASE_MODEL_DIR` | empty | Local official model directory |
| `SPANISH_ENABLED` | `1` | Enable Spanish profile routing |
| `SPANISH_MODEL_REPO` | `groxaxo/chaturbo-espanol` | Spanish artifact repository |
| `SPANISH_MODEL_REVISION` | `main` | Spanish artifact revision |
| `SPANISH_MODEL_DIR` | empty | Local Spanish artifact directory |
| `DEFAULT_SPANISH_PROFILE` | `lucia-ar` | Target for aliases such as `lucia` and `es` |
| `SPANISH_PROFILE_CACHE_SIZE` | `1` | Number of Spanish T3 variants retained per worker |
| `PRELOAD_PROFILES` | empty | Comma-separated Spanish profiles loaded at worker startup |
| `STRICT_SPANISH_TAGS` | `1` | Reject unknown bracketed Turbo tags |
| `VERIFY_MODEL_PROVENANCE` | `1` | Verify adapter and AR warm-base hashes when provenance exists |
| `ENABLE_WATERMARK` | `0` | Enable the upstream Perth implicit watermark post-processing stage |
| `ENABLE_TURBO_FAST_PATH` | `1` | Cache encoded conditioning and remove exact-output-neutral Turbo loop overhead |
| `WORKER_LAZY_LOAD_MODEL` | `0` | Set to `1` to defer loading the GPU model until the first request |
| `WORKER_MODEL_IDLE_UNLOAD_SECONDS` | `0` | Unload a resident worker model after this idle period; `0` disables unloading |
| `WORKER_STARTUP_WARMUP` | `1` | Run a synthesis warmup when the worker starts |
| `HF_TOKEN` | empty | Hugging Face authentication token |

Optional persona overrides:

- `LUCIA_REFERENCE_WAV`;
- `LUCIA_CONDITIONING_PT`;
- per-profile variants such as `LUCIA_AR_REFERENCE_WAV` and `LUCIA_AR_CONDITIONING_PT`.

Use the released persona files unless there is a deliberate deployment-specific replacement.

Watermarking is disabled by default. With `ENABLE_WATERMARK=0`, the worker replaces the Perth
watermarker before constructing any English or Spanish engine, so PerthNet is not loaded and the
synthesized waveform is returned unchanged. Set `ENABLE_WATERMARK=1` and restart the API and workers
only when implicit watermarking is required. This setting does not alter T3/S3 generation, voice
conditioning, sampling parameters, or output encoding.

`GET /status` reports the API process setting as `api_watermark_mode`. Buffered synthesis responses
also return `X-Worker-Watermark-Modes`, which confirms the effective mode inside the GPU worker that
generated the audio.

The default Turbo fast path adapts two ideas from `groxaxo/chatterbox-vllm2` without replacing the
Turbo model or installing vLLM: workers cache final encoded voice conditioning, and autoregressive
generation avoids rebuilding token history on every step. It also suppresses production progress
rendering and skips a redundant CUDA synchronization after the waveform has already moved to CPU.
Source fingerprints guard the compatibility patch against upstream implementation changes. Set
`ENABLE_TURBO_FAST_PATH=0` and restart to use the unmodified installed runtime.

Production workers load and warm the model before serving queued synthesis work and remain resident
by default. Bootstrap runs while the Celery module is importing, before the worker can announce
readiness; model-load, compatibility, OOM, or warmup failures therefore terminate the worker instead
of exposing a cold or broken queue consumer. This moves model loading, voice conditioning, and CUDA
warmup out of user request latency.

## VRAM/cache policy

English remains resident as the shared backbone. Spanish profiles add a separate T3/LoRA model plus conditionals.

For 12 GB workers:

```bash
SPANISH_PROFILE_CACHE_SIZE=1
PRELOAD_PROFILES=
```

This retains English plus the most recently used Spanish profile. Switching Spanish profiles evicts only the least-recently-used Spanish engine.

For larger GPUs:

```bash
SPANISH_PROFILE_CACHE_SIZE=2
PRELOAD_PROFILES=lucia-ar,lucia-latam
```

## Systemd

```bash
sudo ./install_systemd_services.sh
```

Installed units:

- `chatterbox-turbo-fastapi.service`;
- `chatterbox-turbo-celery@.service`;
- `chatterbox-turbo.target`.

Environment file:

```text
/etc/chatterbox-turbo-fastapi.env
```

Existing environment files are preserved during upgrades. Add the multilingual variables manually when one already exists.

## Test

API smoke tests:

```bash
./test_curl.sh
```

Skip actual Spanish generation while checking only API wiring:

```bash
RUN_SPANISH_TESTS=0 ./test_curl.sh
```

Unit tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Model correctness notes

- The official base model is always required.
- `lucia-ar/t3_turbo_finetuned_merged.safetensors` is strict-loaded into the native Turbo T3.
- The merged checkpoint's 298-key structure is expected because the live Turbo T3 removes `tfmr.wte.weight` after native embedding setup.
- LATAM and pilot adapters are loaded only after strict-loading the Lucía AR merged warm base.
- `adapter_provenance.json`, when present, is used to verify both adapter and warm-start SHA-256 values before PEFT loading.
- Spanish text is NFC-normalized while preserving accents, `ñ`, `ü`, inverted punctuation, and recognized Turbo tags.
- Numbers, dates, currencies, abbreviations, and phone numbers are not automatically rewritten because that can change the intended speech.
