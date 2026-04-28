# Chatterbox Turbo FastAPI

FastAPI wrapper for official ResembleAI Chatterbox Turbo.

Key optimization: default and repeated voices are preconditioned and cached. The server avoids calling `generate(audio_prompt_path=...)` on every request because upstream `generate()` re-runs `prepare_conditionals()` when an audio prompt is provided.

This build is configured for:

- **API port:** `7766`
- **Execution model:** FastAPI API + Celery workers + Redis broker
- **GPU policy:** workers only on the physical RTX **3060** GPUs (`2,3`)
- **Default OpenAI-style voice:** `alloy`
- **Default reference clip:** `/home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav` when present, else `voices/default.wav`
- **Default response format:** `mp3` (for Open WebUI compatibility)
- **Auto chunking target:** `520` chars
- **Auto chunking hard limit:** `580` chars

## Install

```bash
conda activate base
cd chatterbox-turbo-fastapi
./install_cuda124.sh
```

## Run

```bash
conda activate chatterbox-turbo-api
mkdir -p voices
cp /path/to/reference.wav voices/default.wav

# Production — API on :7766, Celery-backed, Open WebUI-compatible MP3 default:
API_KEY="$(openssl rand -hex 32)" ./run_api_service.sh

# Local dev only — skip authentication:
ALLOW_NO_AUTH=1 ./run_api_service.sh
```

`API_KEY` is **required** unless `ALLOW_NO_AUTH=1` is set. The server refuses to start with a missing key.

To run a worker manually on a specific physical 3060:

```bash
./run_celery_worker.sh 2
./run_celery_worker.sh 3
```

## Systemd install

```bash
sudo ./install_systemd_services.sh
```

That installs:

- `chatterbox-turbo-fastapi.service`
- `chatterbox-turbo-celery@.service`
- `chatterbox-turbo.target`

Runtime config is stored in:

```bash
/etc/chatterbox-turbo-fastapi.env
```

The generated env file defaults to port `7766`, queue `chatterbox_tts`, `alloy`/British-woman reference voice, and the measured chunking target (`520/580`).

## API endpoints

| Endpoint | Auth | Description |
|---|---|---|
| `GET /healthz` | ✗ | Liveness probe |
| `GET /health` | ✗ | Alias for `/healthz` |
| `GET /status` | ✓ | Runtime status (backend mode, chunking, worker GPU policy) |
| `GET /voices` | ✓ | Voice aliases + local voice files |
| `GET /v1/models` | ✓ | OpenAI-style model list |
| `GET /v1/audio/models` | ✓ | Open WebUI custom TTS model discovery |
| `GET /v1/audio/voices` | ✓ | Open WebUI custom TTS voice discovery |
| `POST /v1/audio/speech` | ✓ | OpenAI-compatible speech synthesis (`mp3` default, `wav` supported) |
| `POST /tts` | ✓ | Multipart speech synthesis (with voice upload) |
| `POST /warmup` | ✓ | Force a warmup inference pass |

## Authentication

Pass the key as `Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Open WebUI

Use these settings in **Settings -> Audio**:

- **TTS Engine:** `OpenAI`
- **OpenAI Base URL:** `http://127.0.0.1:7766/v1`
- **OpenAI API Key:** the value from `/etc/chatterbox-turbo-fastapi.env`
- **TTS Model:** `tts-1`
- **TTS Voice:** `alloy`

The server exposes the extra `/v1/audio/models` and `/v1/audio/voices` endpoints that Open WebUI uses for custom TTS providers.

For local STT correlation, the matching OpenAI-compatible Parakeet endpoint is:

```bash
http://127.0.0.1:5092/v1/audio/transcriptions
```

## Benchmark notes

Measured with Chatterbox audio transcribed back through Parakeet on port `5092`:

| Test | Input chars | Chunks | WER |
|---|---:|---:|---:|
| Short | 101 | 1 | 0.0000 |
| Medium | 255 | 1 | 0.0455 |
| Long | 943 | 2 | 0.0464 |

Chunk-target sweep on the long prompt:

| Target chars | Hard limit | Chunks | WER |
|---|---:|---:|---:|
| 220 | 280 | 5 | 0.0066 |
| 280 | 340 | 4 | 0.0132 |
| 320 | 380 | 4 | 0.0066 |
| 360 | 420 | 3 | 0.0132 |
| 420 | 480 | 3 | 0.0066 |
| 520 | 580 | 2 | **0.0066** |

`520 / 580` was chosen because it tied for the lowest WER while keeping the **longest** chunk size tested.

## Test

```bash
export API_KEY="your-key-here"
./test_curl.sh
```

## Notes

- Use Python 3.11.
- Keep the API process CPU-only when `ENABLE_CELERY=1`; only the Celery workers should see the 3060 GPUs.
- The launcher defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce long-run CUDA allocator fragmentation.
- The service lazy-loads the model and unloads it again after the configured idle timeout.
