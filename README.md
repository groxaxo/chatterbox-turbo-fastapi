# Chatterbox Turbo FastAPI

FastAPI wrapper for official ResembleAI Chatterbox Turbo.

Key optimization: default and repeated voices are preconditioned and cached. The server avoids calling `generate(audio_prompt_path=...)` on every request because upstream `generate()` re-runs `prepare_conditionals()` when an audio prompt is provided.

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

# Production — generate a strong random key:
API_KEY="$(openssl rand -hex 32)" ./run_3090_or_3060.sh

# Local dev only — skip authentication:
ALLOW_NO_AUTH=1 ./run_3090_or_3060.sh
```

`API_KEY` is **required** unless `ALLOW_NO_AUTH=1` is set. The server refuses to start with a missing key.

## API endpoints

| Endpoint | Auth | Description |
|---|---|---|
| `GET /healthz` | ✗ | Liveness probe (`{"ok": true}`) |
| `GET /health` | ✗ | Alias for `/healthz` |
| `GET /status` | ✓ | Full runtime status (GPU, VRAM, cache) |
| `GET /voices` | ✓ | List available voice files |
| `POST /v1/audio/speech` | ✓ | OpenAI-compatible speech synthesis |
| `POST /tts` | ✓ | Multipart speech synthesis (with voice upload) |
| `POST /warmup` | ✓ | Force a warmup inference pass |

## Authentication

Pass the key as `Authorization: Bearer <key>` or `X-API-Key: <key>`.

## Test

```bash
export API_KEY="your-key-here"
./test_curl.sh
```

## Notes

- Use Python 3.11.
- Use one Uvicorn worker per GPU.
- For multiple GPUs, run multiple processes on different ports.
- The launcher defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce long-run CUDA allocator fragmentation.
- Keep realtime call chunks around 1–3 short sentences. Long paragraphs destroy latency.
