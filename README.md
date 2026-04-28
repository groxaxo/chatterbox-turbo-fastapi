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
API_KEY=change-this-key ./run_3090_or_3060.sh
```

## Test

```bash
./test_curl.sh
```

## Notes

- Use Python 3.11.
- Use one Uvicorn worker per GPU.
- For multiple GPUs, run multiple processes on different ports.
- The launcher defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce long-run CUDA allocator fragmentation.
- Keep realtime call chunks around 1-3 short sentences. Long paragraphs destroy latency.
