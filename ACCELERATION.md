# TensorRT and vLLM acceleration recipes

The production default remains the resident PyTorch/Celery path:

```bash
TURBO_ACCELERATOR=torch
```

Both alternative backends are opt-in, observable through `GET /status`, and designed to fall back to the proven local worker unless `TURBO_ACCELERATOR_FAIL_CLOSED=1` is explicitly selected.

## Decision summary

| Backend | Scope | Output contract | Recommended use |
|---|---|---|---|
| `torch` | Existing T3 + S3Gen path | Existing exact-output contract | Default production path |
| `tensorrt` | FP32 HiFT/mel-to-wave vocoder boundary only | Must pass strict token and waveform parity | Single-request latency experiment |
| `vllm` | External T3/TTS sidecar over HTTP | Quality-gated; not byte-exact | Concurrent request batching on a separately managed GPU/process |

TensorRT is deliberately not applied to the autoregressive T3 loop. T3 has dynamic KV-cache and token-by-token sampling behavior, while the current exact-output runtime already removes its avoidable Python allocation overhead. The first TensorRT candidate is the deterministic, exportable HiFT vocoder boundary. If the installed S3Gen package does not expose that boundary, the worker records the incompatibility and stays on PyTorch.

vLLM is deliberately not installed into the Chatterbox FastAPI environment. The custom Chatterbox T3 implementation depends on vLLM internals and a different pinned Torch/Transformers stack. It must run as a sidecar, preferably from `groxaxo/chatterbox-vllm2`, and the FastAPI/Celery worker communicates with it over the OpenAI-compatible speech endpoint.

## TensorRT recipe

### 1. Install the optional worker dependencies

Activate the same environment used by the Celery worker, then run:

```bash
./install_tensorrt.sh
```

The installer reads the already-installed Torch and CUDA runtime, selects a reviewed matching Torch-TensorRT release, derives the correct PyTorch CUDA wheel index, and verifies that installation did not change Torch. The reviewed mappings are:

| Installed Torch family | Torch-TensorRT |
|---|---|
| 2.6 | 2.6.1 |
| 2.7 | 2.7.0 |
| 2.8 | 2.8.0 |
| 2.9 | 2.9.0 |
| 2.10 | 2.10.0 |
| 2.11 | 2.11.0 |
| 2.12 | 2.12.1 |

An unknown future Torch family fails safely. After reviewing upstream compatibility, it can be supplied explicitly:

```bash
TORCH_TENSORRT_VERSION=2.13.0 ./install_tensorrt.sh
```

A custom wheel index can also be supplied with `TORCH_TENSORRT_INDEX_URL`. The installer rejects a Torch-TensorRT version whose major/minor does not match the installed Torch family.

### 2. Run the strict parity gate

Prepare the local cases file if it does not exist:

```bash
cp benchmarks/parity_cases.example.json benchmarks/parity_cases.local.json
```

Run PyTorch twice for reproducibility, then compare it with the TensorRT candidate:

```bash
python benchmarks/benchmark_acceleration.py compare \
  --cases benchmarks/parity_cases.local.json \
  --runs 3 \
  --candidate tensorrt \
  --output-dir artifacts/tensorrt-parity
```

Approval requires all of the following for every fixed-seed English and deployed Spanish case:

- baseline-to-baseline token and waveform equality;
- PyTorch-to-TensorRT speech-token equality;
- float waveform and PCM16 byte equality;
- identical sample rate and sample count;
- no TensorRT fallback recorded in `/status`;
- a measured warm latency improvement.

Do not use `--allow-drift` for production approval.

### 3. Enable one canary worker

```bash
export TURBO_ACCELERATOR=tensorrt
export TURBO_ACCELERATOR_FAIL_CLOSED=1
export TURBO_TENSORRT_BACKEND=torch_tensorrt
export TURBO_TENSORRT_DYNAMIC=0
export TURBO_TENSORRT_FULLGRAPH=0
export TURBO_TENSORRT_REQUIRE_FP32=1
./run_celery_worker.sh 2
```

Startup warmup triggers lazy compilation before the worker accepts traffic. Inspect:

```bash
curl -s http://127.0.0.1:7766/status | jq .acceleration
```

A healthy candidate reports `compile_succeeded: true` after warmup or the first synthesis. Any compile/runtime exception restores the original bound PyTorch method when fail-closed mode is not enabled.

### TensorRT rollback

```bash
export TURBO_ACCELERATOR=torch
```

Restart the affected worker. No model conversion artifact or site-package edit is required.

## vLLM sidecar recipe

### 1. Deploy the sidecar separately

Use a separate environment and preferably a separate GPU. The supported reference implementation is:

```text
groxaxo/chatterbox-vllm2
```

Expose its health endpoint and OpenAI-compatible endpoint only on loopback or a private service network. A typical local endpoint is:

```text
http://127.0.0.1:8000/v1/audio/speech
```

Do not add `vllm` to `requirements-api.txt`; keeping the runtimes isolated avoids Torch, Transformers, FlashInfer, and CUDA dependency collisions.

### 2. Enable sidecar routing on one worker

English is the only routed profile by default. Spanish Lucía profiles continue to use their exact local artifact chains.

```bash
export TURBO_ACCELERATOR=vllm
export TURBO_VLLM_BASE_URL=http://127.0.0.1:8000
export TURBO_VLLM_PROFILES=english
export TURBO_VLLM_VOICE_MAP=english=alloy
export TURBO_VLLM_MODEL=tts-1
export TURBO_VLLM_TIMEOUT_SECONDS=180
export TURBO_VLLM_EXCLUSIVE=0
export TURBO_ACCELERATOR_FAIL_CLOSED=0
./run_celery_worker.sh 2
```

With `TURBO_VLLM_EXCLUSIVE=0`, the local PyTorch engine remains warm and receives unsupported profiles or sidecar failures. This consumes more VRAM, so the recommended production topology is a sidecar on a different GPU/process. Set exclusive mode only after accepting that local fallback and Lucía routing are unavailable on that worker.

A custom `audio_prompt_path` works only when the sidecar can read the same absolute path, such as on the same host or a shared mount.

### 3. Quality and throughput gate

vLLM is not part of the strict byte-parity gate because its scheduler, RNG stream, token-offset model, and sampling implementation are distinct. Before routing production traffic, compare:

- transcript WER/CER on English and any explicitly enabled profile;
- speaker-embedding cosine similarity;
- silence, clipping, repetition, and long-tail rates;
- p50/p95 first-audio and completion latency;
- throughput at concurrency 1, 2, 4, and 8;
- fallback and failure counters in `/status`.

The bridge forwards temperature, top-p, repetition penalty, voice, and the reference path supported by the sidecar. Local `top_k`, seed reproduction, and cross-runtime RNG parity are explicitly not guaranteed and are reported in status metadata.

### vLLM rollback

```bash
export TURBO_ACCELERATOR=torch
```

Restart the worker. The sidecar can then be stopped independently.

## Configuration reference

| Variable | Default | Purpose |
|---|---:|---|
| `TURBO_ACCELERATOR` | `torch` | `torch`, `tensorrt`, or `vllm` |
| `TURBO_ACCELERATOR_FAIL_CLOSED` | `0` | Raise instead of using the local fallback |
| `TURBO_TENSORRT_BACKEND` | `torch_tensorrt` | Torch Dynamo backend name |
| `TURBO_TENSORRT_DYNAMIC` | `0` | Enable symbolic dynamic-shape compilation |
| `TURBO_TENSORRT_FULLGRAPH` | `0` | Require one complete compiled graph when enabled |
| `TURBO_TENSORRT_REQUIRE_FP32` | `1` | Reject reduced-precision HiFT parameters |
| `TURBO_VLLM_BASE_URL` | `http://127.0.0.1:8000` | Sidecar origin |
| `TURBO_VLLM_HEALTH_PATH` | `/health` | Bootstrap health check |
| `TURBO_VLLM_SPEECH_PATH` | `/v1/audio/speech` | OpenAI-compatible synthesis route |
| `TURBO_VLLM_PROFILES` | `english` | Comma-separated routed profiles |
| `TURBO_VLLM_VOICE_MAP` | `english=alloy` | Profile-to-sidecar voice mapping |
| `TURBO_VLLM_EXCLUSIVE` | `0` | Disable local warm fallback when set |
