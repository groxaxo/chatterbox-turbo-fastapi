# Chatterbox Turbo exact-output performance path

This repository includes an opt-in-at-runtime, enabled-by-default performance layer for the reviewed `chatterbox-tts==0.1.6` and `0.1.7` Turbo paths. It is deliberately limited to changes intended to preserve generated speech-token IDs and waveform bytes for fixed seeds and settings.

The layer is installed after multilingual routing, so it covers both the official English engine and every Lucía Spanish profile without modifying `site-packages`.

## Implemented changes

### Exact-output-oriented hot-path changes

- Removes the server-level `torch.cuda.synchronize()` performed after package generation has already returned a CPU waveform.
- Caches the encoded output of `T3.prepare_conditioning()` per immutable conditioning object and model state.
- Invalidates the cache when conditioning tensors, conditioning modules, embedding modules, device/dtype identity, or parameter versions change.
- Replaces per-token `torch.cat(generated_speech_tokens)` history rebuilding with a preallocated batch-size-one token buffer and identical slices passed to the same Transformers logits processors.
- Disables `tqdm` in production while retaining an environment-controlled diagnostic path.
- Reuses the three-token S3 terminal-silence tensor per device.
- Avoids a redundant speech-token `.to(device)` when the generated tensor is already on the target device.
- Increases the default idle residency period from 300 seconds to 1,800 seconds while preserving explicit environment overrides.

### Explicitly not enabled by the exact-output layer

The exact-output performance layer does **not** enable:

- SDPA;
- `torch.compile`;
- static KV cache or CUDA graphs;
- FP16/BF16 whole-model inference;
- quantization;
- request microbatching;
- altered sampling defaults;
- altered chunk boundaries;
- fewer S3/MeanFlow steps.

Those are separate numerical- or quality-parity experiments and must not be mixed into the exact-output rollout. The optional TensorRT and vLLM integrations are installed by `acceleration_runtime.py`, default to `TURBO_ACCELERATOR=torch`, and are documented separately in `ACCELERATION.md`.

## Compatibility safety

The source-sensitive rewrites require both:

1. an installed distribution version in `TURBO_EXPECTED_CHATTERBOX_VERSIONS`, defaulting to `0.1.6,0.1.7`; and
2. expected structural markers in the installed `prepare_conditioning`, `inference_turbo`, and `generate` source.

When either guard fails, the service logs a warning and continues with the original package implementation. Set `TURBO_FAIL_ON_INCOMPATIBLE_PACKAGE=1` to fail startup instead.

The legacy singular `TURBO_EXPECTED_CHATTERBOX_VERSION` variable is still honored when explicitly set and overrides the allowlist. This preserves old deployments while fixing fresh `0.1.7` installs that previously fell back because the runtime expected only `0.1.6`.

The performance `generate` wrapper calls the engine's currently configured `watermarker`. This keeps it compatible with deployments that replace Perth with a no-op watermarker before model construction.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `TURBO_PERFORMANCE_RUNTIME` | `1` | Install the exact-output performance layer |
| `TURBO_EXPECTED_CHATTERBOX_VERSIONS` | `0.1.6,0.1.7` | Reviewed package-version allowlist |
| `TURBO_EXPECTED_CHATTERBOX_VERSION` | unset | Legacy single-version override |
| `TURBO_FAIL_ON_INCOMPATIBLE_PACKAGE` | `0` | Fail instead of falling back when guards fail |
| `TURBO_CACHE_ENCODED_CONDITIONING` | `1` | Cache `cond_enc` output |
| `TURBO_ENCODED_CONDITION_CACHE_SIZE` | `VOICE_CACHE_SIZE` | Maximum encoded condition entries per T3 engine |
| `TURBO_PREALLOCATE_TOKEN_IDS` | `1` | Replace repeated generated-ID concatenation for batch size one |
| `TURBO_CACHE_SILENCE_TENSOR` | `1` | Reuse S3 terminal-silence tokens |
| `TURBO_DISABLE_PROGRESS` | `1` | Disable hot-path `tqdm` rendering |
| `TURBO_STRICT_LOGIT_CHECKS` | `1` | Preserve the package's all-`-inf` malformed-path check |
| `TURBO_REWRITE_PACKAGE_GENERATE` | `1` | Enable guarded terminal-silence/device cleanup |
| `MODEL_IDLE_UNLOAD_SECONDS` | `1800` | Keep engines resident for 30 minutes by default; `0` disables idle eviction |

The all-`-inf` check may synchronize CUDA because its result is consumed by Python. Keep it enabled during exact-parity validation. Disabling it changes only an invalid-generation diagnostic path, but should be handled as a separate rollout decision.

## Runtime status

`GET /status` includes a `performance` object with:

- Python, Torch, Transformers, Chatterbox, CUDA, and declared package requirements;
- package/source compatibility state;
- effective feature flags;
- active attention backend;
- per-engine cache-hit and fallback counters;
- confirmation that SDPA, compilation, and microbatching remain disabled in the exact-output layer.

When `acceleration_runtime.py` is installed, the endpoint also includes an `acceleration` object describing the selected backend, TensorRT compile/fallback state, or vLLM sidecar counters.

Example:

```bash
curl -s http://127.0.0.1:7766/status | jq '{performance, acceleration}'
```

## Capture the runtime manifest

Run this before changing the deployed environment:

```bash
./scripts/capture_runtime_manifest.sh
```

The output directory contains package metadata, Torch/CUDA configuration, GPU details, relevant environment flags, and SHA-256 hashes for configured model/profile/voice directories.

An explicit destination is also supported:

```bash
./scripts/capture_runtime_manifest.sh artifacts/pre-performance-runtime
```

## Exact-parity benchmark

The benchmark launches two baseline captures and one optimized capture in isolated Python processes.
It enables deterministic CUDA algorithms, disables TF32/cuDNN autotuning for the parity gate, and
pins the cuBLAS workspace configuration before CUDA initialization. Candidate approval requires both
baseline-to-baseline reproducibility and baseline-to-candidate parity. It records:

- engine-load, generation, conditioning, T3, S3, and PCM conversion timings;
- CUDA-event timings where CUDA is available;
- generated speech-token count and SHA-256;
- float waveform and PCM16 SHA-256;
- duration, real-time factor, finite values, clipping ratio, and peak CUDA memory;
- cache and instrumentation call counts.

Copy and edit the supplied case file so it references profiles available on the worker:

```bash
cp benchmarks/parity_cases.example.json benchmarks/parity_cases.local.json
```

Run the strict comparison:

```bash
python benchmarks/benchmark_parity.py compare \
  --cases benchmarks/parity_cases.local.json \
  --runs 3 \
  --output-dir artifacts/turbo-parity
```

The command exits non-zero when speech-token, float-waveform, PCM16, sample-rate, or sample-count parity differs. The generated reports are:

```text
artifacts/turbo-parity/baseline.json
artifacts/turbo-parity/baseline-repeat.json
artifacts/turbo-parity/optimized.json
artifacts/turbo-parity/comparison.json
```

Use `--allow-drift` only for later numerical-parity experiments. It should not be used to approve the exact-output phase.

For TensorRT, use the dedicated strict harness:

```bash
python benchmarks/benchmark_acceleration.py compare \
  --cases benchmarks/parity_cases.local.json \
  --runs 3 \
  --candidate tensorrt \
  --output-dir artifacts/tensorrt-parity
```

## Recommended rollout

1. Capture the current runtime manifest.
2. Run the benchmark with English and all deployed Spanish profiles.
3. Require exact token and waveform parity for every fixed-seed case.
4. Re-run warm and cold tests after worker restarts.
5. Deploy to one worker with `TURBO_ACCELERATOR=torch` unless an alternative backend has passed its separate gate.
6. Confirm cache and acceleration counters and latency through `/status` and normal telemetry.
7. Roll out to the remaining workers only after no profile-specific regression appears.

## Rollback

Disable the exact-output layer without changing code:

```bash
TURBO_PERFORMANCE_RUNTIME=0
```

Disable an acceleration candidate independently:

```bash
TURBO_ACCELERATOR=torch
```

Individual exact-output changes can also be disabled separately with their corresponding flags. Restart the affected worker after changing environment variables.
