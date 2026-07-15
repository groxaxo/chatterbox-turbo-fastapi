# Multilingual runtime architecture

## Goals

1. Preserve the existing English FastAPI, streaming, chunking, voice-cache, and Celery behavior.
2. Add Lucía profiles without treating fine-tuned artifacts as standalone models.
3. Reconstruct the exact continual-training ancestry for every LoRA.
4. Serve English and Spanish from one endpoint and worker pool.
5. Bound VRAM usage with deterministic LRU eviction.

## Request routing

The OpenAI `voice` field selects the engine.

- Normal voice names and local audio filenames follow the original English path.
- Reserved Lucía profile names become internal marker paths.
- Existing API/Celery payloads already transport `voice_path`, so markers survive chunk dispatch without changing `server.py` or its task schema.
- The patched `generate_chunk_locked` resolves the marker and selects the correct T3.

Markers are control values and are never dereferenced as files.

## Artifact chain

The official `ResembleAI/chatterbox-turbo` snapshot is always required. It supplies:

- the native 50,276-token T3 architecture;
- S3Gen;
- the voice encoder;
- tokenizer files.

Profile reconstruction is:

```text
English:
  official Turbo T3

lucia-ar:
  official Turbo architecture
    -> strict-load lucia-ar/t3_turbo_finetuned_merged.safetensors

lucia-latam / lucia-cl-pilot / lucia-co-pilot:
  official Turbo architecture
    -> strict-load lucia-ar/t3_turbo_finetuned_merged.safetensors
    -> PeftModel.from_pretrained(profile adapter, is_trainable=False)
```

The LATAM and pilot adapters are continual LoRAs trained on the AR-merged frozen base. Applying them directly to the untouched official T3 would reconstruct the wrong model.

`chaturbo_espanol_runtime.py` enforces this chain. When `adapter_provenance.json` is present and `VERIFY_MODEL_PROVENANCE=1`, it verifies:

- `adapter_model.safetensors` SHA-256;
- the Lucía AR warm-start checkpoint SHA-256.

A mismatch fails before PEFT loading.

## Watermarking

`server.py` configures the shared Chatterbox class before any English or Spanish engine is created.
By default, `ENABLE_WATERMARK=0` replaces the upstream Perth watermarker with an identity adapter,
avoiding PerthNet model loading and post-generation processing. Setting `ENABLE_WATERMARK=1` restores
the upstream behavior after the API and workers are restarted. The setting does not change the T3,
S3Gen, or persona-conditioning paths described above.
The API status identifies its local mode as `api_watermark_mode`; buffered synthesis responses carry
`X-Worker-Watermark-Modes` so deployments can verify the effective GPU-worker mode.

## Turbo fast path

`performance_runtime.py` applies an exact-output compatibility layer after multilingual routing is
constructed. Inspired by the conditioning cache and persistent serving design in
`groxaxo/chatterbox-vllm2`, it caches each T3 engine's final encoded conditionals and removes repeated
token-history allocation and progress rendering from Turbo decoding. It does not transplant vLLM,
change attention, alter sampling, reduce MeanFlow steps, or change precision. Package-version and
source-marker guards fall back to upstream behavior if the pinned methods change.

English and every Spanish profile retain separate T3 instances and therefore separate encoded
conditioning caches. Worker response metadata reports the effective fast-path mode.
Local base-model overrides are validated for the three Turbo weight files and the tokenizer
configuration, vocabulary, merges, and special-token map before engine construction.
The Celery entrypoint performs model load and warmup during module bootstrap, before worker readiness.
Bootstrap exceptions are intentionally not swallowed, so systemd restarts a failed worker rather than
leaving a queue consumer that cannot synthesize.

## Shared components

Each Spanish profile is first constructed with a temporary CPU loader. Only its final T3 is retained.

The final Spanish engine shares the English engine's:

- S3Gen decoder;
- voice encoder;
- tokenizer.

It owns its own:

- merged or PEFT-wrapped T3;
- Lucía conditionals;
- configured watermarker instance (the default is the no-op adapter described above).

This avoids loading a full decoder and voice encoder for every profile.

## Persona conditioning

The runtime calls `prepare_conditionals(reference.wav)` so S3Gen receives a complete reference dictionary. It then replaces the T3 speaker embedding and prompt tokens with the released `conditioning.pt`, matching the source inference contract.

The pilot directories contain adapters only, so they inherit the Lucía LATAM persona files while retaining their own regional T3 delta.

## Concurrency

All generation within a worker remains protected by the existing `model_lock`, because engines share S3Gen and the voice encoder.

Celery workers run `--pool=solo --concurrency=1`. Parallelism comes from separate physical-GPU workers and sentence-level task distribution.

## Cache and unload behavior

The English engine remains `server.model`. Spanish engines use an ordered LRU cache.

- `SPANISH_PROFILE_CACHE_SIZE=1` retains English plus one Spanish profile.
- Loading another Spanish profile evicts only the least-recently-used Spanish T3.
- The existing idle unloader clears Spanish profiles before unloading English.
- Shutdown follows the same path.
- Failed profile loads deterministically release temporary CPU/GPU objects.

## Downloader dependency closure

`download_models.py` expands requested artifacts:

- any continual LoRA includes `lucia-ar` for the merged warm base;
- any pilot includes `lucia-latam` for persona conditioning;
- the official base snapshot is always downloaded separately.

This prevents an apparently successful offline download from failing later because a transitive model artifact is missing.

## Compatibility layer

`server.py` is intentionally not forked. `multilingual_runtime.py` patches the small set of hooks that the existing functions resolve dynamically:

- `ensure_model_loaded_locked`;
- `normalize_voice_path`;
- `generate_chunk_locked`;
- `unload_model_locked`;
- `available_voices_payload`;
- `runtime_status`;
- `run_generation`;
- `response_with_audio`;
- `warm_worker_model_if_needed`.

`ChaturboEspanolRuntime` subclasses that generic layer and overrides only artifact loading, profile metadata, and provenance status. This minimizes divergence from the mature English server while keeping the source repository's continual-learning contract explicit.
