# Multilingual runtime architecture

## Goals

1. Preserve every existing English FastAPI, streaming, chunking, voice-cache, and Celery behavior.
2. Add the Lucía Spanish artifacts without pretending that they are standalone models.
3. Keep the official `ResembleAI/chatterbox-turbo` base model as the explicit source of T3 architecture, S3Gen, voice encoder, and tokenizer.
4. Allow English and Spanish requests to share one endpoint and one worker pool.
5. Bound VRAM usage with deterministic LRU profile eviction.

## Request routing

The OpenAI `voice` field is the profile selector.

- Normal voice names and local audio filenames follow the original English path.
- Reserved Lucía profile names are encoded as internal marker paths.
- The existing API and Celery payloads already transport `voice_path`, so the marker survives chunk dispatch without changing `server.py` or the task schema.
- The patched `generate_chunk_locked` resolves the marker and dispatches to the correct T3 engine.

The marker is never dereferenced as a filesystem path.

## English engine

The English model is loaded from either `BASE_MODEL_DIR` or a Hugging Face snapshot of `ResembleAI/chatterbox-turbo`.

The runtime uses `ChatterboxTurboTTS.from_local` so the same resolved directory can construct Spanish T3 variants. English reference conditioning and the original LRU conditionals cache remain unchanged.

## Spanish engines

Each Spanish profile receives a fresh native Turbo T3 constructed from the official base directory.

- `lucia-ar`: strict merged-checkpoint load.
- `lucia-latam`, `lucia-cl-pilot`, `lucia-co-pilot`: PEFT adapter load with `is_trainable=False`.

After loading the profile T3, the temporary CPU engine is discarded. The final profile engine shares the English engine's:

- S3Gen decoder
- voice encoder
- tokenizer

It retains its own:

- T3 or PEFT-wrapped T3
- Lucía conditionals
- watermarker instance

## Persona conditioning

The runtime first calls `prepare_conditionals(reference.wav)` so S3Gen receives a complete reference dictionary. It then replaces the T3 speaker embedding and prompt tokens with `conditioning.pt`, matching the source repository's inference contract.

The pilot profiles inherit the LATAM conditioning bundle because their directories contain adapters only.

## Concurrency

All generation on a worker remains protected by the existing `model_lock`. This is required because English and Spanish engines share S3Gen and the voice encoder.

Celery workers use `--pool=solo --concurrency=1`, while separate physical-GPU workers provide request/chunk parallelism.

## Cache and unload behavior

The English engine remains `server.model`. Spanish engines are stored in an ordered LRU cache.

- `SPANISH_PROFILE_CACHE_SIZE=1` means English plus one Spanish profile can remain resident.
- Loading another Spanish profile evicts only the least recently used Spanish T3.
- The existing idle unloader now clears the Spanish cache before unloading English.
- Shutdown follows the same path.

## Compatibility

`server.py` is intentionally not forked. The multilingual layer monkey-patches the small set of runtime hooks looked up dynamically by the existing functions:

- `ensure_model_loaded_locked`
- `normalize_voice_path`
- `generate_chunk_locked`
- `unload_model_locked`
- `available_voices_payload`
- `runtime_status`
- `run_generation`
- `response_with_audio`
- `warm_worker_model_if_needed`

This minimizes divergence from the mature English server and keeps future server improvements reusable.
