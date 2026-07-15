# Chatterbox Turbo Runtime And Benchmark Notes

Updated: 2026-06-15

## Runtime shape

- API: `http://127.0.0.1:7766`
- Endpoint: `POST /v1/audio/speech`
- Queue: `chatterbox_tts`
- Workers: 1 active on physical GPU `2` (RTX 3090 24 GB)
- GPU policy: RTX 3090, `EXPECTED_GPU_NAME=RTX 3090` in `/etc/chatterbox-turbo-fastapi.env`
- Systemd: `chatterbox-turbo-celery@2.service` (system unit); target updated from `@4` → `@2`
- Lazy behavior: API stays CPU-side; workers load the model on first request and unload after `300` seconds idle
- VRAM gate: `MIN_FREE_VRAM_MB=3500` before model load
- Chunking: sentence-boundary split, dynamic Celery dispatch, ordered PCM stitching in the API process
- Result collection: chunk task results are polled with `ready()` and read from `task.result`, avoiding Celery Redis `task.get()` hangs on large fan-out

## GPU 2 warm inference benchmark — 2026-06-15 (RTX 3090)

Test: 3 representative phone-call phrases, model already warm.

| Phrase type | Chars | Latency | Audio | RTF |
|---|---:|---:|---:|---:|
| Short | 32 | 811 ms | 3.1 s | 0.26 |
| Medium | 85 | 859 ms | 4.4 s | 0.20 |
| Phone-like | 144 | 1 313 ms | 7.4 s | 0.18 |
| **median** | | **859 ms** | | |

Cold-start (lazy load on first request): ~660 ms added to first call.

GPU 2 VRAM after model load: ~10.7 GB (out of 24 GB).

## 100-sentence benchmark (historical — RTX 3060 reference machine)

| Metric | Value |
| --- | ---: |
| Audio duration | `286.020s` |
| Client wall time | `37.493s` |
| Client speed | `7.6286x` realtime |
| RTF | `0.1311` |
| Chunks / tasks | `50 / 50` |

## Notes

- On RTX 3090 (24 GB) the model fits comfortably; `MIN_FREE_VRAM_MB=3500` gate is rarely triggered.
- The `chatterbox-turbo.target` `Wants=` was pointing at `@4` (non-existent GPU); corrected to `@2` on 2026-06-15.
- `/etc/chatterbox-turbo-fastapi.env` and `run_celery_worker.sh` default both updated from `RTX 3060` → `RTX 3090`.
- The high ASR WER in the 100-sentence test is from repeated number words; ASR speed figure is still valid as a correlation reference.

## Perth watermark bypass

The deployment defaults to `ENABLE_WATERMARK=0`. The worker replaces the upstream Perth constructor
before creating any Turbo engine, which prevents PerthNet initialization and returns the generated
waveform unchanged. Before the bypass, Perth accounted for 14 of 496 warm py-spy samples (2.8%) and
approximately 0.6 seconds of cold initialization. Post-change measurements must use the same model,
reference voice, text, seed, generation settings, GPU, and worker concurrency.

Live verification on 2026-07-15 used Chatterbox `0.1.6`, Perth `1.0.1`, and physical GPU 2:

| Seeded workload | Watermark enabled | Watermark disabled | Change |
|---|---:|---:|---:|
| 28.34s audio, 2 chunks, warm | 5.3538s (RTF 0.1889) | 4.8416s (RTF 0.1708) | 9.6% less wall time |
| 4.88s audio, 1 chunk, warm median of 5 | n/a | 0.7072s (RTF 0.1449) | 6.90x realtime |

The disabled warm py-spy capture contained zero Perth/watermark samples. Both comparable long outputs
had the same 24 kHz sample rate, 680,160 frames, 28.34-second duration, and finite samples. Cold timing
remains dominated by model loading and voice conditioning and was too variable for an attributable
before/after result.
