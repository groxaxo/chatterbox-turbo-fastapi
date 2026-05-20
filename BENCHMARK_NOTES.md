# Chatterbox Turbo Runtime And Benchmark Notes

Updated: 2026-05-15

## Runtime shape

- API: `http://127.0.0.1:7766`
- Endpoint: `POST /v1/audio/speech`
- Queue: `chatterbox_tts`
- Workers: 2 total, one on physical GPU `2`, one on physical GPU `3`
- GPU policy: RTX 3060 only, with `EXPECTED_GPU_NAME=RTX 3060`
- Lazy behavior: API stays CPU-side; workers load the model on first request and unload after `300` seconds idle
- VRAM gate: `MIN_FREE_VRAM_MB=3500` before model load
- Chunking: sentence-boundary split, dynamic Celery dispatch, ordered PCM stitching in the API process
- Result collection: chunk task results are polled with `ready()` and read from `task.result`, avoiding Celery Redis `task.get()` hangs on large fan-out

## 100-sentence benchmark

Test input: 100 English dot-terminated sentences.

| Metric | Value |
| --- | ---: |
| Audio duration | `286.020s` |
| Client wall time | `37.493s` |
| Client speed | `7.6286x` realtime |
| Server speed | `7.6338x` realtime |
| RTF | `0.1311` |
| Chunks / tasks | `50 / 50` |
| Worker split | `pid:1245018=21`, `pid:1245036=29` |
| ASR speed on Parakeet `:5092/v1` | `22.1928x` realtime |
| ASR WER | `0.4700` |

Output files:

- `/home/op/tts_unified_benchmark_outputs/chatterbox_turbo_100_sentences.wav`
- `/home/op/tts_unified_benchmark_outputs/chatterbox_turbo_100_sentences_asr.txt`
- `/home/op/tts_unified_benchmark_outputs/unified_tts_100_sentence_benchmark_results.json`

## Notes

- Earlier tests showed 4 workers across the same two 3060s were slower than 2 workers.
- The current deployed target remains 2 workers across 2x RTX 3060.
- The high ASR WER on the 100-sentence test is mostly from artificial repeated number words; the ASR speed is still useful for correlation.
