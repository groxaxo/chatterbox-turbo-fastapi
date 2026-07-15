#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

install -Dm644 systemd/chatterbox-turbo-fastapi.service /etc/systemd/system/chatterbox-turbo-fastapi.service
install -Dm644 systemd/chatterbox-turbo-celery@.service /etc/systemd/system/chatterbox-turbo-celery@.service
install -Dm644 systemd/chatterbox-turbo.target /etc/systemd/system/chatterbox-turbo.target

if [[ ! -f /etc/chatterbox-turbo-fastapi.env ]]; then
  DEFAULT_VOICE_VALUE="/home/op/chatterbox-turbo-fastapi/voices/default.wav"
  if [[ -f /home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav ]]; then
    DEFAULT_VOICE_VALUE="/home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav"
  fi
  cat >/etc/chatterbox-turbo-fastapi.env <<EOF_ENV
ALLOW_NO_AUTH=1
PORT=7766
VOICE_DIR=/home/op/chatterbox-turbo-fastapi/voices
DEFAULT_VOICE=${DEFAULT_VOICE_VALUE}

# Official English base model. Leave BASE_MODEL_DIR empty to use the HF cache.
BASE_MODEL_REPO=ResembleAI/chatterbox-turbo
BASE_MODEL_REVISION=main
BASE_MODEL_DIR=

# Lucía Spanish profiles. Leave SPANISH_MODEL_DIR empty to use the HF cache.
SPANISH_ENABLED=1
SPANISH_MODEL_REPO=groxaxo/chaturbo-espanol
SPANISH_MODEL_REVISION=main
SPANISH_MODEL_DIR=
DEFAULT_SPANISH_PROFILE=lucia-ar
SPANISH_PROFILE_CACHE_SIZE=1
PRELOAD_PROFILES=
STRICT_SPANISH_TAGS=1
VERIFY_MODEL_PROVENANCE=1
ENABLE_WATERMARK=0

# Exact-output Chatterbox Turbo hot-path optimizations.
TURBO_PERFORMANCE_RUNTIME=1
TURBO_EXPECTED_CHATTERBOX_VERSIONS=0.1.6,0.1.7
TURBO_FAIL_ON_INCOMPATIBLE_PACKAGE=0
TURBO_CACHE_ENCODED_CONDITIONING=1
TURBO_PREALLOCATE_TOKEN_IDS=1
TURBO_CACHE_SILENCE_TENSOR=1
TURBO_DISABLE_PROGRESS=1
TURBO_STRICT_LOGIT_CHECKS=1
TURBO_REWRITE_PACKAGE_GENERATE=1

# Optional acceleration; keep PyTorch until ACCELERATION.md gates pass.
TURBO_ACCELERATOR=torch
TURBO_ACCELERATOR_FAIL_CLOSED=0
TURBO_TENSORRT_BACKEND=torch_tensorrt
TURBO_TENSORRT_DYNAMIC=0
TURBO_TENSORRT_FULLGRAPH=0
TURBO_TENSORRT_REQUIRE_FP32=1
TURBO_VLLM_BASE_URL=http://127.0.0.1:8000
TURBO_VLLM_HEALTH_PATH=/health
TURBO_VLLM_SPEECH_PATH=/v1/audio/speech
TURBO_VLLM_MODEL=tts-1
TURBO_VLLM_PROFILES=english
TURBO_VLLM_VOICE_MAP=english=alloy
TURBO_VLLM_TIMEOUT_SECONDS=180
TURBO_VLLM_EXCLUSIVE=0

ENABLE_CELERY=1
CELERY_BROKER_URL=redis://127.0.0.1:6379/14
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/14
CELERY_QUEUE=chatterbox_tts
LAZY_LOAD_MODEL=1
MODEL_IDLE_UNLOAD_SECONDS=0
MODEL_IDLE_CHECK_INTERVAL_SECONDS=30
WORKER_LAZY_LOAD_MODEL=0
WORKER_MODEL_IDLE_UNLOAD_SECONDS=0
WORKER_STARTUP_WARMUP=1
MIN_FREE_VRAM_MB=3500
MODEL_LOAD_WAIT_TIMEOUT_SECONDS=60
AUTO_CHUNK_ENABLED=1
AUTO_CHUNK_TARGET_CHARS=520
AUTO_CHUNK_HARD_LIMIT=580
CHUNK_PAUSE_MS=140
MAX_INPUT_CHARS=12000
MAX_TEXT_CHARS=520
MAX_UPLOAD_MB=25
VOICE_CACHE_SIZE=8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXPECTED_GPU_NAME=RTX 3090
EOF_ENV
  chmod 600 /etc/chatterbox-turbo-fastapi.env
fi

systemctl daemon-reload
systemctl enable --now chatterbox-turbo.target
