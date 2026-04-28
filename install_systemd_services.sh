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
  API_KEY_VALUE="$(openssl rand -hex 32)"
  DEFAULT_VOICE_VALUE="/home/op/chatterbox-turbo-fastapi/voices/default.wav"
  if [[ -f /home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav ]]; then
    DEFAULT_VOICE_VALUE="/home/op/Libro-Gregoria-Variacion/audio/britishWoman_clean.wav"
  fi
  cat >/etc/chatterbox-turbo-fastapi.env <<EOF
API_KEY=${API_KEY_VALUE}
ALLOW_NO_AUTH=0
PORT=7766
VOICE_DIR=/home/op/chatterbox-turbo-fastapi/voices
DEFAULT_VOICE=${DEFAULT_VOICE_VALUE}
ENABLE_CELERY=1
CELERY_BROKER_URL=redis://127.0.0.1:6379/14
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/14
CELERY_QUEUE=chatterbox_tts
LAZY_LOAD_MODEL=1
MODEL_IDLE_UNLOAD_SECONDS=900
MODEL_IDLE_CHECK_INTERVAL_SECONDS=30
AUTO_CHUNK_ENABLED=1
AUTO_CHUNK_TARGET_CHARS=520
AUTO_CHUNK_HARD_LIMIT=580
CHUNK_PAUSE_MS=140
MAX_INPUT_CHARS=12000
MAX_TEXT_CHARS=520
MAX_UPLOAD_MB=25
VOICE_CACHE_SIZE=8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXPECTED_GPU_NAME=RTX 3060
EOF
  chmod 600 /etc/chatterbox-turbo-fastapi.env
fi

systemctl daemon-reload
systemctl enable --now chatterbox-turbo.target
