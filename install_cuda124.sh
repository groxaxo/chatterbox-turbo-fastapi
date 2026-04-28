#!/usr/bin/env bash
set -euo pipefail

conda create -n chatterbox-turbo-api python=3.11 -y
conda activate chatterbox-turbo-api

python -m pip install -U pip wheel setuptools
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-api.txt

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
from chatterbox.tts_turbo import ChatterboxTurboTTS
print('ChatterboxTurboTTS import OK')
PY
