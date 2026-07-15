#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/runtime-manifest-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"

python -VV >"$OUT_DIR/python.txt"
python -m pip freeze --all >"$OUT_DIR/pip-freeze.txt"
python -m pip inspect >"$OUT_DIR/pip-inspect.json" 2>"$OUT_DIR/pip-inspect.stderr" || true

python - <<'PY' >"$OUT_DIR/runtime.json"
from __future__ import annotations

import importlib.metadata
import json
import os
import platform

import torch


def version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def requirements(name: str):
    try:
        return importlib.metadata.requires(name) or []
    except importlib.metadata.PackageNotFoundError:
        return []


payload = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_tensorrt": version("torch-tensorrt"),
    "tensorrt": version("tensorrt"),
    "vllm": version("vllm"),
    "transformers": version("transformers"),
    "chatterbox_tts": version("chatterbox-tts"),
    "chatterbox_declared_requirements": requirements("chatterbox-tts"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_device_count": torch.cuda.device_count(),
    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "environment": {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TURBO_PERFORMANCE_RUNTIME",
            "TURBO_CACHE_ENCODED_CONDITIONING",
            "TURBO_PREALLOCATE_TOKEN_IDS",
            "TURBO_CACHE_SILENCE_TENSOR",
            "TURBO_DISABLE_PROGRESS",
            "TURBO_STRICT_LOGIT_CHECKS",
            "TURBO_EXPECTED_CHATTERBOX_VERSION",
            "TURBO_EXPECTED_CHATTERBOX_VERSIONS",
            "TURBO_ACCELERATOR",
            "TURBO_ACCELERATOR_FAIL_CLOSED",
            "TURBO_TENSORRT_BACKEND",
            "TURBO_TENSORRT_DYNAMIC",
            "TURBO_TENSORRT_FULLGRAPH",
            "TURBO_TENSORRT_REQUIRE_FP32",
            "TURBO_VLLM_BASE_URL",
            "TURBO_VLLM_HEALTH_PATH",
            "TURBO_VLLM_SPEECH_PATH",
            "TURBO_VLLM_MODEL",
            "TURBO_VLLM_PROFILES",
            "TURBO_VLLM_VOICE_MAP",
            "TURBO_VLLM_TIMEOUT_SECONDS",
            "TURBO_VLLM_EXCLUSIVE",
            "MODEL_IDLE_UNLOAD_SECONDS",
        )
    },
}
if torch.cuda.is_available():
    payload["devices"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": torch.cuda.get_device_capability(index),
        }
        for index in range(torch.cuda.device_count())
    ]

print(json.dumps(payload, indent=2, sort_keys=True))
PY

python - <<'PY' >"$OUT_DIR/torch-config.txt"
import torch
print(torch.__config__.show())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"$OUT_DIR/nvidia-smi-q.txt"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader >"$OUT_DIR/gpus.csv"
fi

HASH_PATHS=(
  "${BASE_MODEL_DIR:-}"
  "${SPANISH_MODEL_DIR:-}"
  "${VOICE_DIR:-}"
)
: >"$OUT_DIR/artifact-hashes.sha256"
for path in "${HASH_PATHS[@]}"; do
  [[ -n "$path" && -e "$path" ]] || continue
  if [[ -d "$path" ]]; then
    find "$path" -type f -print0 | sort -z | xargs -0 -r sha256sum \
      >>"$OUT_DIR/artifact-hashes.sha256"
  else
    sha256sum "$path" >>"$OUT_DIR/artifact-hashes.sha256"
  fi
done

printf '%s\n' "$OUT_DIR"
