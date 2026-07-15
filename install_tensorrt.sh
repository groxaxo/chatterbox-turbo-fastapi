#!/usr/bin/env bash
set -euo pipefail

# Install into the existing Chatterbox worker environment. Do not reinstall Torch:
# this recipe intentionally matches the repository's torch==2.6.x runtime.
python -m pip install --upgrade pip wheel 'setuptools<81'
python -m pip install \
  'torch-tensorrt==2.6.1' \
  tensorrt \
  --extra-index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
import torch_tensorrt

print("torch", torch.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in this environment")

list_backends = getattr(getattr(torch, "_dynamo", None), "list_backends", None)
backends = sorted(list_backends()) if callable(list_backends) else []
print("dynamo backends", backends)
if backends and not ({"tensorrt", "torch_tensorrt"} & set(backends)):
    raise SystemExit("ERROR: Torch-TensorRT Dynamo backend is not registered")
PY
