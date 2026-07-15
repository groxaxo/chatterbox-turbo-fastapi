#!/usr/bin/env bash
set -euo pipefail

# Install Torch-TensorRT into the existing Chatterbox worker environment without
# changing the installed Torch major/minor or CUDA wheel family.
python -m pip install --upgrade pip wheel 'setuptools<81'

mapfile -t RUNTIME < <(python - <<'PY'
from __future__ import annotations

import os
import re

import torch

public_torch = torch.__version__.split("+", 1)[0]
match = re.match(r"^(\d+)\.(\d+)\.(\d+)", public_torch)
if match is None:
    raise SystemExit(f"ERROR: Could not parse installed Torch version {torch.__version__!r}")

torch_family = f"{match.group(1)}.{match.group(2)}"
known = {
    "2.6": "2.6.1",
    "2.7": "2.7.0",
    "2.8": "2.8.0",
    "2.9": "2.9.0",
    "2.10": "2.10.0",
    "2.11": "2.11.0",
    "2.12": "2.12.1",
}
requested = os.environ.get("TORCH_TENSORRT_VERSION", "").strip()
tensorrt_version = requested or known.get(torch_family, "")
if not tensorrt_version:
    raise SystemExit(
        "ERROR: No reviewed Torch-TensorRT mapping exists for Torch "
        f"{public_torch}. Set TORCH_TENSORRT_VERSION explicitly after checking "
        "the Torch-TensorRT release compatibility table."
    )
if not tensorrt_version.startswith(f"{torch_family}."):
    raise SystemExit(
        "ERROR: TORCH_TENSORRT_VERSION must match the installed Torch major/minor: "
        f"torch={public_torch}, torch-tensorrt={tensorrt_version}."
    )

cuda = str(torch.version.cuda or "").strip()
if not cuda:
    raise SystemExit("ERROR: Installed Torch has no CUDA runtime.")
cuda_match = re.match(r"^(\d+)\.(\d+)", cuda)
if cuda_match is None:
    raise SystemExit(f"ERROR: Could not parse torch.version.cuda={cuda!r}")
wheel_family = f"cu{cuda_match.group(1)}{cuda_match.group(2)}"
index_url = os.environ.get(
    "TORCH_TENSORRT_INDEX_URL",
    f"https://download.pytorch.org/whl/{wheel_family}",
).strip()

print(public_torch)
print(torch_family)
print(tensorrt_version)
print(cuda)
print(index_url)
PY
)

TORCH_PUBLIC_VERSION="${RUNTIME[0]}"
TORCH_FAMILY="${RUNTIME[1]}"
TORCH_TENSORRT_VERSION="${RUNTIME[2]}"
TORCH_CUDA_VERSION="${RUNTIME[3]}"
TORCH_TENSORRT_INDEX_URL="${RUNTIME[4]}"

printf 'Installing Torch-TensorRT %s for Torch %s / CUDA %s\n' \
  "$TORCH_TENSORRT_VERSION" "$TORCH_PUBLIC_VERSION" "$TORCH_CUDA_VERSION"
printf 'Wheel index: %s\n' "$TORCH_TENSORRT_INDEX_URL"

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
printf 'torch==%s\n' "$TORCH_PUBLIC_VERSION" >"$CONSTRAINTS_FILE"

python -m pip install \
  --constraint "$CONSTRAINTS_FILE" \
  "torch-tensorrt==${TORCH_TENSORRT_VERSION}" \
  tensorrt \
  --extra-index-url "$TORCH_TENSORRT_INDEX_URL"

EXPECTED_TORCH_VERSION="$TORCH_PUBLIC_VERSION" \
EXPECTED_TORCH_FAMILY="$TORCH_FAMILY" \
EXPECTED_TORCH_TENSORRT_VERSION="$TORCH_TENSORRT_VERSION" \
python - <<'PY'
from __future__ import annotations

import os

import torch
import torch_tensorrt

expected_torch = os.environ["EXPECTED_TORCH_VERSION"]
expected_family = os.environ["EXPECTED_TORCH_FAMILY"]
expected_tensorrt = os.environ["EXPECTED_TORCH_TENSORRT_VERSION"]
actual_torch = torch.__version__.split("+", 1)[0]
actual_tensorrt = torch_tensorrt.__version__.split("+", 1)[0]

print("torch", torch.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())

if actual_torch != expected_torch:
    raise SystemExit(
        f"ERROR: Torch changed during installation: expected={expected_torch}, actual={actual_torch}"
    )
if actual_tensorrt != expected_tensorrt:
    raise SystemExit(
        "ERROR: Unexpected Torch-TensorRT version: "
        f"expected={expected_tensorrt}, actual={actual_tensorrt}"
    )
if not actual_tensorrt.startswith(f"{expected_family}."):
    raise SystemExit(
        f"ERROR: Torch/Torch-TensorRT families differ: {actual_torch} vs {actual_tensorrt}"
    )
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in this environment")

list_backends = getattr(getattr(torch, "_dynamo", None), "list_backends", None)
backends = sorted(list_backends()) if callable(list_backends) else []
print("dynamo backends", backends)
if backends and not ({"tensorrt", "torch_tensorrt"} & set(backends)):
    raise SystemExit("ERROR: Torch-TensorRT Dynamo backend is not registered")
PY
