"""Lightweight import fallbacks for pure runtime unit tests.

Production installs the real packages from requirements-api.txt. CI unit tests only
exercise profile routing, dependency closure, and normalization, so missing heavy
ML packages are replaced with minimal import stubs rather than downloading model
runtimes on every source-only check.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace


def _install_torch_stub() -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.inference_mode = nullcontext
    sys.modules["torch"] = torch


def _install_chatterbox_stub() -> None:
    if importlib.util.find_spec("chatterbox") is not None:
        return
    chatterbox = ModuleType("chatterbox")
    chatterbox.__path__ = []
    tts_turbo = ModuleType("chatterbox.tts_turbo")

    class ChatterboxTurboTTS:  # pragma: no cover - type/import placeholder only
        pass

    tts_turbo.ChatterboxTurboTTS = ChatterboxTurboTTS
    sys.modules["chatterbox"] = chatterbox
    sys.modules["chatterbox.tts_turbo"] = tts_turbo


def _install_huggingface_stub() -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        return
    module = ModuleType("huggingface_hub")
    module.snapshot_download = lambda **_: ""
    sys.modules["huggingface_hub"] = module


def _install_safetensors_stub() -> None:
    if importlib.util.find_spec("safetensors") is not None:
        return
    package = ModuleType("safetensors")
    package.__path__ = []
    torch_module = ModuleType("safetensors.torch")
    torch_module.load_file = lambda *_args, **_kwargs: {}
    sys.modules["safetensors"] = package
    sys.modules["safetensors.torch"] = torch_module


_install_torch_stub()
_install_chatterbox_stub()
_install_huggingface_stub()
_install_safetensors_stub()
