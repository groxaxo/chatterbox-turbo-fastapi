"""Lightweight import fallbacks for pure runtime unit tests.

Production installs the real packages from requirements-api.txt. CI unit tests only
exercise routing, source guards, lifecycle wiring, and normalization, so missing
heavy ML packages are replaced with minimal import stubs rather than downloading
model runtimes on every source-only check.
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
    torch.__version__ = "unavailable-test-stub"
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        current_device=lambda: 0,
        synchronize=lambda: None,
    )
    torch.version = SimpleNamespace(cuda=None)
    torch.inference_mode = lambda: nullcontext()
    torch.is_grad_enabled = lambda: False

    nn = ModuleType("torch.nn")
    functional = ModuleType("torch.nn.functional")
    functional.softmax = lambda value, dim=-1: value

    class Module:
        def __init__(self, *_args, **_kwargs):
            pass

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    nn.Module = Module
    nn.functional = functional

    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional


def _install_chatterbox_stub() -> None:
    if importlib.util.find_spec("chatterbox") is not None:
        return

    chatterbox = ModuleType("chatterbox")
    chatterbox.__path__ = []

    tts_turbo = ModuleType("chatterbox.tts_turbo")

    class ChatterboxTurboTTS:  # pragma: no cover - type/import placeholder only
        pass

    tts_turbo.ChatterboxTurboTTS = ChatterboxTurboTTS
    tts_turbo.punc_norm = lambda text: text

    models = ModuleType("chatterbox.models")
    models.__path__ = []
    s3gen = ModuleType("chatterbox.models.s3gen")
    s3gen.__path__ = []
    const = ModuleType("chatterbox.models.s3gen.const")
    const.S3GEN_SIL = 0

    sys.modules["chatterbox"] = chatterbox
    sys.modules["chatterbox.tts_turbo"] = tts_turbo
    sys.modules["chatterbox.models"] = models
    sys.modules["chatterbox.models.s3gen"] = s3gen
    sys.modules["chatterbox.models.s3gen.const"] = const


def _install_transformers_stub() -> None:
    if importlib.util.find_spec("transformers") is not None:
        return

    package = ModuleType("transformers")
    package.__path__ = []
    generation = ModuleType("transformers.generation")
    generation.__path__ = []
    logits_process = ModuleType("transformers.generation.logits_process")

    class LogitsProcessorList(list):
        def __call__(self, input_ids, scores):
            for processor in self:
                scores = processor(input_ids, scores)
            return scores

    class _IdentityProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, _input_ids, scores):
            return scores

    logits_process.LogitsProcessorList = LogitsProcessorList
    logits_process.RepetitionPenaltyLogitsProcessor = _IdentityProcessor
    logits_process.TemperatureLogitsWarper = _IdentityProcessor
    logits_process.TopKLogitsWarper = _IdentityProcessor
    logits_process.TopPLogitsWarper = _IdentityProcessor

    sys.modules["transformers"] = package
    sys.modules["transformers.generation"] = generation
    sys.modules["transformers.generation.logits_process"] = logits_process


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
_install_transformers_stub()
_install_huggingface_stub()
_install_safetensors_stub()
