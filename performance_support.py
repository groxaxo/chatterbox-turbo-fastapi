from __future__ import annotations

import importlib.metadata
import inspect
import os
import re
from typing import Any, Optional

import torch


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution_requirements(name: str) -> list[str]:
    try:
        return list(importlib.metadata.requires(name) or [])
    except importlib.metadata.PackageNotFoundError:
        return []


def support_matrix() -> dict[str, Any]:
    requirements = distribution_requirements("chatterbox-tts")
    exact_pins: dict[str, str] = {}
    for requirement in requirements:
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([^;,\s]+)", requirement.strip())
        if match:
            exact_pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    try:
        requires_python = importlib.metadata.metadata("chatterbox-tts").get("Requires-Python")
    except importlib.metadata.PackageNotFoundError:
        requires_python = None
    actual = {
        "torch": torch.__version__.split("+", 1)[0],
        "torchaudio": distribution_version("torchaudio"),
        "transformers": distribution_version("transformers"),
    }
    mismatches = [
        {"package": package, "expected": expected, "actual": actual.get(package)}
        for package, expected in sorted(exact_pins.items())
        if package in actual and actual.get(package) != expected
    ]
    return {
        "requires_python": requires_python,
        "declared_requirements": requirements,
        "exact_pins": exact_pins,
        "actual": actual,
        "mismatches": mismatches,
        "inside_declared_exact_pins": not mismatches,
    }


def tensor_version(value: torch.Tensor) -> Optional[int]:
    try:
        return int(value._version)
    except RuntimeError:
        return None


def tensor_fingerprint(value: torch.Tensor) -> tuple[Any, ...]:
    try:
        data_ptr = value.data_ptr()
    except RuntimeError:
        data_ptr = None
    return (
        id(value), data_ptr, tuple(value.shape), tuple(value.stride()),
        str(value.dtype), str(value.device), tensor_version(value), bool(value.requires_grad),
    )


def module_fingerprint(module: Any) -> tuple[Any, ...]:
    parameters = tuple(tensor_fingerprint(parameter) for parameter in module.parameters()) \
        if module is not None and hasattr(module, "parameters") else ()
    return id(module), bool(getattr(module, "training", False)), parameters


def conditioning_fingerprint(core: Any, conditional: Any) -> tuple[Any, ...]:
    values: list[tuple[str, Any]] = []
    for name, value in sorted(vars(conditional).items()):
        if torch.is_tensor(value):
            values.append((name, tensor_fingerprint(value)))
        elif value is None or isinstance(value, (str, int, float, bool)):
            values.append((name, value))
        else:
            values.append((name, (type(value).__qualname__, id(value))))
    return (
        id(conditional), tuple(values),
        module_fingerprint(getattr(core, "cond_enc", None)),
        module_fingerprint(getattr(core, "speech_emb", None)),
    )


def source_matches(function: Any, required_markers: tuple[str, ...]) -> bool:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(marker in source for marker in required_markers)


def resolve_t3_core(t3: Any) -> Any:
    """Resolve the native T3 owner behind optional PEFT proxy objects."""
    for method_name in ("inference_turbo", "prepare_input_embeds", "prepare_conditioning"):
        owner = getattr(getattr(t3, method_name, None), "__self__", None)
        if owner is not None and all(hasattr(owner, attr) for attr in ("tfmr", "speech_emb", "speech_head", "hp")):
            return owner
    get_base_model = getattr(t3, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
        if candidate is not t3:
            return resolve_t3_core(candidate)
    if all(hasattr(t3, attr) for attr in ("tfmr", "speech_emb", "speech_head", "hp")):
        return t3
    raise TypeError(f"Could not resolve the native T3 core from {type(t3)!r}")
