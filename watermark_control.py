from __future__ import annotations

from types import ModuleType
from typing import Any


class NoOpPerthImplicitWatermarker:
    """Drop-in Perth adapter that leaves synthesized audio untouched."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def apply_watermark(self, wav: Any, sample_rate: int) -> Any:
        return wav


def configure_watermarking(tts_turbo: ModuleType | Any, *, enabled: bool) -> str:
    """Configure Perth before any ChatterboxTurboTTS instance is constructed."""

    if enabled:
        return "enabled"

    perth = getattr(tts_turbo, "perth", None)
    if perth is None or not hasattr(perth, "PerthImplicitWatermarker"):
        if getattr(tts_turbo, "__file__", None):
            raise RuntimeError("Installed chatterbox.tts_turbo has no supported Perth watermarker constructor.")
        return "disabled-unverified"

    perth.PerthImplicitWatermarker = NoOpPerthImplicitWatermarker
    return "disabled"
