from types import SimpleNamespace

import pytest

from watermark_control import NoOpPerthImplicitWatermarker, configure_watermarking


def test_noop_watermarker_returns_same_waveform() -> None:
    waveform = object()

    result = NoOpPerthImplicitWatermarker().apply_watermark(waveform, sample_rate=24_000)

    assert result is waveform


def test_disabled_mode_replaces_constructor_before_engine_creation() -> None:
    class ExplodingWatermarker:
        def __init__(self) -> None:
            raise AssertionError("Original Perth constructor must not run")

    perth = SimpleNamespace(PerthImplicitWatermarker=ExplodingWatermarker)
    tts_turbo = SimpleNamespace(perth=perth)

    mode = configure_watermarking(tts_turbo, enabled=False)
    watermarker = perth.PerthImplicitWatermarker()

    assert mode == "disabled"
    assert isinstance(watermarker, NoOpPerthImplicitWatermarker)


def test_disabled_mode_tolerates_lightweight_test_stub() -> None:
    tts_turbo = SimpleNamespace()

    assert configure_watermarking(tts_turbo, enabled=False) == "disabled-unverified"


def test_disabled_mode_fails_for_changed_installed_package() -> None:
    tts_turbo = SimpleNamespace(__file__="/site-packages/chatterbox/tts_turbo.py")

    with pytest.raises(RuntimeError, match="no supported Perth watermarker constructor"):
        configure_watermarking(tts_turbo, enabled=False)


def test_enabled_mode_preserves_upstream_constructor() -> None:
    original = object()
    perth = SimpleNamespace(PerthImplicitWatermarker=original)
    tts_turbo = SimpleNamespace(perth=perth)

    assert configure_watermarking(tts_turbo, enabled=True) == "enabled"
    assert perth.PerthImplicitWatermarker is original
