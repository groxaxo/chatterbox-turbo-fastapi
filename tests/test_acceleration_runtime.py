from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import acceleration_runtime
from benchmarks.benchmark_acceleration import _capture_env


class _FakeServer(SimpleNamespace):
    def __init__(self):
        super().__init__(logger=logging.getLogger("acceleration-runtime-test"))
        self.generate_chunk_locked = lambda **_: ("local", 24_000, False)
        self.warm_worker_model_if_needed = lambda: setattr(self, "local_warmed", True)
        self.runtime_status = lambda include_sensitive=True: {
            "include_sensitive": include_sensitive
        }
        self.local_warmed = False


class _FakeMultilingual(SimpleNamespace):
    def __init__(self):
        super().__init__(_profile_engines={})

    @staticmethod
    def profile_from_marker(voice_path):
        return "lucia-ar" if voice_path == "spanish-marker" else None


class _FakePerformance:
    @staticmethod
    def prepare_engine(engine, _profile_id):
        return engine


def _install(monkeypatch, mode: str = "torch"):
    monkeypatch.setenv("TURBO_ACCELERATOR", mode)
    server = _FakeServer()
    runtime = acceleration_runtime.install_acceleration_runtime(
        server,
        _FakeMultilingual(),
        _FakePerformance(),
    )
    return server, runtime


def test_torch_mode_preserves_generation_and_reports_default(monkeypatch):
    server = _FakeServer()
    original = server.generate_chunk_locked
    monkeypatch.setenv("TURBO_ACCELERATOR", "torch")

    runtime = acceleration_runtime.install_acceleration_runtime(
        server,
        _FakeMultilingual(),
        _FakePerformance(),
    )

    assert runtime.mode == "torch"
    assert server.generate_chunk_locked is original
    status = server.runtime_status(include_sensitive=False)["acceleration"]
    assert status["default_path_unchanged"] is True
    assert status["classification"].startswith("E0")


def test_invalid_accelerator_is_rejected(monkeypatch):
    monkeypatch.setenv("TURBO_ACCELERATOR", "magic")
    with pytest.raises(ValueError, match="TURBO_ACCELERATOR"):
        acceleration_runtime.AccelerationRuntime(
            _FakeServer(),
            _FakeMultilingual(),
            _FakePerformance(),
        )


def test_tensorrt_without_cuda_keeps_pytorch_fallback(monkeypatch):
    monkeypatch.setenv("TURBO_ACCELERATOR", "tensorrt")
    monkeypatch.setattr(acceleration_runtime.torch.cuda, "is_available", lambda: False)
    server = _FakeServer()
    performance = _FakePerformance()
    runtime = acceleration_runtime.install_acceleration_runtime(
        server,
        _FakeMultilingual(),
        performance,
    )
    engine = SimpleNamespace()

    assert performance.prepare_engine(engine, "english") is engine
    status = server.runtime_status()["acceleration"]["tensorrt"]["engines"]["english"]
    assert status["prepared"] is True
    assert status["fallback_count"] == 1
    assert "requires CUDA" in status["last_error"]


def test_vllm_routes_only_allowlisted_profiles(monkeypatch):
    server, runtime = _install(monkeypatch, "vllm")
    monkeypatch.setattr(
        runtime,
        "_generate_vllm",
        lambda **_: ("sidecar", 24_000, False),
    )

    english = server.generate_chunk_locked(
        text="Hello.", voice_path=None, temperature=0.8, top_p=0.95,
        top_k=1000, repetition_penalty=1.2, norm_loudness=True, seed=7,
    )
    spanish = server.generate_chunk_locked(
        text="Hola.", voice_path="spanish-marker", temperature=0.8, top_p=0.95,
        top_k=1000, repetition_penalty=1.2, norm_loudness=True, seed=7,
    )

    assert english[0] == "sidecar"
    assert spanish[0] == "local"
    assert server.runtime_status()["acceleration"]["vllm"]["fallbacks"] == 1


def test_vllm_failure_falls_back_and_redacts_url(monkeypatch):
    monkeypatch.setenv("TURBO_VLLM_BASE_URL", "http://127.0.0.1:9000/private/path")
    server, runtime = _install(monkeypatch, "vllm")

    def fail(**_kwargs):
        raise OSError("sidecar down")

    monkeypatch.setattr(runtime, "_generate_vllm", fail)
    result = server.generate_chunk_locked(
        text="Hello.", voice_path=None, temperature=0.8, top_p=0.95,
        top_k=1000, repetition_penalty=1.2, norm_loudness=True, seed=7,
    )

    assert result[0] == "local"
    status = server.runtime_status(include_sensitive=False)["acceleration"]["vllm"]
    assert status["base_url"] == "http://127.0.0.1:9000"
    assert status["failures"] == 1
    assert status["fallbacks"] == 1


def test_acceleration_benchmark_pins_candidate_environment():
    env = _capture_env("tensorrt")
    assert env["TURBO_PERFORMANCE_RUNTIME"] == "1"
    assert env["TURBO_ACCELERATOR"] == "tensorrt"
    assert env["TURBO_ACCELERATOR_FAIL_CLOSED"] == "1"
    assert env["TURBO_PARITY_DETERMINISTIC"] == "1"
