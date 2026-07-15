from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import performance_runtime


class _FakeServer(SimpleNamespace):
    def __init__(self):
        super().__init__(
            logger=logging.getLogger("performance-runtime-test"),
            VOICE_CACHE_SIZE=8,
            MODEL_IDLE_UNLOAD_SECONDS=1800,
            DEVICE="cuda",
            model=None,
            model_lock=threading.Lock(),
        )
        self.generate_chunk_locked = lambda **_: ("original", 0, False)
        self.unload_model_locked = lambda _reason: None
        self.runtime_status = lambda include_sensitive=True: {
            "include_sensitive": include_sensitive
        }


class _FakeMultilingual(SimpleNamespace):
    def __init__(self):
        super().__init__(_profile_engines={})

    @staticmethod
    def profile_from_marker(_voice_path):
        return None


def test_disabled_runtime_preserves_generation(monkeypatch):
    monkeypatch.setenv("TURBO_PERFORMANCE_RUNTIME", "0")
    server = _FakeServer()
    original = server.generate_chunk_locked

    runtime = performance_runtime.install_turbo_performance_runtime(
        server,
        _FakeMultilingual(),
    )

    assert runtime.enabled is False
    assert server.generate_chunk_locked is original
    assert server.runtime_status()["performance"]["enabled"] is False


def test_version_mismatch_disables_source_sensitive_rewrites(monkeypatch):
    monkeypatch.setenv("TURBO_PERFORMANCE_RUNTIME", "1")
    monkeypatch.setenv("TURBO_EXPECTED_CHATTERBOX_VERSION", "0.1.6")
    monkeypatch.setattr(
        performance_runtime,
        "_distribution_version",
        lambda name: "9.9.9" if name == "chatterbox-tts" else "test",
    )

    runtime = performance_runtime.TurboPerformanceRuntime(
        _FakeServer(),
        _FakeMultilingual(),
    )

    assert runtime.package_compatible is False


def test_resolve_t3_core_uses_bound_method_owner():
    class Core:
        tfmr = object()
        speech_emb = object()
        speech_head = object()
        hp = object()

        def inference_turbo(self):
            return None

    core = Core()

    class Proxy:
        inference_turbo = core.inference_turbo

    assert performance_runtime._resolve_t3_core(Proxy()) is core


def test_generate_wrapper_does_not_add_cuda_synchronize(monkeypatch):
    monkeypatch.setenv("TURBO_PERFORMANCE_RUNTIME", "1")
    server = _FakeServer()
    multilingual = _FakeMultilingual()
    engine = SimpleNamespace(
        t3=SimpleNamespace(),
        conds=object(),
        sr=24000,
        generate=lambda *_args, **_kwargs: "cpu-waveform",
    )

    server.ensure_model_loaded_locked = lambda: engine
    server.set_seed = lambda _seed: None
    server.touch_model_usage = lambda: None
    server.tensor_to_float_array = lambda waveform: waveform

    runtime = performance_runtime.TurboPerformanceRuntime(server, multilingual)
    runtime.prepare_engine = lambda candidate, _profile: candidate

    monkeypatch.setattr(
        performance_runtime.torch.cuda,
        "synchronize",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected synchronize")),
        raising=False,
    )
    monkeypatch.setattr(
        performance_runtime.torch,
        "inference_mode",
        lambda: nullcontext(),
        raising=False,
    )

    waveform, sample_rate, cache_hit = runtime.generate_chunk_locked(
        text="Hello.",
        voice_path=None,
        temperature=0.8,
        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,
        norm_loudness=True,
        seed=7,
    )

    assert waveform == "cpu-waveform"
    assert sample_rate == 24000
    assert cache_hit is False


def test_source_guards_and_experimental_backends_remain_off():
    root = Path(performance_runtime.__file__).parent
    runtime_source = (root / "performance_runtime.py").read_text(encoding="utf-8")
    patch_source = (root / "performance_patches.py").read_text(encoding="utf-8")
    support_source = (root / "performance_support.py").read_text(encoding="utf-8")

    assert "source_matches" in patch_source
    assert "def source_matches" in support_source
    assert "TURBO_EXPECTED_CHATTERBOX_VERSION" in runtime_source
    assert '"sdpa": False' in runtime_source
    assert '"torch_compile": False' in runtime_source
    assert '"microbatching": False' in runtime_source
