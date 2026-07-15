from __future__ import annotations

import os
import platform
from typing import Any, Callable, Optional

import torch
from multilingual_runtime import normalize_spanish_text

from performance_patches import EngineOptimizer
from performance_support import (
    distribution_version,
    env_bool,
    resolve_t3_core,
    support_matrix,
)

# Re-exported for benchmark/test instrumentation.
_resolve_t3_core = resolve_t3_core
_distribution_version = distribution_version


class TurboPerformanceRuntime:
    """Install exact-output-oriented Turbo optimizations after multilingual routing."""

    def __init__(self, server_module: Any, multilingual_runtime: Any):
        self.server = server_module
        self.multilingual = multilingual_runtime
        self.enabled = env_bool("TURBO_PERFORMANCE_RUNTIME", True)
        self.cache_encoded_conditioning = env_bool("TURBO_CACHE_ENCODED_CONDITIONING", True)
        self.preallocate_token_ids = env_bool("TURBO_PREALLOCATE_TOKEN_IDS", True)
        self.cache_silence_tensor = env_bool("TURBO_CACHE_SILENCE_TENSOR", True)
        self.disable_progress = env_bool("TURBO_DISABLE_PROGRESS", True)
        self.strict_logit_checks = env_bool("TURBO_STRICT_LOGIT_CHECKS", True)
        self.rewrite_package_generate = env_bool("TURBO_REWRITE_PACKAGE_GENERATE", True)
        self.fail_on_incompatible_package = env_bool("TURBO_FAIL_ON_INCOMPATIBLE_PACKAGE", False)
        self.expected_package_version = os.getenv("TURBO_EXPECTED_CHATTERBOX_VERSION", "0.1.6").strip()
        self.conditioning_cache_size = max(1, int(os.getenv(
            "TURBO_ENCODED_CONDITION_CACHE_SIZE",
            str(max(1, getattr(server_module, "VOICE_CACHE_SIZE", 8))),
        )))
        self.chatterbox_version = _distribution_version("chatterbox-tts")
        self.transformers_version = _distribution_version("transformers")
        self.package_compatible = not self.expected_package_version or self.chatterbox_version == self.expected_package_version
        self.optimizer = EngineOptimizer(self)
        self._original_generate_chunk_locked: Optional[Callable[..., Any]] = None
        self._original_unload_model_locked: Optional[Callable[..., Any]] = None
        self._original_runtime_status: Optional[Callable[..., Any]] = None
        if not self.package_compatible:
            message = (
                f"Turbo rewrites expect chatterbox-tts {self.expected_package_version}; "
                f"installed={self.chatterbox_version or 'unknown'}. Falling back to package code."
            )
            if self.fail_on_incompatible_package:
                raise RuntimeError(message)
            self.server.logger.warning(message)

    def prepare_engine(self, engine: Any, profile_id: str) -> Any:
        try:
            return self.optimizer.prepare(engine, profile_id)
        except Exception:
            if self.fail_on_incompatible_package:
                raise
            self.server.logger.exception(
                "Could not install Turbo performance rewrites for profile '%s'; using package code.",
                profile_id,
            )
            return engine

    def generate_chunk_locked(
        self, *, text: str, voice_path: Any, temperature: float, top_p: float,
        top_k: int, repetition_penalty: float, norm_loudness: bool, seed: int,
    ) -> tuple[Any, int, bool]:
        server = self.server
        profile_id = self.multilingual.profile_from_marker(voice_path)
        if profile_id:
            normalized = normalize_spanish_text(text)
            with server.model_lock:
                was_cached = profile_id in self.multilingual._profile_engines
                engine = self.multilingual.ensure_spanish_engine_loaded_locked(profile_id)
                self.multilingual._validate_spanish_tags(normalized, engine)
                self.prepare_engine(engine, profile_id)
                server.set_seed(seed)
                with torch.inference_mode():
                    waveform = engine.generate(
                        normalized, audio_prompt_path=None, temperature=float(temperature),
                        top_p=float(top_p), top_k=int(top_k),
                        repetition_penalty=float(repetition_penalty),
                        norm_loudness=bool(norm_loudness),
                    )
                server.touch_model_usage()
                return server.tensor_to_float_array(waveform), int(engine.sr), was_cached

        with server.model_lock:
            engine = server.ensure_model_loaded_locked()
            self.prepare_engine(engine, "english")
            server.set_seed(seed)
            cache_hit = False
            if voice_path is not None:
                conditionals, cache_hit = server.get_or_prepare_conditionals(
                    voice_path, norm_loudness=norm_loudness
                )
                engine.conds = conditionals
            elif engine.conds is None:
                raise RuntimeError("No default voice conditionals are loaded.")
            with torch.inference_mode():
                waveform = engine.generate(
                    text.strip(), audio_prompt_path=None, temperature=float(temperature),
                    top_p=float(top_p), top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    norm_loudness=bool(norm_loudness),
                )
            # No post-generate synchronize: generate has already copied to CPU.
            server.touch_model_usage()
            return server.tensor_to_float_array(waveform), int(engine.sr), cache_hit

    def unload_model_locked(self, reason: str) -> None:
        assert self._original_unload_model_locked is not None
        self._original_unload_model_locked(reason)

    def engine_status(self, engine: Any) -> dict[str, Any]:
        try:
            core = resolve_t3_core(engine.t3)
        except Exception:
            return {"prepared": False}
        config = getattr(getattr(core, "tfmr", None), "config", None)
        return {
            "prepared": bool(getattr(engine, "_turbo_perf_runtime_prepared", False)),
            "condition_cache": bool(getattr(core, "_turbo_perf_conditioning_cache_installed", False)),
            "preallocated_token_ids": bool(getattr(core, "_turbo_perf_preallocated_inference_installed", False)),
            "package_generate_rewrite": bool(getattr(engine, "_turbo_perf_generate_installed", False)),
            "attention_backend": getattr(config, "_attn_implementation", "eager"),
            "stats": dict(self.optimizer.stats(core)),
        }

    def runtime_status(self, include_sensitive: bool = True) -> dict[str, Any]:
        assert self._original_runtime_status is not None
        status = self._original_runtime_status(include_sensitive=include_sensitive)
        engines: dict[str, Any] = {}
        if self.server.model is not None:
            engines["english"] = self.engine_status(self.server.model)
        for profile_id, engine in self.multilingual._profile_engines.items():
            engines[profile_id] = self.engine_status(engine)
        status["performance"] = {
            "enabled": self.enabled,
            "classification": "E0 valid-path exact",
            "expected_chatterbox_version": self.expected_package_version,
            "package_compatible": self.package_compatible,
            "runtime": {
                "python": platform.python_version(), "torch": torch.__version__,
                "transformers": self.transformers_version,
                "chatterbox_tts": self.chatterbox_version,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
            },
            "support_matrix": support_matrix(),
            "flags": {
                "cache_encoded_conditioning": self.cache_encoded_conditioning,
                "preallocate_token_ids": self.preallocate_token_ids,
                "cache_silence_tensor": self.cache_silence_tensor,
                "disable_progress": self.disable_progress,
                "strict_logit_checks": self.strict_logit_checks,
                "rewrite_package_generate": self.rewrite_package_generate,
                "sdpa": False, "torch_compile": False, "microbatching": False,
            },
            "model_idle_unload_seconds": self.server.MODEL_IDLE_UNLOAD_SECONDS,
            "engines": engines,
        }
        return status

    def install(self) -> "TurboPerformanceRuntime":
        server = self.server
        if getattr(server, "_turbo_performance_runtime_installed", False):
            return server._turbo_performance_runtime
        self._original_generate_chunk_locked = server.generate_chunk_locked
        self._original_unload_model_locked = server.unload_model_locked
        self._original_runtime_status = server.runtime_status
        if self.enabled:
            server.generate_chunk_locked = self.generate_chunk_locked
            server.unload_model_locked = self.unload_model_locked
        server.runtime_status = self.runtime_status
        server._turbo_performance_runtime_installed = True
        server._turbo_performance_runtime = self
        server.logger.info(
            "Turbo performance runtime: enabled=%s package=%s compatible=%s",
            self.enabled, self.chatterbox_version, self.package_compatible,
        )
        return self


def install_turbo_performance_runtime(server_module: Any, multilingual_runtime: Any) -> TurboPerformanceRuntime:
    return TurboPerformanceRuntime(server_module, multilingual_runtime).install()
