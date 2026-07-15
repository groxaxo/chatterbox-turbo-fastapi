from __future__ import annotations

import io
import json
import os
import types
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from performance_support import env_bool, source_matches


_VALID_ACCELERATORS = {"torch", "tensorrt", "vllm"}


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return "configured"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


@dataclass
class EngineAccelerationState:
    profile_id: str
    prepared: bool = False
    backend: str = "torch"
    compile_attempted: bool = False
    compile_succeeded: bool = False
    fallback_count: int = 0
    call_count: int = 0
    last_error: Optional[str] = None
    notes: list[str] = field(default_factory=list)


class _HiFTTensorRTModule(torch.nn.Module):
    """Small export boundary around the deterministic mel-to-wave vocoder stage."""

    def __init__(self, mel2wav: torch.nn.Module):
        super().__init__()
        self.mel2wav = mel2wav

    def forward(
        self,
        speech_feat: torch.Tensor,
        cache_source: torch.Tensor,
    ) -> Any:
        return self.mel2wav.inference(
            speech_feat=speech_feat,
            cache_source=cache_source,
        )


class AccelerationRuntime:
    """Install opt-in TensorRT or isolated vLLM acceleration around the proven runtime."""

    def __init__(self, server_module: Any, multilingual_runtime: Any, performance_runtime: Any):
        self.server = server_module
        self.multilingual = multilingual_runtime
        self.performance = performance_runtime
        self.mode = os.getenv("TURBO_ACCELERATOR", "torch").strip().lower()
        if self.mode not in _VALID_ACCELERATORS:
            raise ValueError(
                f"TURBO_ACCELERATOR must be one of {sorted(_VALID_ACCELERATORS)}, got {self.mode!r}."
            )

        self.fail_closed = env_bool("TURBO_ACCELERATOR_FAIL_CLOSED", False)
        self.tensorrt_backend = os.getenv("TURBO_TENSORRT_BACKEND", "torch_tensorrt").strip()
        self.tensorrt_dynamic = env_bool("TURBO_TENSORRT_DYNAMIC", False)
        self.tensorrt_fullgraph = env_bool("TURBO_TENSORRT_FULLGRAPH", False)
        self.tensorrt_require_fp32 = env_bool("TURBO_TENSORRT_REQUIRE_FP32", True)

        self.vllm_base_url = os.getenv(
            "TURBO_VLLM_BASE_URL", "http://127.0.0.1:8000"
        ).rstrip("/")
        self.vllm_health_path = os.getenv("TURBO_VLLM_HEALTH_PATH", "/health")
        self.vllm_speech_path = os.getenv(
            "TURBO_VLLM_SPEECH_PATH", "/v1/audio/speech"
        )
        self.vllm_model = os.getenv("TURBO_VLLM_MODEL", "tts-1")
        self.vllm_timeout_seconds = float(os.getenv("TURBO_VLLM_TIMEOUT_SECONDS", "180"))
        self.vllm_profiles = set(_env_csv("TURBO_VLLM_PROFILES", "english"))
        self.vllm_exclusive = env_bool("TURBO_VLLM_EXCLUSIVE", False)
        self.vllm_voice_map = self._parse_voice_map(
            os.getenv("TURBO_VLLM_VOICE_MAP", "english=alloy")
        )

        self._engine_states: dict[int, EngineAccelerationState] = {}
        self._original_prepare_engine: Optional[Callable[..., Any]] = None
        self._original_generate_chunk_locked: Optional[Callable[..., Any]] = None
        self._original_warm_worker_model_if_needed: Optional[Callable[..., Any]] = None
        self._original_runtime_status: Optional[Callable[..., Any]] = None
        self._vllm_calls = 0
        self._vllm_fallbacks = 0
        self._vllm_failures = 0
        self._vllm_health_checks = 0
        self._vllm_last_error: Optional[str] = None

    @staticmethod
    def _parse_voice_map(raw: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "TURBO_VLLM_VOICE_MAP entries must use profile=voice syntax."
                )
            profile, voice = (part.strip() for part in item.split("=", 1))
            if not profile or not voice:
                raise ValueError(
                    "TURBO_VLLM_VOICE_MAP entries must contain non-empty profile and voice values."
                )
            result[profile] = voice
        return result

    def _state_for(self, engine: Any, profile_id: str) -> EngineAccelerationState:
        state = self._engine_states.get(id(engine))
        if state is None:
            state = EngineAccelerationState(profile_id=profile_id, backend=self.mode)
            self._engine_states[id(engine)] = state
        return state

    def _choose_tensorrt_backend(self) -> str:
        try:
            import torch_tensorrt  # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised on GPU deployment
            raise RuntimeError(
                "TensorRT mode requires the optional torch-tensorrt package. "
                "Run ./install_tensorrt.sh in the worker environment."
            ) from exc

        list_backends = getattr(getattr(torch, "_dynamo", None), "list_backends", None)
        available = set(list_backends()) if callable(list_backends) else set()
        requested = self.tensorrt_backend
        aliases = (requested, "tensorrt", "torch_tensorrt")
        for candidate in aliases:
            if not available or candidate in available:
                return candidate
        raise RuntimeError(
            f"No Torch-TensorRT Dynamo backend is registered; available={sorted(available)}."
        )

    def prepare_engine(self, engine: Any, profile_id: str) -> Any:
        if self.mode != "tensorrt":
            return engine

        state = self._state_for(engine, profile_id)
        if state.prepared:
            return engine
        state.prepared = True

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("TensorRT acceleration requires CUDA.")
            s3gen = getattr(engine, "s3gen", None)
            mel2wav = getattr(s3gen, "mel2wav", None)
            original_hift = getattr(s3gen, "hift_inference", None)
            if s3gen is None or mel2wav is None or not callable(original_hift):
                raise RuntimeError(
                    "Installed S3Gen does not expose mel2wav + hift_inference; keeping PyTorch."
                )
            if not source_matches(original_hift, ("mel2wav.inference", "cache_source")):
                raise RuntimeError(
                    "Installed hift_inference source does not match the reviewed export boundary."
                )
            if self.tensorrt_require_fp32:
                parameters = list(mel2wav.parameters())
                if parameters and any(parameter.dtype != torch.float32 for parameter in parameters):
                    raise RuntimeError(
                        "TURBO_TENSORRT_REQUIRE_FP32=1 but the HiFT module is not entirely FP32."
                    )

            backend = self._choose_tensorrt_backend()
            wrapper = _HiFTTensorRTModule(mel2wav).eval()
            compiled = torch.compile(
                wrapper,
                backend=backend,
                dynamic=self.tensorrt_dynamic,
                fullgraph=self.tensorrt_fullgraph,
            )
            state.backend = backend
            state.compile_attempted = True

            def accelerated_hift(
                module: Any,
                speech_feat: torch.Tensor,
                cache_source: Optional[torch.Tensor] = None,
            ) -> Any:
                state.call_count += 1
                if cache_source is None:
                    cache_source = torch.zeros(
                        1,
                        1,
                        0,
                        device=speech_feat.device,
                        dtype=speech_feat.dtype,
                    )
                try:
                    result = compiled(speech_feat, cache_source)
                    state.compile_succeeded = True
                    return result
                except Exception as exc:
                    state.last_error = f"{type(exc).__name__}: {exc}"
                    state.fallback_count += 1
                    module.hift_inference = original_hift
                    if self.fail_closed:
                        raise RuntimeError(
                            f"TensorRT HiFT execution failed for {profile_id}."
                        ) from exc
                    self.server.logger.exception(
                        "TensorRT HiFT failed for profile '%s'; restored PyTorch fallback.",
                        profile_id,
                    )
                    return original_hift(speech_feat, cache_source)

            s3gen._turbo_accel_original_hift_inference = original_hift
            s3gen.hift_inference = types.MethodType(accelerated_hift, s3gen)
            state.notes.append("FP32 HiFT/vocoder boundary compiled lazily during warmup.")
            self.server.logger.info(
                "Prepared TensorRT HiFT candidate for profile '%s' (backend=%s).",
                profile_id,
                backend,
            )
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
            state.fallback_count += 1
            if self.fail_closed:
                raise
            self.server.logger.warning(
                "TensorRT preparation unavailable for profile '%s': %s. Using PyTorch.",
                profile_id,
                exc,
            )
        return engine

    def _profile_for_voice(self, voice_path: Any) -> str:
        return self.multilingual.profile_from_marker(voice_path) or "english"

    def _http_request(
        self,
        url: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> bytes:
        data = None
        headers = {"Accept": "application/json, audio/wav"}
        method = "GET"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(
            request,
            timeout=self.vllm_timeout_seconds if timeout is None else timeout,
        ) as response:
            return response.read()

    def _check_vllm_health(self) -> None:
        self._vllm_health_checks += 1
        url = f"{self.vllm_base_url}{self.vllm_health_path}"
        body = self._http_request(url, timeout=min(self.vllm_timeout_seconds, 10.0))
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and payload.get("status") not in (None, "healthy", "ok"):
                raise RuntimeError(f"vLLM sidecar is not healthy: {payload!r}")

    def _generate_vllm(
        self,
        *,
        profile_id: str,
        text: str,
        voice_path: Any,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
    ) -> tuple[Any, int, bool]:
        payload: dict[str, Any] = {
            "model": self.vllm_model,
            "input": text.strip(),
            "voice": self.vllm_voice_map.get(
                profile_id,
                "alloy" if profile_id == "english" else profile_id,
            ),
            "response_format": "wav",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "seed": int(seed),
        }
        if voice_path is not None and profile_id == "english":
            payload["audio_prompt_path"] = str(voice_path)
        body = self._http_request(
            f"{self.vllm_base_url}{self.vllm_speech_path}",
            payload=payload,
        )
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1, dtype=np.float32)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise RuntimeError(
                f"vLLM sidecar returned invalid waveform shape={array.shape}."
            )
        return np.ascontiguousarray(array), int(sample_rate), False

    def generate_chunk_locked(self, **kwargs: Any) -> tuple[Any, int, bool]:
        assert self._original_generate_chunk_locked is not None
        voice_path = kwargs.get("voice_path")
        profile_id = self._profile_for_voice(voice_path)
        if profile_id not in self.vllm_profiles:
            if self.vllm_exclusive:
                raise RuntimeError(
                    f"Profile {profile_id!r} is not enabled in TURBO_VLLM_PROFILES."
                )
            self._vllm_fallbacks += 1
            return self._original_generate_chunk_locked(**kwargs)

        try:
            self._vllm_calls += 1
            return self._generate_vllm(
                profile_id=profile_id,
                text=str(kwargs["text"]),
                voice_path=voice_path,
                temperature=float(kwargs["temperature"]),
                top_p=float(kwargs["top_p"]),
                repetition_penalty=float(kwargs["repetition_penalty"]),
                seed=int(kwargs["seed"]),
            )
        except Exception as exc:
            self._vllm_failures += 1
            self._vllm_last_error = f"{type(exc).__name__}: {exc}"
            if self.fail_closed or self.vllm_exclusive:
                raise RuntimeError(
                    f"vLLM sidecar generation failed for profile {profile_id!r}."
                ) from exc
            self._vllm_fallbacks += 1
            self.server.logger.exception(
                "vLLM sidecar failed for profile '%s'; using resident PyTorch worker.",
                profile_id,
            )
            return self._original_generate_chunk_locked(**kwargs)

    def warm_worker_model_if_needed(self) -> None:
        assert self._original_warm_worker_model_if_needed is not None
        try:
            self._check_vllm_health()
            self.server.logger.info("vLLM sidecar health check passed.")
        except Exception as exc:
            self._vllm_last_error = f"{type(exc).__name__}: {exc}"
            if self.fail_closed or self.vllm_exclusive:
                raise RuntimeError("vLLM sidecar health check failed during bootstrap.") from exc
            self.server.logger.warning(
                "vLLM sidecar health check failed during bootstrap: %s. Local fallback remains warm.",
                exc,
            )
        if not self.vllm_exclusive:
            self._original_warm_worker_model_if_needed()

    def runtime_status(self, include_sensitive: bool = True) -> dict[str, Any]:
        assert self._original_runtime_status is not None
        status = self._original_runtime_status(include_sensitive=include_sensitive)
        engine_states = {
            state.profile_id: {
                "prepared": state.prepared,
                "backend": state.backend,
                "compile_attempted": state.compile_attempted,
                "compile_succeeded": state.compile_succeeded,
                "call_count": state.call_count,
                "fallback_count": state.fallback_count,
                "last_error": state.last_error,
                "notes": list(state.notes),
            }
            for state in self._engine_states.values()
        }
        classification = {
            "torch": "E0 exact-output PyTorch",
            "tensorrt": "E1 exact-parity candidate: FP32 HiFT only",
            "vllm": "E2 quality-gated sidecar: sampling is not byte-exact",
        }[self.mode]
        status["acceleration"] = {
            "mode": self.mode,
            "classification": classification,
            "fail_closed": self.fail_closed,
            "default_path_unchanged": self.mode == "torch",
            "tensorrt": {
                "backend": self.tensorrt_backend,
                "dynamic": self.tensorrt_dynamic,
                "fullgraph": self.tensorrt_fullgraph,
                "require_fp32": self.tensorrt_require_fp32,
                "engines": engine_states,
            },
            "vllm": {
                "base_url": self.vllm_base_url if include_sensitive else _redact_url(self.vllm_base_url),
                "profiles": sorted(self.vllm_profiles),
                "exclusive": self.vllm_exclusive,
                "calls": self._vllm_calls,
                "fallbacks": self._vllm_fallbacks,
                "failures": self._vllm_failures,
                "health_checks": self._vllm_health_checks,
                "last_error": self._vllm_last_error,
                "unsupported_exact_controls": ["top_k", "cross-runtime RNG parity"],
            },
        }
        return status

    def install(self) -> "AccelerationRuntime":
        server = self.server
        if getattr(server, "_turbo_acceleration_runtime_installed", False):
            return server._turbo_acceleration_runtime

        self._original_runtime_status = server.runtime_status
        server.runtime_status = self.runtime_status

        if self.mode == "tensorrt":
            self._original_prepare_engine = self.performance.prepare_engine

            def prepare_with_acceleration(engine: Any, profile_id: str) -> Any:
                assert self._original_prepare_engine is not None
                prepared = self._original_prepare_engine(engine, profile_id)
                return self.prepare_engine(prepared, profile_id)

            self.performance.prepare_engine = prepare_with_acceleration
        elif self.mode == "vllm":
            self._original_generate_chunk_locked = server.generate_chunk_locked
            self._original_warm_worker_model_if_needed = server.warm_worker_model_if_needed
            server.generate_chunk_locked = self.generate_chunk_locked
            server.warm_worker_model_if_needed = self.warm_worker_model_if_needed

        server._turbo_acceleration_runtime_installed = True
        server._turbo_acceleration_runtime = self
        server.logger.info(
            "Turbo acceleration runtime installed: mode=%s fail_closed=%s",
            self.mode,
            self.fail_closed,
        )
        return self


def install_acceleration_runtime(
    server_module: Any,
    multilingual_runtime: Any,
    performance_runtime: Any,
) -> AccelerationRuntime:
    return AccelerationRuntime(
        server_module,
        multilingual_runtime,
        performance_runtime,
    ).install()
