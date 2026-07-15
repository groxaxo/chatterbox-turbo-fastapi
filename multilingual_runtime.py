from __future__ import annotations

import gc
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


PROFILE_MARKER_ROOT = Path("/__chatterbox_profile__")
_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9_-]*\]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRANSLATION_TABLE = str.maketrans(
    {
        "\u00a0": " ",
        "\u202f": " ",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)


@dataclass(frozen=True)
class SpanishProfile:
    profile_id: str
    language: str
    name: str
    mode: str
    checkpoint_relpath: Optional[str] = None
    adapter_relpath: Optional[str] = None
    conditioning_profile: Optional[str] = None


SPANISH_PROFILES: dict[str, SpanishProfile] = {
    "lucia-ar": SpanishProfile(
        profile_id="lucia-ar",
        language="es-AR",
        name="Lucía — Argentina",
        mode="merged-checkpoint",
        checkpoint_relpath="lucia-ar/t3_turbo_finetuned_merged.safetensors",
    ),
    "lucia-latam": SpanishProfile(
        profile_id="lucia-latam",
        language="es-419",
        name="Lucía — balanced Latin America",
        mode="lora-adapter",
        adapter_relpath="lucia-latam/adapter",
    ),
    "lucia-cl-pilot": SpanishProfile(
        profile_id="lucia-cl-pilot",
        language="es-CL",
        name="Lucía — Chile pilot",
        mode="lora-adapter",
        adapter_relpath="lucia-cl-pilot/adapter",
        conditioning_profile="lucia-latam",
    ),
    "lucia-co-pilot": SpanishProfile(
        profile_id="lucia-co-pilot",
        language="es-CO",
        name="Lucía — Colombia pilot",
        mode="lora-adapter",
        adapter_relpath="lucia-co-pilot/adapter",
        conditioning_profile="lucia-latam",
    ),
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_spanish_text(text: str) -> str:
    """Conservative Spanish normalization copied from the fine-tuning contract.

    It preserves accents, ñ, ü, inverted punctuation and valid Turbo tags. Number,
    date and abbreviation expansion is deliberately not attempted because it can
    make the text disagree with the speech target learned by the adapters.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    value = unicodedata.normalize("NFC", text)
    value = _ZERO_WIDTH_RE.sub("", value)
    value = _CONTROL_RE.sub(" ", value)
    value = value.translate(_TRANSLATION_TABLE)
    value = " ".join(value.split())
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([¿¡])\s+", r"\1", value).strip()

    if not value:
        raise ValueError("Text is empty after Spanish normalization.")
    if not re.search(r"[.!?][\]\)\"']*$", value):
        value += "."
    return value


class MultilingualRuntime:
    def __init__(self, server_module: Any):
        self.server = server_module
        self.enabled = _env_bool("SPANISH_ENABLED", True)
        self.base_repo = os.getenv("BASE_MODEL_REPO", "ResembleAI/chatterbox-turbo").strip()
        self.base_revision = os.getenv("BASE_MODEL_REVISION", "main").strip() or "main"
        self.base_model_dir_override = os.getenv("BASE_MODEL_DIR", "").strip()
        self.spanish_repo = os.getenv("SPANISH_MODEL_REPO", "groxaxo/chaturbo-espanol").strip()
        self.spanish_revision = os.getenv("SPANISH_MODEL_REVISION", "main").strip() or "main"
        self.spanish_model_dir_override = os.getenv("SPANISH_MODEL_DIR", "").strip()
        self.hf_token = os.getenv("HF_TOKEN") or None
        self.profile_cache_size = max(1, int(os.getenv("SPANISH_PROFILE_CACHE_SIZE", "1")))
        self.strict_spanish_tags = _env_bool("STRICT_SPANISH_TAGS", True)
        self.default_profile = self.resolve_profile_id(
            os.getenv("DEFAULT_SPANISH_PROFILE", "lucia-ar")
        ) or "lucia-ar"
        self.preload_profiles = self._parse_preload_profiles(os.getenv("PRELOAD_PROFILES", ""))

        self._base_model_dir: Optional[Path] = None
        self._spanish_model_dir: Optional[Path] = None
        self._profile_engines: "OrderedDict[str, ChatterboxTurboTTS]" = OrderedDict()
        self._download_lock = threading.Lock()

        self._original_normalize_voice_path: Optional[Callable[..., Any]] = None
        self._original_generate_chunk_locked: Optional[Callable[..., Any]] = None
        self._original_unload_model_locked: Optional[Callable[..., Any]] = None
        self._original_available_voices_payload: Optional[Callable[..., Any]] = None
        self._original_runtime_status: Optional[Callable[..., Any]] = None
        self._original_run_generation: Optional[Callable[..., Any]] = None
        self._original_response_with_audio: Optional[Callable[..., Any]] = None
        self._original_warm_worker_model_if_needed: Optional[Callable[..., Any]] = None

    @staticmethod
    def _normalize_profile_key(value: str) -> str:
        return value.strip().casefold().replace("_", "-")

    def resolve_profile_id(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        key = self._normalize_profile_key(value)
        aliases = {
            "lucia": os.getenv("DEFAULT_SPANISH_PROFILE", "lucia-ar"),
            "lucía": os.getenv("DEFAULT_SPANISH_PROFILE", "lucia-ar"),
            "spanish": os.getenv("DEFAULT_SPANISH_PROFILE", "lucia-ar"),
            "es": os.getenv("DEFAULT_SPANISH_PROFILE", "lucia-ar"),
            "es-ar": "lucia-ar",
            "argentina": "lucia-ar",
            "es-419": "lucia-latam",
            "latam": "lucia-latam",
            "latin-america": "lucia-latam",
            "es-cl": "lucia-cl-pilot",
            "lucia-cl": "lucia-cl-pilot",
            "chile": "lucia-cl-pilot",
            "es-co": "lucia-co-pilot",
            "lucia-co": "lucia-co-pilot",
            "colombia": "lucia-co-pilot",
        }
        key = self._normalize_profile_key(aliases.get(key, key))
        return key if key in SPANISH_PROFILES else None

    def _parse_preload_profiles(self, value: str) -> list[str]:
        resolved: list[str] = []
        for item in value.split(","):
            item = item.strip()
            if not item or item.casefold() == "english":
                continue
            profile_id = self.resolve_profile_id(item)
            if profile_id and profile_id not in resolved:
                resolved.append(profile_id)
        return resolved

    @staticmethod
    def marker_for_profile(profile_id: str) -> Path:
        return PROFILE_MARKER_ROOT / profile_id

    def profile_from_marker(self, voice_path: Optional[Path]) -> Optional[str]:
        if voice_path is None:
            return None
        try:
            path = Path(voice_path)
        except TypeError:
            return None
        parts = path.parts
        marker_parts = PROFILE_MARKER_ROOT.parts
        if len(parts) != len(marker_parts) + 1 or parts[: len(marker_parts)] != marker_parts:
            return None
        return self.resolve_profile_id(parts[-1])

    def _snapshot_download(self, *, repo_id: str, revision: str, allow_patterns: list[str]) -> Path:
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": revision,
            "allow_patterns": allow_patterns,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return Path(snapshot_download(**kwargs)).resolve()

    def resolve_base_model_dir(self) -> Path:
        if self._base_model_dir is not None:
            return self._base_model_dir
        with self._download_lock:
            if self._base_model_dir is not None:
                return self._base_model_dir
            if self.base_model_dir_override:
                path = Path(self.base_model_dir_override).expanduser().resolve()
                if not path.is_dir():
                    raise FileNotFoundError(f"BASE_MODEL_DIR does not exist: {path}")
            else:
                path = self._snapshot_download(
                    repo_id=self.base_repo,
                    revision=self.base_revision,
                    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
                )
            required = (
                "t3_turbo_v1.safetensors",
                "s3gen_meanflow.safetensors",
                "ve.safetensors",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            )
            missing = [name for name in required if not (path / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Base model directory is incomplete ({path}); missing: {', '.join(missing)}"
                )
            self._base_model_dir = path
            return path

    def resolve_spanish_model_dir(self) -> Path:
        if self._spanish_model_dir is not None:
            return self._spanish_model_dir
        with self._download_lock:
            if self._spanish_model_dir is not None:
                return self._spanish_model_dir
            if self.spanish_model_dir_override:
                path = Path(self.spanish_model_dir_override).expanduser().resolve()
                if not path.is_dir():
                    raise FileNotFoundError(f"SPANISH_MODEL_DIR does not exist: {path}")
            else:
                allow_patterns: list[str] = []
                for profile_id in SPANISH_PROFILES:
                    allow_patterns.append(f"{profile_id}/**")
                path = self._snapshot_download(
                    repo_id=self.spanish_repo,
                    revision=self.spanish_revision,
                    allow_patterns=allow_patterns,
                )
            self._spanish_model_dir = path
            return path

    def ensure_english_model_loaded_locked(self) -> ChatterboxTurboTTS:
        server = self.server
        if server.model is not None:
            server.touch_model_usage()
            return server.model

        final_device = server.configure_torch()
        server.wait_for_free_vram_if_needed()
        model_dir = self.resolve_base_model_dir()
        server.logger.info("Loading English Chatterbox Turbo from %s on %s...", model_dir, final_device)
        server.model = ChatterboxTurboTTS.from_local(str(model_dir), device=final_device)

        default_path = server.resolve_default_voice_path()
        if default_path is not None:
            server.logger.info("Preparing English default voice conditionals: %s", default_path)
            default_conds, _ = server.get_or_prepare_conditionals(
                default_path,
                norm_loudness=server.DEFAULT_NORM_LOUDNESS,
            )
            server.model.conds = default_conds
        else:
            server.logger.warning(
                "No English default voice reference found. English requests must provide a voice file."
            )

        server.touch_model_usage()
        server.ensure_idle_monitor_started()
        return server.model

    def _profile_paths(self, profile: SpanishProfile) -> tuple[Optional[Path], Optional[Path], Path, Path]:
        root = self.resolve_spanish_model_dir()
        checkpoint = root / profile.checkpoint_relpath if profile.checkpoint_relpath else None
        adapter = root / profile.adapter_relpath if profile.adapter_relpath else None

        conditioning_profile = profile.conditioning_profile or profile.profile_id
        condition_dir = root / conditioning_profile
        reference_override = os.getenv(
            f"{profile.profile_id.upper().replace('-', '_')}_REFERENCE_WAV",
            os.getenv("LUCIA_REFERENCE_WAV", ""),
        ).strip()
        bundle_override = os.getenv(
            f"{profile.profile_id.upper().replace('-', '_')}_CONDITIONING_PT",
            os.getenv("LUCIA_CONDITIONING_PT", ""),
        ).strip()
        reference = (
            Path(reference_override).expanduser().resolve()
            if reference_override
            else condition_dir / "reference.wav"
        )
        bundle = (
            Path(bundle_override).expanduser().resolve()
            if bundle_override
            else condition_dir / "conditioning.pt"
        )
        return checkpoint, adapter, reference, bundle

    def _validate_spanish_tags(self, text: str, engine: ChatterboxTurboTTS) -> None:
        if not self.strict_spanish_tags:
            return
        unknown: list[str] = []
        for tag in sorted(set(_TAG_RE.findall(text))):
            token_ids = engine.tokenizer(tag, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                unknown.append(tag)
                continue
            token = engine.tokenizer.convert_ids_to_tokens(token_ids[0])
            if token != tag:
                unknown.append(tag)
        if unknown:
            raise ValueError(f"Unknown Chatterbox Turbo tag(s): {', '.join(unknown)}")

    def _apply_persona_conditioning(
        self,
        engine: ChatterboxTurboTTS,
        *,
        reference_path: Path,
        bundle_path: Path,
    ) -> None:
        if not reference_path.is_file():
            raise FileNotFoundError(f"Lucía reference WAV not found: {reference_path}")
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Lucía conditioning bundle not found: {bundle_path}")

        engine.prepare_conditionals(
            str(reference_path),
            norm_loudness=self.server.DEFAULT_NORM_LOUDNESS,
        )
        payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
        required = {"speaker_emb", "prompt_tokens"}
        missing = sorted(required.difference(payload))
        if missing:
            raise RuntimeError(
                f"Conditioning bundle {bundle_path} is missing keys: {', '.join(missing)}"
            )

        device = self.server.DEVICE
        engine.conds.t3.speaker_emb = payload["speaker_emb"].float().view(1, -1).to(device)
        engine.conds.t3.cond_prompt_speech_tokens = (
            payload["prompt_tokens"].long().view(1, -1).to(device)
        )
        engine.conds.t3.cond_prompt_speech_emb = None

    def _release_engine(self, engine: ChatterboxTurboTTS, profile_id: str) -> None:
        self.server.logger.info("Evicting Spanish profile '%s' from the worker cache.", profile_id)
        try:
            engine.conds = None
            engine.t3 = None  # type: ignore[assignment]
            engine.s3gen = None  # type: ignore[assignment]
            engine.ve = None  # type: ignore[assignment]
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def ensure_spanish_engine_loaded_locked(self, profile_id: str) -> ChatterboxTurboTTS:
        profile_id = self.resolve_profile_id(profile_id) or ""
        if profile_id not in SPANISH_PROFILES:
            raise ValueError(f"Unknown Spanish profile: {profile_id}")

        cached = self._profile_engines.get(profile_id)
        if cached is not None:
            self._profile_engines.move_to_end(profile_id)
            return cached

        server = self.server
        base_engine = self.ensure_english_model_loaded_locked()
        profile = SPANISH_PROFILES[profile_id]
        checkpoint_path, adapter_path, reference_path, bundle_path = self._profile_paths(profile)

        server.wait_for_free_vram_if_needed()
        server.logger.info(
            "Loading Spanish profile '%s' (%s, %s).",
            profile_id,
            profile.language,
            profile.mode,
        )

        scratch_engine = ChatterboxTurboTTS.from_local(
            str(self.resolve_base_model_dir()),
            device="cpu",
        )
        t3_model = scratch_engine.t3

        if checkpoint_path is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Merged Spanish checkpoint not found: {checkpoint_path}")
            state = load_file(str(checkpoint_path), device="cpu")
            t3_model.load_state_dict(state, strict=True)
        elif adapter_path is not None:
            if not adapter_path.is_dir():
                raise FileNotFoundError(f"Spanish LoRA adapter not found: {adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Spanish LoRA profiles require the 'peft' package. Re-run install_cuda124.sh."
                ) from exc
            t3_model = PeftModel.from_pretrained(
                t3_model,
                str(adapter_path),
                is_trainable=False,
            )
        else:
            raise RuntimeError(f"Profile {profile_id} has neither a checkpoint nor an adapter.")

        t3_model = t3_model.to(server.DEVICE).eval()
        engine = ChatterboxTurboTTS(
            t3=t3_model,
            s3gen=base_engine.s3gen,
            ve=base_engine.ve,
            tokenizer=base_engine.tokenizer,
            device=server.DEVICE,
            conds=None,
        )
        self._apply_persona_conditioning(
            engine,
            reference_path=reference_path,
            bundle_path=bundle_path,
        )

        # The temporary loader supplied only the profile T3. The decoder, voice
        # encoder and tokenizer are deliberately shared with the English engine.
        scratch_engine.t3 = None  # type: ignore[assignment]
        del scratch_engine
        gc.collect()

        self._profile_engines[profile_id] = engine
        self._profile_engines.move_to_end(profile_id)
        while len(self._profile_engines) > self.profile_cache_size:
            evicted_id, evicted_engine = self._profile_engines.popitem(last=False)
            self._release_engine(evicted_engine, evicted_id)

        server.touch_model_usage()
        return engine

    def normalize_voice_path(self, voice: Optional[str]) -> Optional[Path]:
        if self.enabled:
            profile_id = self.resolve_profile_id(voice)
            if profile_id:
                return self.marker_for_profile(profile_id)
        assert self._original_normalize_voice_path is not None
        return self._original_normalize_voice_path(voice)

    def generate_chunk_locked(
        self,
        *,
        text: str,
        voice_path: Optional[Path],
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        norm_loudness: bool,
        seed: int,
    ):
        profile_id = self.profile_from_marker(voice_path)
        if not profile_id:
            assert self._original_generate_chunk_locked is not None
            return self._original_generate_chunk_locked(
                text=text,
                voice_path=voice_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
                seed=seed,
            )

        server = self.server
        normalized_text = normalize_spanish_text(text)
        with server.model_lock:
            was_cached = profile_id in self._profile_engines
            engine = self.ensure_spanish_engine_loaded_locked(profile_id)
            self._validate_spanish_tags(normalized_text, engine)
            server.set_seed(seed)

            with torch.inference_mode():
                waveform = engine.generate(
                    normalized_text,
                    audio_prompt_path=None,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                    norm_loudness=bool(norm_loudness),
                )

            sample_rate = int(engine.sr)
            server.touch_model_usage()
            return server.tensor_to_float_array(waveform), sample_rate, was_cached

    def clear_spanish_profiles_locked(self, reason: str) -> None:
        if self._profile_engines:
            self.server.logger.info(
                "Unloading %d Spanish profile(s) (%s).",
                len(self._profile_engines),
                reason,
            )
        while self._profile_engines:
            profile_id, engine = self._profile_engines.popitem(last=False)
            self._release_engine(engine, profile_id)

    def unload_model_locked(self, reason: str) -> None:
        self.clear_spanish_profiles_locked(reason)
        assert self._original_unload_model_locked is not None
        self._original_unload_model_locked(reason)

    def available_voices_payload(self) -> dict[str, list[dict[str, str]]]:
        assert self._original_available_voices_payload is not None
        payload = self._original_available_voices_payload()
        voices = payload.setdefault("voices", [])
        if self.enabled:
            for profile in SPANISH_PROFILES.values():
                voices.append(
                    {
                        "id": profile.profile_id,
                        "name": profile.name,
                        "language": profile.language,
                        "type": profile.mode,
                    }
                )
        return payload

    def profiles_payload(self) -> dict[str, Any]:
        loaded = set(self._profile_engines)
        return {
            "enabled": self.enabled,
            "selection": "Set the OpenAI voice field to a profile id.",
            "default_spanish_profile": self.default_profile,
            "profiles": [
                {
                    "id": profile.profile_id,
                    "name": profile.name,
                    "language": profile.language,
                    "mode": profile.mode,
                    "conditioning_profile": profile.conditioning_profile or profile.profile_id,
                    "loaded": profile.profile_id in loaded,
                }
                for profile in SPANISH_PROFILES.values()
            ],
        }

    def runtime_status(self, include_sensitive: bool = True) -> dict[str, Any]:
        assert self._original_runtime_status is not None
        status = self._original_runtime_status(include_sensitive=include_sensitive)
        multilingual: dict[str, Any] = {
            "spanish_enabled": self.enabled,
            "spanish_model_repo": self.spanish_repo,
            "spanish_model_revision": self.spanish_revision,
            "default_spanish_profile": self.default_profile,
            "loaded_spanish_profiles": list(self._profile_engines),
            "spanish_profile_cache_size": self.profile_cache_size,
            "preload_profiles": self.preload_profiles,
            "profile_selection": "voice",
        }
        if include_sensitive:
            multilingual["base_model_dir"] = (
                str(self._base_model_dir)
                if self._base_model_dir
                else self.base_model_dir_override or None
            )
            multilingual["spanish_model_dir"] = (
                str(self._spanish_model_dir)
                if self._spanish_model_dir
                else self.spanish_model_dir_override or None
            )
        status["multilingual"] = multilingual
        return status

    async def run_generation(self, **kwargs: Any):
        assert self._original_run_generation is not None
        audio_bytes, metadata = await self._original_run_generation(**kwargs)
        profile_id = self.profile_from_marker(kwargs.get("voice_path"))
        if profile_id:
            profile = SPANISH_PROFILES[profile_id]
            metadata["profile"] = profile_id
            metadata["language"] = profile.language
        else:
            metadata["profile"] = "english"
            metadata["language"] = "en"
        return audio_bytes, metadata

    def response_with_audio(self, audio_bytes: bytes, metadata: dict[str, Any]):
        assert self._original_response_with_audio is not None
        response = self._original_response_with_audio(audio_bytes, metadata)
        if metadata.get("profile"):
            response.headers["X-Chatterbox-Profile"] = str(metadata["profile"])
        if metadata.get("language"):
            response.headers["X-Chatterbox-Language"] = str(metadata["language"])
        return response

    def warm_worker_model_if_needed(self) -> None:
        assert self._original_warm_worker_model_if_needed is not None
        self._original_warm_worker_model_if_needed()
        if not self.enabled or not self.preload_profiles:
            return
        with self.server.model_lock:
            for profile_id in self.preload_profiles:
                self.ensure_spanish_engine_loaded_locked(profile_id)

    def install(self) -> "MultilingualRuntime":
        server = self.server
        if getattr(server, "_multilingual_runtime_installed", False):
            return getattr(server, "_multilingual_runtime")

        self._original_normalize_voice_path = server.normalize_voice_path
        self._original_generate_chunk_locked = server.generate_chunk_locked
        self._original_unload_model_locked = server.unload_model_locked
        self._original_available_voices_payload = server.available_voices_payload
        self._original_runtime_status = server.runtime_status
        self._original_run_generation = server.run_generation
        self._original_response_with_audio = server.response_with_audio
        self._original_warm_worker_model_if_needed = server.warm_worker_model_if_needed

        server.ensure_model_loaded_locked = self.ensure_english_model_loaded_locked
        server.normalize_voice_path = self.normalize_voice_path
        server.generate_chunk_locked = self.generate_chunk_locked
        server.unload_model_locked = self.unload_model_locked
        server.available_voices_payload = self.available_voices_payload
        server.runtime_status = self.runtime_status
        server.run_generation = self.run_generation
        server.response_with_audio = self.response_with_audio
        server.warm_worker_model_if_needed = self.warm_worker_model_if_needed

        server._multilingual_runtime_installed = True
        server._multilingual_runtime = self
        server.app.title = "Chatterbox Turbo English + Spanish FastAPI"
        server.app.version = "4.0.0"

        @server.app.get("/profiles", dependencies=[server.Depends(server.require_api_key)])
        @server.app.get("/v1/profiles", dependencies=[server.Depends(server.require_api_key)])
        def list_profiles() -> dict[str, Any]:
            return self.profiles_payload()

        @server.app.post(
            "/v1/audio/speech/{profile_id}",
            dependencies=[server.Depends(server.require_api_key)],
        )
        async def profile_speech(profile_id: str, req: server.SpeechRequest):
            canonical = self.resolve_profile_id(profile_id)
            if canonical is None:
                raise server.HTTPException(status_code=404, detail=f"Unknown profile: {profile_id}")
            req.voice = canonical
            response = await server.openai_style_speech(req)
            if hasattr(response, "headers"):
                profile = SPANISH_PROFILES[canonical]
                response.headers["X-Chatterbox-Profile"] = canonical
                response.headers["X-Chatterbox-Language"] = profile.language
            return response

        server.logger.info(
            "Multilingual runtime installed: English plus Spanish profiles %s; cache_size=%d.",
            ", ".join(SPANISH_PROFILES),
            self.profile_cache_size,
        )
        return self


def install_multilingual_runtime(server_module: Any) -> MultilingualRuntime:
    return MultilingualRuntime(server_module).install()
