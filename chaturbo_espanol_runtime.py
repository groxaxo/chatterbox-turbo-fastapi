from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from safetensors.torch import load_file

from multilingual_runtime import MultilingualRuntime, SPANISH_PROFILES


AR_MERGED_RELATIVE_PATH = "lucia-ar/t3_turbo_finetuned_merged.safetensors"
CONTINUAL_LORA_PROFILES = {
    "lucia-latam",
    "lucia-cl-pilot",
    "lucia-co-pilot",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ChaturboEspanolRuntime(MultilingualRuntime):
    """Artifact-aware runtime for the groxaxo/chaturbo-espanol release.

    The LATAM and country-pilot LoRAs are continual adapters. Their frozen base is
    the Lucía AR merged T3, not the untouched official T3. The official
    ResembleAI model is still required to construct the architecture and provide
    the shared S3Gen decoder, voice encoder, and tokenizer.
    """

    def __init__(self, server_module: Any):
        super().__init__(server_module)
        self.verify_model_provenance = _env_bool("VERIFY_MODEL_PROVENANCE", True)
        self._sha256_cache: dict[str, tuple[tuple[int, int], str]] = {}

    def _cached_sha256(self, path: Path) -> str:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        key = str(path.resolve())
        cached = self._sha256_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._sha256_cache[key] = (signature, value)
        return value

    def _validate_adapter_provenance(
        self,
        *,
        profile_id: str,
        adapter_path: Path,
        warm_start_path: Path,
    ) -> None:
        if not self.verify_model_provenance:
            return

        adapter_model = adapter_path / "adapter_model.safetensors"
        if not adapter_model.is_file():
            raise FileNotFoundError(f"Adapter weights not found: {adapter_model}")

        provenance_path = adapter_path / "adapter_provenance.json"
        if not provenance_path.is_file():
            self.server.logger.warning(
                "Profile '%s' has no adapter_provenance.json; strict PEFT loading will still run, "
                "but content hashes cannot be attested.",
                profile_id,
            )
            return

        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_adapter_hash = payload.get("adapter_sha256")
        if expected_adapter_hash:
            actual_adapter_hash = self._cached_sha256(adapter_model)
            if actual_adapter_hash != expected_adapter_hash:
                raise RuntimeError(
                    f"Adapter provenance mismatch for {profile_id}: "
                    f"expected={expected_adapter_hash}, actual={actual_adapter_hash}"
                )

        expected_warm_hash = payload.get("warm_start_sha256")
        if not expected_warm_hash:
            raise RuntimeError(
                f"Profile {profile_id} is configured as an AR-anchored continual LoRA, "
                "but its provenance does not declare a warm-start checkpoint."
            )
        actual_warm_hash = self._cached_sha256(warm_start_path)
        if actual_warm_hash != expected_warm_hash:
            raise RuntimeError(
                f"Warm-start provenance mismatch for {profile_id}: "
                f"expected={expected_warm_hash}, actual={actual_warm_hash}"
            )

    def _warm_start_path_for_profile(self, profile_id: str) -> Optional[Path]:
        if profile_id not in CONTINUAL_LORA_PROFILES:
            return None
        return self.resolve_spanish_model_dir() / AR_MERGED_RELATIVE_PATH

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
        warm_start_path = self._warm_start_path_for_profile(profile_id)

        server.wait_for_free_vram_if_needed()
        server.logger.info(
            "Loading Spanish profile '%s' (%s, %s%s).",
            profile_id,
            profile.language,
            profile.mode,
            ", AR-merged warm base" if warm_start_path else "",
        )

        scratch_engine: Optional[ChatterboxTurboTTS] = None
        engine: Optional[ChatterboxTurboTTS] = None
        try:
            scratch_engine = ChatterboxTurboTTS.from_local(
                str(self.resolve_base_model_dir()),
                device="cpu",
            )
            t3_model = scratch_engine.t3

            if warm_start_path is not None:
                if not warm_start_path.is_file():
                    raise FileNotFoundError(
                        f"AR merged warm-start checkpoint required by {profile_id} was not found: "
                        f"{warm_start_path}"
                    )
                warm_state = load_file(str(warm_start_path), device="cpu")
                t3_model.load_state_dict(warm_state, strict=True)

            if checkpoint_path is not None:
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Merged Spanish checkpoint not found: {checkpoint_path}"
                    )
                checkpoint_state = load_file(str(checkpoint_path), device="cpu")
                t3_model.load_state_dict(checkpoint_state, strict=True)
            elif adapter_path is not None:
                if not adapter_path.is_dir():
                    raise FileNotFoundError(f"Spanish LoRA adapter not found: {adapter_path}")
                if warm_start_path is None:
                    raise RuntimeError(
                        f"Adapter profile {profile_id} has no declared continual-learning base."
                    )
                self._validate_adapter_provenance(
                    profile_id=profile_id,
                    adapter_path=adapter_path,
                    warm_start_path=warm_start_path,
                )
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Spanish LoRA profiles require the 'peft' package. "
                        "Re-run install_cuda124.sh."
                    ) from exc
                t3_model = PeftModel.from_pretrained(
                    t3_model,
                    str(adapter_path),
                    is_trainable=False,
                )
            else:
                raise RuntimeError(
                    f"Profile {profile_id} has neither a checkpoint nor an adapter."
                )

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

            # Transfer T3 ownership to the final engine. The temporary loader's
            # decoder and voice encoder are then released; inference shares the
            # already resident English components.
            scratch_engine.t3 = None  # type: ignore[assignment]
            self._profile_engines[profile_id] = engine
            self._profile_engines.move_to_end(profile_id)
            while len(self._profile_engines) > self.profile_cache_size:
                evicted_id, evicted_engine = self._profile_engines.popitem(last=False)
                self._release_engine(evicted_engine, evicted_id)

            server.touch_model_usage()
            return engine
        except Exception:
            if engine is not None and profile_id not in self._profile_engines:
                self._release_engine(engine, profile_id)
            raise
        finally:
            if scratch_engine is not None:
                del scratch_engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def profiles_payload(self) -> dict[str, Any]:
        payload = super().profiles_payload()
        for item in payload["profiles"]:
            if item["id"] in CONTINUAL_LORA_PROFILES:
                item["base_profile"] = "lucia-ar"
                item["base_checkpoint"] = AR_MERGED_RELATIVE_PATH
            else:
                item["base_profile"] = "official-turbo"
        payload["artifact_chain"] = {
            "english": "ResembleAI/chatterbox-turbo",
            "lucia-ar": "official Turbo + merged Lucía AR T3",
            "continual_loras": "official Turbo architecture + Lucía AR merged T3 + profile adapter",
        }
        payload["verify_model_provenance"] = self.verify_model_provenance
        return payload

    def runtime_status(self, include_sensitive: bool = True) -> dict[str, Any]:
        status = super().runtime_status(include_sensitive=include_sensitive)
        status["multilingual"]["continual_lora_base"] = AR_MERGED_RELATIVE_PATH
        status["multilingual"]["verify_model_provenance"] = self.verify_model_provenance
        return status


def install_chaturbo_espanol_runtime(server_module: Any) -> ChaturboEspanolRuntime:
    return ChaturboEspanolRuntime(server_module).install()
