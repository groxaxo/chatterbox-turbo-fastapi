from __future__ import annotations

import inspect
import logging
import types
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import punc_norm
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from performance_support import conditioning_fingerprint, resolve_t3_core, source_matches


@dataclass
class ConditioningCacheEntry:
    conditional_ref: Callable[[], Any]
    fingerprint: tuple[Any, ...]
    encoded: torch.Tensor


class EngineOptimizer:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    def stats(self, core: Any) -> dict[str, int]:
        stats = getattr(core, "_turbo_perf_stats", None)
        if stats is None:
            stats = {
                "conditioning_cache_hits": 0,
                "conditioning_cache_misses": 0,
                "preallocated_generations": 0,
                "preallocated_fallbacks": 0,
                "silence_cache_hits": 0,
                "silence_cache_misses": 0,
            }
            core._turbo_perf_stats = stats
        return stats

    def install_conditioning_cache(self, core: Any) -> None:
        runtime = self.runtime
        if not runtime.cache_encoded_conditioning or getattr(core, "_turbo_perf_conditioning_cache_installed", False):
            return
        original = core.prepare_conditioning
        if not source_matches(original, ("cond_prompt_speech_emb", "self.speech_emb", "self.cond_enc")):
            runtime.server.logger.warning("Skipping encoded-conditioning cache: guarded source structure did not match.")
            return
        cache: "OrderedDict[int, ConditioningCacheEntry]" = OrderedDict()

        def clear_cache() -> None:
            cache.clear()

        def cached_prepare(module: Any, conditional: Any) -> torch.Tensor:
            if bool(getattr(module, "training", False)) or torch.is_grad_enabled():
                return original(conditional)
            stats = self.stats(module)
            key = id(conditional)
            fingerprint = conditioning_fingerprint(module, conditional)
            entry = cache.get(key)
            if entry is not None and entry.conditional_ref() is conditional and entry.fingerprint == fingerprint:
                cache.move_to_end(key)
                stats["conditioning_cache_hits"] += 1
                return entry.encoded
            encoded = original(conditional)
            if encoded.requires_grad:
                return encoded
            try:
                conditional_ref: Callable[[], Any] = weakref.ref(conditional)
            except TypeError:
                conditional_ref = lambda conditional=conditional: conditional
            cache[key] = ConditioningCacheEntry(
                conditional_ref, conditioning_fingerprint(module, conditional), encoded.detach()
            )
            cache.move_to_end(key)
            while len(cache) > runtime.conditioning_cache_size:
                cache.popitem(last=False)
            stats["conditioning_cache_misses"] += 1
            return encoded

        core._turbo_perf_original_prepare_conditioning = original
        core._turbo_perf_clear_conditioning_cache = clear_cache
        core.prepare_conditioning = types.MethodType(cached_prepare, core)
        core._turbo_perf_conditioning_cache_installed = True

    def install_preallocated_inference(self, core: Any) -> None:
        runtime = self.runtime
        if not runtime.preallocate_token_ids or getattr(core, "_turbo_perf_preallocated_inference_installed", False):
            return
        original = core.inference_turbo
        if not source_matches(original, (
            "generated_speech_tokens", "torch.cat(generated_speech_tokens",
            "RepetitionPenaltyLogitsProcessor", "max_gen_len", "stop_speech_token",
        )):
            runtime.server.logger.warning("Skipping preallocated token history: guarded source structure did not match.")
            return

        @torch.inference_mode()
        def preallocated(module: Any, t3_cond: Any, text_tokens: torch.Tensor,
                         temperature: float = 0.8, top_k: int = 1000, top_p: float = 0.95,
                         repetition_penalty: float = 1.2, max_gen_len: int = 1000) -> torch.Tensor:
            if text_tokens.ndim != 2 or text_tokens.shape[0] != 1 or max_gen_len < 0:
                self.stats(module)["preallocated_fallbacks"] += 1
                return original(t3_cond, text_tokens, temperature=temperature, top_k=top_k,
                                top_p=top_p, repetition_penalty=repetition_penalty,
                                max_gen_len=max_gen_len)
            processors = LogitsProcessorList()
            if temperature > 0 and temperature != 1.0:
                processors.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                processors.append(TopKLogitsWarper(top_k))
            if top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p))
            if repetition_penalty != 1.0:
                processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

            start = module.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
            embeds, _ = module.prepare_input_embeds(
                t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=start, cfg_weight=0.0
            )
            outputs = module.tfmr(inputs_embeds=embeds, use_cache=True)
            past = outputs.past_key_values
            logits = module.speech_head(outputs[0][:, -1:])
            probs = F.softmax(processors(start, logits[:, -1, :]), dim=-1)
            current = torch.multinomial(probs, num_samples=1)
            buffer = torch.empty((1, max_gen_len + 1), dtype=torch.long, device=current.device)
            buffer[:, 0:1].copy_(current)
            count = 1
            iterator: Any = range(max_gen_len)
            if not runtime.disable_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator)
            for _ in iterator:
                outputs = module.tfmr(
                    inputs_embeds=module.speech_emb(current), past_key_values=past, use_cache=True
                )
                past = outputs.past_key_values
                logits = module.speech_head(outputs[0])
                processed = processors(buffer[:, :count], logits[:, -1, :])
                if runtime.strict_logit_checks and torch.all(processed == -float("inf")):
                    print("Warning: All logits are -inf")
                    break
                current = torch.multinomial(F.softmax(processed, dim=-1), num_samples=1)
                buffer[:, count:count + 1].copy_(current)
                count += 1
                if torch.all(current == module.hp.stop_speech_token):
                    break
            tokens = buffer[:, :count]
            if tokens.size(1) > 0 and tokens[0, -1] == module.hp.stop_speech_token:
                tokens = tokens[:, :-1]
            self.stats(module)["preallocated_generations"] += 1
            return tokens

        core._turbo_perf_original_inference_turbo = original
        core.inference_turbo = types.MethodType(preallocated, core)
        core._turbo_perf_preallocated_inference_installed = True

    @staticmethod
    def device_matches(tensor: torch.Tensor, target: Any) -> bool:
        target_device = torch.device(target)
        if tensor.device.type != target_device.type:
            return False
        if target_device.type != "cuda":
            return tensor.device.index == target_device.index
        target_index = target_device.index if target_device.index is not None else torch.cuda.current_device()
        return tensor.device.index == target_index

    def install_generate_rewrite(self, engine: Any, core: Any) -> None:
        runtime = self.runtime
        if not runtime.rewrite_package_generate or not runtime.package_compatible or getattr(engine, "_turbo_perf_generate_installed", False):
            return
        original = engine.generate
        required = {"text", "repetition_penalty", "min_p", "top_p", "audio_prompt_path",
                    "exaggeration", "cfg_weight", "temperature", "top_k", "norm_loudness"}
        if not required.issubset(inspect.signature(original).parameters) or not source_matches(
            original, ("punc_norm", "speech_tokens < 6561", "n_cfm_timesteps=2", "apply_watermark")
        ):
            runtime.server.logger.warning("Skipping package generate rewrite: guarded source structure did not match.")
            return
        silence_cache: dict[tuple[str, Optional[int]], torch.Tensor] = {}

        def optimized(self_engine: Any, text: str, repetition_penalty: float = 1.2,
                      min_p: float = 0.0, top_p: float = 0.95,
                      audio_prompt_path: Optional[str] = None, exaggeration: float = 0.0,
                      cfg_weight: float = 0.0, temperature: float = 0.8,
                      top_k: int = 1000, norm_loudness: bool = True) -> torch.Tensor:
            if audio_prompt_path:
                self_engine.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration,
                                                 norm_loudness=norm_loudness)
                clear = getattr(core, "_turbo_perf_clear_conditioning_cache", None)
                if callable(clear):
                    clear()
            else:
                assert self_engine.conds is not None, "Please prepare conditionals first"
            if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
                logging.getLogger("chatterbox.tts_turbo").warning(
                    "CFG, min_p and exaggeration are not supported by Turbo version and will be ignored."
                )
            normalized = punc_norm(text)
            text_tokens = self_engine.tokenizer(
                normalized, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(self_engine.device)
            speech_tokens = self_engine.t3.inference_turbo(
                t3_cond=self_engine.conds.t3, text_tokens=text_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            speech_tokens = speech_tokens[speech_tokens < 6561]
            if not self.device_matches(speech_tokens, self_engine.device):
                speech_tokens = speech_tokens.to(self_engine.device)
            key = (speech_tokens.device.type, speech_tokens.device.index)
            silence = silence_cache.get(key) if runtime.cache_silence_tensor else None
            if silence is None:
                silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=speech_tokens.device)
                if runtime.cache_silence_tensor:
                    silence_cache[key] = silence
                    self.stats(core)["silence_cache_misses"] += 1
            else:
                self.stats(core)["silence_cache_hits"] += 1
            waveform, _ = self_engine.s3gen.inference(
                speech_tokens=torch.cat([speech_tokens, silence]),
                ref_dict=self_engine.conds.gen, n_cfm_timesteps=2,
            )
            array = waveform.squeeze(0).detach().cpu().numpy()
            watermarked = self_engine.watermarker.apply_watermark(array, sample_rate=self_engine.sr)
            return torch.from_numpy(watermarked).unsqueeze(0)

        engine._turbo_perf_original_generate = original
        engine._turbo_perf_silence_cache = silence_cache
        engine.generate = types.MethodType(optimized, engine)
        engine._turbo_perf_generate_installed = True

    def prepare(self, engine: Any, profile_id: str) -> Any:
        if not self.runtime.enabled:
            return engine
        core = resolve_t3_core(engine.t3)
        self.stats(core)
        if self.runtime.package_compatible:
            self.install_conditioning_cache(core)
            self.install_preallocated_inference(core)
            self.install_generate_rewrite(engine, core)
        engine._turbo_perf_profile_id = profile_id
        engine._turbo_perf_runtime_prepared = True
        return engine
