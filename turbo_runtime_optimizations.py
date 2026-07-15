from __future__ import annotations

import hashlib
import inspect
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any


EXPECTED_SOURCES = {
    "0.1.6": {
        "inference_turbo": "f311e763626cb781171855cbd39f6b900f01a16a5fc83de07ba884693c193bef",
        "prepare_conditioning": "35526c9ea80d14ddf3fa9a000a1bbc30267c22258e4ba87a870cdfb71387d69b",
        "basic_euler": "38cabe972357a74756dd03b37e4cec26f96da7174fddc0146c2edd9993a1f625",
    },
}


@dataclass(frozen=True)
class TurboOptimizationState:
    mode: str
    conditioning_cache_size: int
    source_verified: bool
    package_version: str | None


def _source_sha256(value: Any) -> str:
    source = (
        textwrap.dedent(inspect.getsource(inspect.unwrap(value)))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .rstrip()
    )
    return hashlib.sha256(source.encode()).hexdigest()


def _tensor_fingerprint(value: Any) -> Any:
    import torch

    if not torch.is_tensor(value):
        return (type(value).__name__, value)
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (
        id(value),
        value.data_ptr(),
        version,
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
    )


def _conditioning_fingerprint(t3: Any, cond: Any) -> tuple[Any, ...]:
    condition_fields = (
        "speaker_emb",
        "clap_emb",
        "cond_prompt_speech_tokens",
        "cond_prompt_speech_emb",
        "emotion_adv",
    )
    fields = tuple(
        (name, _tensor_fingerprint(getattr(cond, name, None)))
        for name in condition_fields
    )
    parameters = tuple(
        (name, _tensor_fingerprint(value))
        for name, value in t3.cond_enc.named_parameters()
    )
    speech_embedding = _tensor_fingerprint(t3.speech_emb.weight)
    return fields, parameters, speech_embedding


def _refresh_derived_prompt_embedding(t3: Any, cond: Any) -> None:
    provenance = getattr(cond, "_turbo_prompt_embedding_provenance", None)
    current_embedding = getattr(cond, "cond_prompt_speech_emb", None)
    if provenance is None:
        return
    derived_embedding, source_fingerprint = provenance
    if current_embedding is not derived_embedding:
        delattr(cond, "_turbo_prompt_embedding_provenance")
        return
    current_sources = (
        _tensor_fingerprint(getattr(cond, "cond_prompt_speech_tokens", None)),
        _tensor_fingerprint(t3.speech_emb.weight),
    )
    if current_sources != source_fingerprint:
        cond.cond_prompt_speech_emb = None
        delattr(cond, "_turbo_prompt_embedding_provenance")


def _cached_prepare_conditioning(self: Any, t3_cond: Any) -> Any:
    _refresh_derived_prompt_embedding(self, t3_cond)
    cache: "OrderedDict[int, tuple[Any, tuple[Any, ...], Any]]" = getattr(
        self,
        "_turbo_conditioning_cache",
        OrderedDict(),
    )
    cache_size = int(getattr(self, "_turbo_conditioning_cache_size", 8))
    key = id(t3_cond)
    fingerprint = _conditioning_fingerprint(self, t3_cond)
    cached = cache.get(key)
    if cached is not None and cached[0] is t3_cond and cached[1] == fingerprint:
        cache.move_to_end(key)
        self._turbo_conditioning_cache_hits = int(getattr(self, "_turbo_conditioning_cache_hits", 0)) + 1
        return cached[2]

    prompt_embedding_was_missing = getattr(t3_cond, "cond_prompt_speech_emb", None) is None
    original = type(self)._turbo_original_prepare_conditioning
    result = original(self, t3_cond)
    if prompt_embedding_was_missing and getattr(t3_cond, "cond_prompt_speech_emb", None) is not None:
        t3_cond._turbo_prompt_embedding_provenance = (
            t3_cond.cond_prompt_speech_emb,
            (
                _tensor_fingerprint(getattr(t3_cond, "cond_prompt_speech_tokens", None)),
                _tensor_fingerprint(self.speech_emb.weight),
            ),
        )
    self._turbo_conditioning_cache_misses = int(getattr(self, "_turbo_conditioning_cache_misses", 0)) + 1
    cache[key] = (t3_cond, _conditioning_fingerprint(self, t3_cond), result)
    cache.move_to_end(key)
    while len(cache) > max(1, cache_size):
        cache.popitem(last=False)
    self._turbo_conditioning_cache = cache
    return result


def _inference_turbo_fast(
    self: Any,
    t3_cond: Any,
    text_tokens: Any,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_gen_len: int = 1000,
) -> Any:
    import torch
    import torch.nn.functional as F
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    logits_processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        logits_processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

    speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    llm_outputs = self.tfmr(inputs_embeds=embeds, use_cache=True)
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values
    speech_logits = self.speech_head(hidden_states[:, -1:])

    processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    current_speech_token = torch.multinomial(probs, num_samples=1)

    generated_tokens = torch.empty(
        (current_speech_token.size(0), max(1, max_gen_len + 1)),
        dtype=current_speech_token.dtype,
        device=current_speech_token.device,
    )
    generated_tokens[:, 0:1] = current_speech_token
    generated_count = 1

    for _ in range(max_gen_len):
        current_speech_embed = self.speech_emb(current_speech_token)
        llm_outputs = self.tfmr(
            inputs_embeds=current_speech_embed,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = self.speech_head(hidden_states)

        input_ids = generated_tokens[:, :generated_count]
        if not input_ids.is_contiguous():
            input_ids = input_ids.contiguous()
        processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
        if torch.all(processed_logits == -float("inf")):
            print("Warning: All logits are -inf")
            break

        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, generated_count : generated_count + 1] = next_speech_token
        generated_count += 1
        current_speech_token = next_speech_token
        if torch.all(next_speech_token == self.hp.stop_speech_token):
            break

    all_tokens = generated_tokens[:, :generated_count].clone()
    if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
        all_tokens = all_tokens[:, :-1]
    return all_tokens


def _basic_euler_fast(self: Any, x: Any, t_span: Any, mu: Any, mask: Any, spks: Any, cond: Any) -> Any:
    from chatterbox.models.s3gen.flow_matching import cast_all

    in_dtype = x.dtype
    x, t_span, mu, mask, spks, cond = cast_all(
        x,
        t_span,
        mu,
        mask,
        spks,
        cond,
        dtype=self.estimator.dtype,
    )
    for t, r in zip(t_span[..., :-1], t_span[..., 1:]):
        t, r = t[None], r[None]
        dxdt = self.estimator.forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        dt = r - t
        x = x + dt * dxdt
    return x.to(in_dtype)


def configure_turbo_runtime(
    tts_turbo: Any,
    *,
    enabled: bool,
    conditioning_cache_size: int,
) -> TurboOptimizationState:
    if not enabled:
        return TurboOptimizationState("disabled", conditioning_cache_size, False, None)

    t3_class = getattr(tts_turbo, "T3", None)
    if t3_class is None:
        if getattr(tts_turbo, "__file__", None):
            raise RuntimeError("Installed chatterbox.tts_turbo exposes no T3 class for the Turbo fast path.")
        return TurboOptimizationState("enabled-unverified", conditioning_cache_size, False, None)

    package_version = version("chatterbox-tts")
    expected_hashes = EXPECTED_SOURCES.get(package_version)
    if expected_hashes is None:
        raise RuntimeError(f"Turbo fast path does not support chatterbox-tts {package_version}.")

    if getattr(t3_class, "_turbo_fast_path_installed", False):
        return TurboOptimizationState("enabled", conditioning_cache_size, True, package_version)

    from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
    import torch

    actual_hashes = {
        "inference_turbo": _source_sha256(t3_class.inference_turbo),
        "prepare_conditioning": _source_sha256(t3_class.prepare_conditioning),
        "basic_euler": _source_sha256(CausalConditionalCFM.basic_euler),
    }
    mismatches = [name for name, digest in actual_hashes.items() if digest != expected_hashes[name]]
    if mismatches:
        details = ", ".join(
            f"{name} expected={expected_hashes[name]} actual={actual_hashes[name]}"
            for name in mismatches
        )
        raise RuntimeError(f"Turbo fast path source mismatch for chatterbox-tts {package_version}: {details}")

    t3_class._turbo_original_prepare_conditioning = t3_class.prepare_conditioning
    t3_class._turbo_original_inference_turbo = t3_class.inference_turbo
    t3_class._turbo_conditioning_cache_size = max(1, conditioning_cache_size)
    t3_class.prepare_conditioning = _cached_prepare_conditioning
    t3_class.inference_turbo = torch.inference_mode()(_inference_turbo_fast)
    t3_class._turbo_fast_path_installed = True
    CausalConditionalCFM._turbo_original_basic_euler = CausalConditionalCFM.basic_euler
    CausalConditionalCFM.basic_euler = torch.inference_mode()(_basic_euler_fast)

    return TurboOptimizationState("enabled", conditioning_cache_size, True, package_version)
