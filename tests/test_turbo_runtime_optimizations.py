from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "manual_seed"):
    pytest.skip("requires the real torch package", allow_module_level=True)
pytest.importorskip("transformers")

from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond

from turbo_runtime_optimizations import (
    _basic_euler_fast,
    _cached_prepare_conditioning,
    _inference_turbo_fast,
    configure_turbo_runtime,
)


class FakeOutput:
    def __init__(self, hidden_states, past_key_values):
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values

    def __getitem__(self, index):
        assert index == 0
        return self.hidden_states


class FakeTransformer:
    def __call__(self, *, inputs_embeds, past_key_values=None, use_cache=True):
        assert use_cache
        step = 0 if past_key_values is None else past_key_values
        hidden = inputs_embeds + float(step) / 10
        return FakeOutput(hidden, step + 1)


class FakeSpeechHead(torch.nn.Module):
    def forward(self, hidden_states):
        base = torch.linspace(-1.5, 1.5, 8, device=hidden_states.device)
        return base.view(1, 1, -1).expand(hidden_states.size(0), hidden_states.size(1), -1)


class FakeInferenceT3:
    def __init__(self):
        self.hp = SimpleNamespace(start_speech_token=0, stop_speech_token=7)
        self.tfmr = FakeTransformer()
        self.speech_emb = torch.nn.Embedding(8, 4)
        self.speech_head = FakeSpeechHead()

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight):
        assert cfg_weight == 0.0
        return torch.zeros(text_tokens.size(0), 3, 4), 0


def test_fast_inference_preserves_tokens_and_rng_state():
    text_tokens = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)

    torch.manual_seed(1234)
    original_inference = getattr(T3, "_turbo_original_inference_turbo", T3.inference_turbo)
    baseline = original_inference(
        FakeInferenceT3(),
        None,
        text_tokens,
        temperature=0.8,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=5,
    )
    baseline_rng = torch.random.get_rng_state()

    torch.manual_seed(1234)
    optimized = _inference_turbo_fast(
        FakeInferenceT3(),
        None,
        text_tokens,
        temperature=0.8,
        top_k=4,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=5,
    )
    optimized_rng = torch.random.get_rng_state()

    assert torch.equal(optimized, baseline)
    assert torch.equal(optimized_rng, baseline_rng)


class CountingConditionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def forward(self, cond):
        self.calls += 1
        return cond.speaker_emb * self.weight


class FakeConditioningT3:
    _turbo_conditioning_cache_size = 2

    def __init__(self):
        self.cond_enc = CountingConditionEncoder()
        self.speech_emb = torch.nn.Embedding(8, 2)

    def original_prepare_conditioning(self, cond):
        if cond.cond_prompt_speech_tokens is not None and cond.cond_prompt_speech_emb is None:
            cond.cond_prompt_speech_emb = self.speech_emb(cond.cond_prompt_speech_tokens)
        return self.cond_enc(cond)


FakeConditioningT3._turbo_original_prepare_conditioning = FakeConditioningT3.original_prepare_conditioning


def make_conditioning():
    return T3Cond(
        speaker_emb=torch.ones(1, 2),
        cond_prompt_speech_tokens=torch.tensor([[1, 2]], dtype=torch.long),
        emotion_adv=torch.ones(1, 1, 1),
    )


def test_conditioning_cache_hits_and_invalidates_on_tensor_mutation():
    t3 = FakeConditioningT3()
    cond = make_conditioning()

    first = _cached_prepare_conditioning(t3, cond)
    second = _cached_prepare_conditioning(t3, cond)
    with torch.no_grad():
        cond.speaker_emb.add_(1)
    third = _cached_prepare_conditioning(t3, cond)

    assert second is first
    assert third is not first
    assert t3.cond_enc.calls == 2


def test_derived_prompt_embedding_recomputes_when_tokens_change():
    t3 = FakeConditioningT3()
    cond = make_conditioning()

    _cached_prepare_conditioning(t3, cond)
    original_embedding = cond.cond_prompt_speech_emb
    cond.cond_prompt_speech_tokens[0, 0] = 3
    _cached_prepare_conditioning(t3, cond)

    assert cond.cond_prompt_speech_emb is not original_embedding
    assert t3.cond_enc.calls == 2


def test_conditioning_caches_are_isolated_per_engine():
    cond = make_conditioning()
    first = FakeConditioningT3()
    second = FakeConditioningT3()

    _cached_prepare_conditioning(first, cond)
    _cached_prepare_conditioning(first, cond)
    _cached_prepare_conditioning(second, cond)

    assert first.cond_enc.calls == 1
    assert second.cond_enc.calls == 1


def test_conditioning_cache_respects_lru_limit():
    t3 = FakeConditioningT3()

    for _ in range(3):
        _cached_prepare_conditioning(t3, make_conditioning())

    assert len(t3._turbo_conditioning_cache) == 2


class FakeEstimator:
    dtype = torch.float32

    def forward(self, x, *, mask, mu, t, spks, cond, r):
        return x * 0.1 + mu * 0.2 + cond * 0.05


def test_basic_euler_fast_path_preserves_output():
    flow = SimpleNamespace(estimator=FakeEstimator())
    x = torch.ones(1, 2, 3)
    t_span = torch.linspace(0, 1, 3)
    mu = torch.full_like(x, 0.5)
    mask = torch.ones(1, 1, 3)
    spks = torch.ones(1, 2)
    cond = torch.full_like(x, 0.25)

    original_basic_euler = getattr(
        CausalConditionalCFM,
        "_turbo_original_basic_euler",
        CausalConditionalCFM.basic_euler,
    )
    baseline = original_basic_euler(flow, x, t_span, mu, mask, spks, cond)
    optimized = _basic_euler_fast(flow, x, t_span, mu, mask, spks, cond)

    assert torch.equal(optimized, baseline)


def test_disabled_fast_path_does_not_require_runtime_symbols():
    state = configure_turbo_runtime(SimpleNamespace(), enabled=False, conditioning_cache_size=8)

    assert state.mode == "disabled"
    assert state.source_verified is False
    assert state.package_version is None
