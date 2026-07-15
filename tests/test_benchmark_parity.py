from benchmarks.benchmark_parity import _build_comparison, _deterministic_capture_env


def test_deterministic_capture_env_pins_cuda_before_process_start() -> None:
    baseline = _deterministic_capture_env(performance_enabled=False)
    candidate = _deterministic_capture_env(performance_enabled=True)

    assert baseline["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert baseline["PYTHONHASHSEED"] == "0"
    assert baseline["TURBO_PARITY_DETERMINISTIC"] == "1"
    assert baseline["TURBO_PERFORMANCE_RUNTIME"] == "0"
    assert candidate["TURBO_PERFORMANCE_RUNTIME"] == "1"


def test_comparison_detects_waveform_drift_with_matching_tokens() -> None:
    common = {
        "case": {"name": "english-short"},
        "run_index": 0,
        "speech_token_sha256": "tokens",
        "sample_rate": 24_000,
        "sample_count": 100,
        "generation_ms": 1.0,
    }
    baseline = {"records": [{**common, "waveform_float32_sha256": "a", "pcm16_sha256": "a"}]}
    candidate = {"records": [{**common, "waveform_float32_sha256": "b", "pcm16_sha256": "b"}]}

    comparison = _build_comparison(baseline, candidate)

    assert comparison["exact"] is False
    assert comparison["pairs"][0]["fields"]["speech_tokens"] is True
    assert comparison["pairs"][0]["fields"]["waveform_float32"] is False
