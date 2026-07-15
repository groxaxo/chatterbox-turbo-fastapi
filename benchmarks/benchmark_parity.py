#!/usr/bin/env python3
"""Isolated baseline/candidate benchmark and exact-parity harness.

The compare command launches baseline and optimized captures in separate Python
processes so mutable Conditionals, RNG state, allocator state, and runtime monkey
patches cannot leak from one side of the comparison to the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    voice: str
    text: str
    seed: int = 1234
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True


class StageAccumulator:
    def __init__(self, torch_module: Any):
        self.torch = torch_module
        self.cpu_ms: dict[str, float] = {}
        self.gpu_ms: dict[str, float] = {}

    def measure(self, name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        use_cuda_events = bool(
            self.torch.cuda.is_available()
            and hasattr(self.torch.cuda, "Event")
        )
        start_event = end_event = None
        if use_cuda_events:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()

        started = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            self.cpu_ms[name] = self.cpu_ms.get(name, 0.0) + elapsed_ms
            if start_event is not None and end_event is not None:
                end_event.record()
                # Benchmark-only synchronization. Production requests do not call this.
                end_event.synchronize()
                gpu_elapsed = float(start_event.elapsed_time(end_event))
                self.gpu_ms[name] = self.gpu_ms.get(name, 0.0) + gpu_elapsed


def _load_cases(path: Path) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Case file must contain a non-empty JSON array")
    return [BenchmarkCase(**item) for item in payload]


def _resolve_engine(server: Any, multilingual: Any, performance: Any, voice_path: Any) -> tuple[Any, str]:
    profile_id = multilingual.profile_from_marker(voice_path)
    with server.model_lock:
        if profile_id:
            engine = multilingual.ensure_spanish_engine_loaded_locked(profile_id)
            profile = profile_id
        else:
            engine = server.ensure_model_loaded_locked()
            profile = "english"
        if performance is not None:
            performance.prepare_engine(engine, profile)
        return engine, profile


def _patch_method(owner: Any, name: str, wrapper: Callable[..., Any]) -> Callable[[], None]:
    original = getattr(owner, name)
    setattr(owner, name, types.MethodType(wrapper, owner))

    def restore() -> None:
        setattr(owner, name, original)

    return restore


def _capture_case(
    *,
    server: Any,
    multilingual: Any,
    performance: Any,
    case: BenchmarkCase,
    run_index: int,
    torch_module: Any,
    np_module: Any,
) -> dict[str, Any]:
    voice_path = server.normalize_voice_path(case.voice)

    load_started = time.perf_counter_ns()
    engine, profile = _resolve_engine(server, multilingual, performance, voice_path)
    engine_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000

    from performance_runtime import _resolve_t3_core

    core = _resolve_t3_core(engine.t3)
    stages = StageAccumulator(torch_module)
    captured: dict[str, Any] = {
        "speech_tokens": None,
        "conditioning_calls": 0,
        "t3_calls": 0,
        "s3_calls": 0,
    }
    restorers: list[Callable[[], None]] = []

    original_prepare = core.prepare_conditioning

    def measured_prepare(module: Any, conditional: Any):
        captured["conditioning_calls"] += 1
        return stages.measure(
            "conditioning",
            original_prepare,
            conditional,
        )

    restorers.append(_patch_method(core, "prepare_conditioning", measured_prepare))

    original_t3 = core.inference_turbo

    def measured_t3(module: Any, *args: Any, **kwargs: Any):
        captured["t3_calls"] += 1
        tokens = stages.measure("t3_total", original_t3, *args, **kwargs)
        captured["speech_tokens"] = tokens.detach().cpu().contiguous()
        return tokens

    restorers.append(_patch_method(core, "inference_turbo", measured_t3))

    original_s3 = engine.s3gen.inference

    def measured_s3(module: Any, *args: Any, **kwargs: Any):
        captured["s3_calls"] += 1
        return stages.measure("s3_total", original_s3, *args, **kwargs)

    restorers.append(_patch_method(engine.s3gen, "inference", measured_s3))

    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()

    generation_started = time.perf_counter_ns()
    try:
        waveform, sample_rate, voice_cache_hit = server.generate_chunk_locked(
            text=case.text,
            voice_path=voice_path,
            temperature=case.temperature,
            top_p=case.top_p,
            top_k=case.top_k,
            repetition_penalty=case.repetition_penalty,
            norm_loudness=case.norm_loudness,
            seed=case.seed,
        )
    finally:
        for restore in reversed(restorers):
            restore()
    generation_ms = (time.perf_counter_ns() - generation_started) / 1_000_000

    if captured["t3_calls"] != 1 or captured["s3_calls"] != 1:
        raise RuntimeError(
            "Benchmark instrumentation did not observe exactly one T3 and one S3 "
            f"call (t3={captured['t3_calls']}, s3={captured['s3_calls']})."
        )
    if captured["speech_tokens"] is None:
        raise RuntimeError("Benchmark did not capture generated speech token IDs")

    encode_started = time.perf_counter_ns()
    pcm16 = server.pcm16_bytes_from_array(waveform)
    encode_ms = (time.perf_counter_ns() - encode_started) / 1_000_000

    array = np_module.asarray(waveform, dtype=np_module.float32)
    tokens = captured["speech_tokens"]
    token_bytes = tokens.numpy().tobytes()
    duration_seconds = float(array.size / sample_rate) if sample_rate else 0.0
    clipping_ratio = float(np_module.mean(np_module.abs(array) >= 1.0)) if array.size else 0.0

    peak_allocated = peak_reserved = None
    if torch_module.cuda.is_available():
        peak_allocated = int(torch_module.cuda.max_memory_allocated())
        peak_reserved = int(torch_module.cuda.max_memory_reserved())

    return {
        "case": asdict(case),
        "profile": profile,
        "run_index": run_index,
        "engine_load_ms": round(engine_load_ms, 4),
        "generation_ms": round(generation_ms, 4),
        "encode_ms": round(encode_ms, 4),
        "audio_seconds": round(duration_seconds, 6),
        "rtf": round((generation_ms / 1000) / duration_seconds, 6)
        if duration_seconds > 0
        else None,
        "sample_rate": int(sample_rate),
        "sample_count": int(array.size),
        "voice_cache_hit": bool(voice_cache_hit),
        "finite": bool(np_module.isfinite(array).all()),
        "clipping_ratio": round(clipping_ratio, 9),
        "speech_token_count": int(tokens.numel()),
        "speech_token_sha256": _sha256_bytes(token_bytes),
        "waveform_float32_sha256": _sha256_bytes(array.tobytes()),
        "pcm16_sha256": _sha256_bytes(pcm16),
        "stage_cpu_ms": {key: round(value, 4) for key, value in stages.cpu_ms.items()},
        "stage_gpu_ms": {key: round(value, 4) for key, value in stages.gpu_ms.items()},
        "instrumentation": {
            "conditioning_calls": captured["conditioning_calls"],
            "t3_calls": captured["t3_calls"],
            "s3_calls": captured["s3_calls"],
        },
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }


def capture(args: argparse.Namespace) -> int:
    os.environ["TURBO_PERFORMANCE_RUNTIME"] = "1" if args.mode == "optimized" else "0"
    os.environ.setdefault("TURBO_DISABLE_PROGRESS", "1")

    import numpy as np
    import torch

    import multilingual_server
    import server

    cases = _load_cases(args.cases)
    performance = getattr(server, "_turbo_performance_runtime", None)
    records: list[dict[str, Any]] = []

    for case in cases:
        for run_index in range(args.runs):
            records.append(
                _capture_case(
                    server=server,
                    multilingual=multilingual_server.runtime,
                    performance=performance,
                    case=case,
                    run_index=run_index,
                    torch_module=torch,
                    np_module=np,
                )
            )

    status = server.runtime_status(include_sensitive=False)
    output = {
        "mode": args.mode,
        "captured_at_unix": time.time(),
        "runtime_status": status,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


def _median(records: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.median(values) if values else None


def _build_comparison(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    baseline_records = baseline["records"]
    optimized_records = optimized["records"]
    if len(baseline_records) != len(optimized_records):
        raise ValueError("Baseline and optimized captures have different record counts")

    pairs: list[dict[str, Any]] = []
    exact = True
    for base, candidate in zip(baseline_records, optimized_records):
        identity = (
            base["case"]["name"],
            base["run_index"],
        )
        candidate_identity = (
            candidate["case"]["name"],
            candidate["run_index"],
        )
        if identity != candidate_identity:
            raise ValueError(f"Record ordering mismatch: {identity!r} != {candidate_identity!r}")

        fields = {
            "speech_tokens": base["speech_token_sha256"]
            == candidate["speech_token_sha256"],
            "waveform_float32": base["waveform_float32_sha256"]
            == candidate["waveform_float32_sha256"],
            "pcm16": base["pcm16_sha256"] == candidate["pcm16_sha256"],
            "sample_rate": base["sample_rate"] == candidate["sample_rate"],
            "sample_count": base["sample_count"] == candidate["sample_count"],
        }
        pair_exact = all(fields.values())
        exact = exact and pair_exact
        pairs.append(
            {
                "case": identity[0],
                "run_index": identity[1],
                "exact": pair_exact,
                "fields": fields,
                "baseline_generation_ms": base["generation_ms"],
                "optimized_generation_ms": candidate["generation_ms"],
            }
        )

    baseline_median = _median(baseline_records, "generation_ms")
    optimized_median = _median(optimized_records, "generation_ms")
    speedup = None
    if baseline_median is not None and optimized_median and optimized_median > 0:
        speedup = baseline_median / optimized_median

    return {
        "exact": exact,
        "summary": {
            "baseline_median_generation_ms": baseline_median,
            "optimized_median_generation_ms": optimized_median,
            "median_speedup_x": speedup,
            "record_count": len(pairs),
        },
        "pairs": pairs,
    }


def _deterministic_capture_env(*, performance_enabled: bool) -> dict[str, str]:
    return {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TURBO_PARITY_DETERMINISTIC": "1",
        "TURBO_PERFORMANCE_RUNTIME": "1" if performance_enabled else "0",
    }


def compare(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline.json"
    baseline_repeat_path = args.output_dir / "baseline-repeat.json"
    optimized_path = args.output_dir / "optimized.json"
    report_path = args.output_dir / "comparison.json"

    base_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "capture",
        "--cases",
        str(args.cases.resolve()),
        "--runs",
        str(args.runs),
    ]
    subprocess.run(
        base_command + ["--mode", "baseline", "--output", str(baseline_path)],
        check=True,
        env=_deterministic_capture_env(performance_enabled=False),
    )
    subprocess.run(
        base_command + ["--mode", "baseline", "--output", str(baseline_repeat_path)],
        check=True,
        env=_deterministic_capture_env(performance_enabled=False),
    )
    subprocess.run(
        base_command + ["--mode", "optimized", "--output", str(optimized_path)],
        check=True,
        env=_deterministic_capture_env(performance_enabled=True),
    )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_repeat = json.loads(baseline_repeat_path.read_text(encoding="utf-8"))
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    comparison = _build_comparison(baseline, optimized)
    baseline_reproducibility = _build_comparison(baseline, baseline_repeat)
    comparison["baseline_reproducibility"] = baseline_reproducibility
    report_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(report_path)

    if not baseline_reproducibility["exact"] and not args.allow_drift:
        print("ERROR: deterministic baseline is not reproducible across isolated processes", file=sys.stderr)
        return 3
    if not comparison["exact"] and not args.allow_drift:
        print("ERROR: exact token/waveform parity failed", file=sys.stderr)
        return 2
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    capture_parser.add_argument("--cases", type=Path, required=True)
    capture_parser.add_argument("--runs", type=int, default=3)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.set_defaults(handler=capture)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cases", type=Path, required=True)
    compare_parser.add_argument("--runs", type=int, default=3)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Write the report without failing when exact parity differs.",
    )
    compare_parser.set_defaults(handler=compare)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
