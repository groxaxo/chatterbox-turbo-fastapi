#!/usr/bin/env python3
"""Strict local PyTorch-versus-TensorRT acceleration parity harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from benchmarks.benchmark_parity import _build_comparison
except ModuleNotFoundError:  # Direct execution: python benchmarks/benchmark_acceleration.py
    from benchmark_parity import _build_comparison


def _capture_env(accelerator: str) -> dict[str, str]:
    return {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "TURBO_PARITY_DETERMINISTIC": "1",
        "TURBO_PERFORMANCE_RUNTIME": "1",
        "TURBO_ACCELERATOR": accelerator,
        "TURBO_ACCELERATOR_FAIL_CLOSED": "1" if accelerator != "torch" else "0",
    }


def _capture(
    *,
    cases: Path,
    runs: int,
    output: Path,
    accelerator: str,
) -> None:
    benchmark = Path(__file__).with_name("benchmark_parity.py")
    command = [
        sys.executable,
        str(benchmark),
        "capture",
        "--mode",
        "optimized",
        "--cases",
        str(cases),
        "--runs",
        str(runs),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, env=_capture_env(accelerator))


def compare(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "torch-baseline.json"
    repeat_path = args.output_dir / "torch-baseline-repeat.json"
    candidate_path = args.output_dir / f"{args.candidate}-candidate.json"
    report_path = args.output_dir / "acceleration-comparison.json"

    _capture(
        cases=args.cases,
        runs=args.runs,
        output=baseline_path,
        accelerator="torch",
    )
    _capture(
        cases=args.cases,
        runs=args.runs,
        output=repeat_path,
        accelerator="torch",
    )
    _capture(
        cases=args.cases,
        runs=args.runs,
        output=candidate_path,
        accelerator=args.candidate,
    )

    baseline: dict[str, Any] = json.loads(baseline_path.read_text(encoding="utf-8"))
    repeat: dict[str, Any] = json.loads(repeat_path.read_text(encoding="utf-8"))
    candidate: dict[str, Any] = json.loads(candidate_path.read_text(encoding="utf-8"))
    report = _build_comparison(baseline, candidate)
    report["baseline_reproducibility"] = _build_comparison(baseline, repeat)
    report["candidate_accelerator"] = args.candidate
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(report_path)

    if not report["baseline_reproducibility"]["exact"]:
        print("ERROR: deterministic PyTorch baseline is not reproducible", file=sys.stderr)
        return 3
    if not report["exact"] and not args.allow_drift:
        print("ERROR: candidate changed tokens or waveform bytes", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cases", type=Path, required=True)
    compare_parser.add_argument("--runs", type=int, default=3)
    compare_parser.add_argument("--candidate", choices=("tensorrt",), default="tensorrt")
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--allow-drift", action="store_true")
    compare_parser.set_defaults(handler=compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
