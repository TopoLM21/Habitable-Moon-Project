#!/usr/bin/env python3
"""Experimental deterministic CUDA slice over the optimized CPU runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental CUDA slice. All unrecognised options are forwarded to "
            "run_long_evolution_v131_cpu.py."
        )
    )
    parser.add_argument("--gpu-device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument(
        "--gpu-arcs",
        action="store_true",
        help="Enable tolerance-based CUDA volcanic-arc painting (not byte-exact)",
    )
    parser.add_argument(
        "--gpu-surface",
        action="store_true",
        help="Enable resident CUDA erosion, sediment and relief pipeline",
    )
    parser.add_argument("--assignment-columns", action=argparse.BooleanOptionalAction, default=False,
                        help="Compact unused SciPy columns when --no-assignment-optimized is selected")
    parser.add_argument("--assignment-optimized", action=argparse.BooleanOptionalAction, default=True,
                        help="Use certified sparse assignment without the cardinality precheck (default: on)")
    parser.add_argument("--boundary-forces", action=argparse.BooleanOptionalAction, default=False,
                        help="Use cached geometry and exact-order CPU boundary forces (opt-in)")
    options, remaining = parser.parse_known_args()
    if options.gpu_device < 0:
        parser.error("--gpu-device must be non-negative")

    from tectonics.gpu_runtime import GpuExecution
    import run_long_evolution_v131_cpu as cpu_runner

    sys.argv = ["run_long_evolution_v131_cpu.py",
                "--assignment-columns" if options.assignment_columns else "--no-assignment-columns",
                "--assignment-optimized" if options.assignment_optimized else "--no-assignment-optimized",
                "--boundary-forces" if options.boundary_forces else "--no-boundary-forces", *remaining]
    with GpuExecution(options.gpu_device, arc_painting=options.gpu_arcs,
                      surface_pipeline=options.gpu_surface) as gpu:
        print(
            f"CUDA mode: {gpu.device_name}; FP64 deterministic sediment routing; "
            f"volcanic arcs: {'CUDA tolerance mode' if options.gpu_arcs else 'optimized CPU'}; "
            f"surface pipeline: {'resident CUDA' if options.gpu_surface else 'optimized CPU'}; "
            "all other physics remains on optimized CPU",
            flush=True,
        )
        cpu_runner.main()
        gpu_report = gpu.report()

    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output")
    output_options, _ = output_parser.parse_known_args(remaining)
    if output_options.output:
        report_path = Path(output_options.output) / "render_timings.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report["gpu_execution"] = gpu_report
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
