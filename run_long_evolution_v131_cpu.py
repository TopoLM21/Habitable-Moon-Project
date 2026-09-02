#!/usr/bin/env python3
"""Experimental CPU-optimized v0.31; the original runner stays available."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--render-workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--cell-kernels", action="store_true", help="Use exact-order batched sediment routing")
    options, remaining = parser.parse_known_args()
    if not 1 <= options.cpu_workers <= 32:
        parser.error("--cpu-workers must be between 1 and 32")
    # Keep the same BLAS settings as the reference: changing its reduction
    # threading can change floating-point rounding in the material ledger.
    # Nested cKDTree query threads are controlled separately by CpuExecution.
    from tectonics.cpu_runtime import CpuExecution
    from visualization.render_runtime import RenderExecution

    sys.argv = ["run_long_evolution_v131.py", *remaining]
    print(f"Experimental CPU mode: cached geometry, {options.cpu_workers} plate worker(s)", flush=True)
    with CpuExecution(options.cpu_workers, cell_kernels=options.cell_kernels), RenderExecution(options.render_workers) as rendering:
        import run_long_evolution_v131 as runner
        rendering.install_runner_hooks()
        print(f"Render mode: {options.render_workers} process(es)", flush=True)
        runner.main()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output")
    output_options, _ = output_parser.parse_known_args(remaining)
    if output_options.output:
        report = Path(output_options.output) / "render_timings.json"
        report.write_text(json.dumps(rendering.report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
