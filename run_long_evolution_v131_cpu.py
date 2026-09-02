#!/usr/bin/env python3
"""Experimental CPU-optimized v0.31; the original runner stays available."""
from __future__ import annotations

import argparse
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-workers", type=int, default=1)
    options, remaining = parser.parse_known_args()
    if not 1 <= options.cpu_workers <= 32:
        parser.error("--cpu-workers must be between 1 and 32")
    # Keep the same BLAS settings as the reference: changing its reduction
    # threading can change floating-point rounding in the material ledger.
    # Nested cKDTree query threads are controlled separately by CpuExecution.
    from tectonics.cpu_runtime import CpuExecution

    sys.argv = ["run_long_evolution_v131.py", *remaining]
    print(f"Experimental CPU mode: cached geometry, {options.cpu_workers} plate worker(s)", flush=True)
    with CpuExecution(options.cpu_workers):
        runpy.run_module("run_long_evolution_v131", run_name="__main__")


if __name__ == "__main__":
    main()
