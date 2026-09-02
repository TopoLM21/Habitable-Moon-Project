"""Serial, reproducible wall-clock comparisons on an existing checkpoint.

Each variant runs in a fresh process and a new results directory. Input
checkpoints are read only. Default jobs match the GUI's 20-Myr full-map segment.
Do not compare simultaneous runs: they would compete for the same CPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def compare_checkpoints(reference: Path, actual: Path) -> dict:
    differences = []
    array_count = 0
    with np.load(reference / "state.npz", allow_pickle=False) as a, np.load(actual / "state.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            differences.append("array names")
        for name in sorted(set(a.files) & set(b.files)):
            x, y = a[name], b[name]
            array_count += 1
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                differences.append(name)
    # Compare all model metadata and accumulated diagnostic history, not just
    # the visible surface or the last row. File whitespace is not model state.
    a_meta = json.loads((reference / "meta.json").read_text(encoding="utf-8"))
    b_meta = json.loads((actual / "meta.json").read_text(encoding="utf-8"))
    if a_meta != b_meta:
        differences.append("metadata/history")
    return {"exact": not differences, "arrays_compared": array_count, "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--span", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["baseline", "cpu1", "cpu2", "cpu4", "cpu8"],
                        choices=["baseline", "cpu1", "cpu2", "cpu4", "cpu8"])
    parser.add_argument("--reference", type=Path, help="An existing independent endpoint checkpoint to compare with")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--paired-baseline", action="store_true", help="Measure a fresh baseline immediately before each optimized case")
    args = parser.parse_args()
    if args.repeat < 1 or args.span <= 0 or args.dt <= 0:
        parser.error("repeat, span and dt must be positive")
    if args.paired_baseline and all(mode == "baseline" for mode in args.modes):
        parser.error("paired comparison needs at least one optimized mode")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = float(json.loads((args.resume / "meta.json").read_text(encoding="utf-8"))["time_myr"])
    reference = args.reference.resolve() if args.reference else None
    rows = []
    for repeat in range(args.repeat):
        # Alternate order to reduce first-run/cache/order bias.
        modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
        if args.paired_baseline:
            modes = [item for mode in modes if mode != "baseline" for item in ("baseline", mode)]
        for case_index, mode in enumerate(modes):
            case = output / f"{mode}_{repeat + 1}_{case_index}"
            case.mkdir()
            checkpoint = case / "checkpoint"
            runner = "run_long_evolution_v131.py" if mode == "baseline" else "run_long_evolution_v131_cpu.py"
            command = [sys.executable]
            if args.profile:
                command += ["-m", "cProfile", "-o", str(case / "profile.prof")]
            command += [str(ROOT / runner), "--config", str(args.config.resolve()),
                        "--resume", str(args.resume.resolve()), "--end-time", f"{start + args.span:g}",
                        "--dt", f"{args.dt:g}", "--frame-interval", f"{args.span:g}",
                        "--output", str(case), "--checkpoint", str(checkpoint)]
            if mode != "baseline":
                command += ["--cpu-workers", mode.removeprefix("cpu")]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT)
            environment["PYTHONUNBUFFERED"] = "1"
            environment["MPLBACKEND"] = "Agg"
            print(f"START {mode} repeat {repeat + 1}: {start:g}->{start + args.span:g} Myr", flush=True)
            began = perf_counter()
            with (case / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            duration = perf_counter() - began
            if completed.returncode != 0:
                raise RuntimeError(f"{mode} failed; see {case / 'run.log'}")
            if reference is None:
                reference = checkpoint
            comparison = compare_checkpoints(reference, checkpoint)
            row = {"mode": mode, "repeat": repeat + 1, "case_index": case_index, "wall_seconds": duration, **comparison}
            rows.append(row)
            (output / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"DONE {mode}: {duration:.3f}s; exact={comparison['exact']}; {comparison['differences']}", flush=True)
            if not comparison["exact"]:
                raise RuntimeError("CPU variant changed scientific state; stopping benchmark")
    summary = [{"mode": mode, "median_seconds": statistics.median(row["wall_seconds"] for row in rows if row["mode"] == mode),
                "runs": sum(row["mode"] == mode for row in rows)} for mode in dict.fromkeys(row["mode"] for row in rows)]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "median_seconds", "runs"])
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
