"""Interleaved full-segment rendering comparison; no concurrent model cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.benchmark_cpu_modes import compare_checkpoints
from analysis.windows_cpu_affinity import performance_core_benchmark
from execution_policy import RENDER_WORKER_CHOICES, PROCESS_PRIORITY_CHOICES, is_lower_priority


def compare_pngs(reference, actual):
    expected = {p.relative_to(reference) for p in reference.rglob("*.png")}
    found = {p.relative_to(actual) for p in actual.rglob("*.png")}
    differences = [str(p) for p in expected ^ found]
    for path in sorted(expected & found):
        if hashlib.sha256((reference / path).read_bytes()).digest() != hashlib.sha256((actual / path).read_bytes()).digest():
            differences.append(str(path))
    return {"png_exact": not differences, "png_count": len(expected), "png_differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--modes", type=int, nargs="+", choices=RENDER_WORKER_CHOICES, default=[1, 2, 4])
    parser.add_argument("--priorities", nargs="+", choices=PROCESS_PRIORITY_CHOICES, default=["normal"])
    parser.add_argument("--cell-kernels", action="store_true")
    parser.add_argument("--performance-cores", action="store_true")
    parser.add_argument("--compare-cell-kernels", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.cell_kernels and args.compare_cell_kernels:
        parser.error("Choose --cell-kernels or --compare-cell-kernels, not both")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "results/cpu_performance"):
        parser.error("Benchmark output must be in the experimental performance folder")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results/gui_runs/stable_copy_0_700"
    reference = ROOT / "results/cpu_performance/baseline_full"
    rows = []
    with performance_core_benchmark(args.performance_cores) as affinity:
        (output / "cpu_environment.json").write_text(json.dumps(affinity, indent=2))
        for repeat in range(args.repeat):
            modes = [(workers, cells, priority) for workers in args.modes
                     for cells in ([False, True] if args.compare_cell_kernels else [args.cell_kernels])
                     for priority in args.priorities]
            if repeat % 2:
                modes.reverse()
            for workers, cells, priority in modes:
                suffix = f"_cells{int(cells)}" if args.compare_cell_kernels else ""
                case = output / f"render{workers}_{repeat + 1}{suffix}_{priority}"
                case.mkdir()
                command = [sys.executable, str(ROOT / "run_long_evolution_v131_cpu.py"),
                           "--cpu-workers", "1", "--render-workers", str(workers),
                           "--process-priority", priority,
                           "--config", str(source / "gui_runtime_config.yaml"),
                           "--resume", str(source / "gui_checkpoint_000700_Myr"),
                           "--end-time", "720", "--dt", "4", "--frame-interval", "20",
                           "--output", str(case), "--checkpoint", str(case / "checkpoint")]
                if cells:
                    command.append("--cell-kernels")
                env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
                print(f"START render{workers} cells={cells} priority={priority} repeat {repeat + 1}", flush=True)
                began = perf_counter()
                with (case / "run.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                wall = perf_counter() - began
                comparison = compare_checkpoints(reference / "checkpoint_720", case / "checkpoint")
                images = compare_pngs(reference, case)
                report = json.loads((case / "render_timings.json").read_text())
                priority_verified = priority == "normal" or all(is_lower_priority(value) for value in
                    [report["coordinator_priority"], *[job["priority"] for job in report["jobs"]]])
                row = {"render_workers": workers, "cell_kernels": cells, "process_priority": priority,
                       "priority_verified": priority_verified, "repeat": repeat + 1, "wall_seconds": wall,
                       **comparison, **images, "render_report": report}
                rows.append(row)
                (output / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                print(f"DONE render{workers}: {wall:.3f}s; state={comparison['exact']}; PNG={images['png_exact']}", flush=True)
                if not comparison["exact"] or not images["png_exact"]:
                    raise RuntimeError("Rendering changed the reference result")
                if not priority_verified:
                    raise RuntimeError("Requested lower priority was not applied to every process")


if __name__ == "__main__":
    main()
