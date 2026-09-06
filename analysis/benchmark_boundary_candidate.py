"""Sequential whole-process comparison against the existing GPU surface mode.

Every case is also compared byte-exactly against an independent CPU checkpoint.
The candidate remains an analysis-only, prepared-CPU boundary-loop replacement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_cpu_modes import compare_checkpoints
from analysis.validate_gpu_surface import (
    checkpoint_hashes, child_environment, require_surface_execution, sha256_file, telemetry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    for name in ("config", "resume", "reference", "output"):
        setattr(args, name, getattr(args, name).resolve())
    results = (ROOT / "results" / "gpu_surface").resolve()
    if args.output == results or not args.output.is_relative_to(results) or args.output.exists():
        parser.error("output must be a new directory beneath results/gpu_surface")
    if args.repeat < 1 or args.gpu_device < 0 or not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("invalid repeats, device or timestep")
    start = float(json.loads((args.resume / "meta.json").read_text(encoding="utf-8"))["time_myr"])
    reference_time = float(json.loads((args.reference / "meta.json").read_text(encoding="utf-8"))["time_myr"])
    steps = (args.end_time - start) / args.dt
    if (not math.isfinite(steps) or steps <= 0 or abs(steps - round(steps)) > 1e-9
            or not math.isclose(args.end_time, reference_time, rel_tol=0., abs_tol=1e-9)):
        parser.error("reference/end-time mismatch or nonpositive/unaligned simulation interval")
    if args.frames and abs(20 / args.dt - round(20 / args.dt)) > 1e-9:
        parser.error("frame interval must align to timestep")
    code_paths = [ROOT / name for name in (
        "analysis/run_boundary_candidate.py", "analysis/probe_gpu_boundary_forces.py",
        "tectonics/dynamics.py", "tectonics/gpu_runtime.py", "tectonics/gpu_surface.py",
        "tectonics/sediment.py", "run_long_evolution_v131_gpu.py")]

    def fingerprints():
        return {"config": sha256_file(args.config), "input": checkpoint_hashes(args.resume),
                "reference": checkpoint_hashes(args.reference),
                "candidate_sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths}}

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "start_myr": start, "end_myr": args.end_time,
        "steps": int(round(steps)), "frames": args.frames,
        "reference": str(args.reference), "python": sys.executable,
        "notes": ["Fresh child processes; complete wall time includes CUDA setup, I/O and final rendering.",
                  "Candidate is prepared CPU boundary forces, not a GPU boundary speedup.",
                  "Orders alternate; cache and background activity are not fully controlled.",
                  "All inputs, candidate code and independent reference are hash checked."],
        "fingerprints_before": fingerprints(), "runs": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    failure = None
    save()
    try:
        for repeat in range(1, args.repeat + 1):
            order = ["gpu_surface", "boundary_candidate"]
            if repeat % 2 == 0:
                order.reverse()
            for mode in order:
                case = args.output / f"{mode}_{repeat}"
                case.mkdir()
                runner = ROOT / ("run_long_evolution_v131_gpu.py" if mode == "gpu_surface"
                                 else "analysis/run_boundary_candidate.py")
                command = [sys.executable, str(runner), "--gpu-surface", "--gpu-device", str(args.gpu_device),
                           "--cpu-workers", "1", "--render-workers", "4", "--cell-kernels",
                           "--config", str(args.config), "--resume", str(args.resume),
                           "--end-time", str(args.end_time), "--dt", str(args.dt),
                           "--output", str(case), "--checkpoint", str(case / "checkpoint")]
                if args.frames:
                    command += ["--frame-interval", "20"]
                row = {"mode": mode, "repeat": repeat, "command": command,
                       "telemetry_before": telemetry(args.gpu_device)}
                report["runs"].append(row)
                save()
                print(f"START {case.name}", flush=True)
                with (case / "run.log").open("x", encoding="utf-8") as log:
                    began = perf_counter()
                    result = subprocess.run(command, cwd=ROOT, env=child_environment(args.output, ROOT),
                                            stdout=log, stderr=subprocess.STDOUT)
                    row["wall_seconds"] = perf_counter() - began
                row["returncode"] = result.returncode
                row["telemetry_after"] = telemetry(args.gpu_device)
                if result.returncode:
                    raise RuntimeError(f"{case.name} failed; see {case / 'run.log'}")
                data = json.loads((case / "render_timings.json").read_text(encoding="utf-8"))
                row["execution_report"] = data
                require_surface_execution(data.get("gpu_execution", {}))
                if data["gpu_execution"]["surface_pipeline"].get("calls") != int(round(steps)):
                    raise RuntimeError("GPU surface execution did not cover every expected step")
                if mode == "boundary_candidate":
                    candidate = data.get("boundary_candidate", {})
                    if (candidate.get("backend") != "prepared_cpu_boundary_forces"
                            or candidate.get("calls") != int(round(steps))):
                        raise RuntimeError("Candidate boundary execution provenance is missing or incomplete")
                row["comparison"] = compare_checkpoints(args.reference, case / "checkpoint")
                if not row["comparison"]["exact"]:
                    raise RuntimeError(f"Candidate changed trajectory: {row['comparison']}")
                print(f"DONE {case.name}: {row['wall_seconds']:.3f}s; exact=True", flush=True)
                save()
        report["status"] = "exact_validation_passed"
    except (Exception, KeyboardInterrupt) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        report.update(status="failed", error=failure)
    finally:
        try:
            report["fingerprints_after"] = fingerprints()
            report["inputs_and_sources_unchanged"] = report["fingerprints_before"] == report["fingerprints_after"]
            if not report["inputs_and_sources_unchanged"]:
                failure = "Input, reference or candidate sources changed during validation"
                report.update(status="failed", error=failure)
        except Exception as exc:
            failure = f"Final fingerprint verification failed: {type(exc).__name__}: {exc}"
            report["inputs_and_sources_unchanged"] = False
            report.update(status="failed", error=failure)
        if report["status"] == "exact_validation_passed":
            medians = {mode: statistics.median(row["wall_seconds"] for row in report["runs"] if row["mode"] == mode)
                       for mode in ("gpu_surface", "boundary_candidate")}
            report["median_wall_seconds"] = medians
            report["observed_time_reduction_percent"] = 100 * (1 - medians["boundary_candidate"] / medians["gpu_surface"])
        save()
    print(f"{report['status']}: {args.output / 'summary.json'}", flush=True)
    if failure:
        print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
