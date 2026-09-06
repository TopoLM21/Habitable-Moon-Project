"""Validate exact GPU surface continuations and measure complete child processes.

CPU and GPU cases run sequentially in fresh processes, with output confined to a
new directory beneath results/gpu_surface. CUDA caches and optional PYTHONPATH
entries are inherited. GPU telemetry is diagnostic context, not evidence that
the GPU was idle throughout a measurement. Even without --frames the production
runner writes final reports; elapsed times are not physics-only timings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_cpu_modes import compare_checkpoints

RESULTS = ROOT / "results" / "gpu_surface"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def telemetry(device: int) -> dict:
    command = [
        "nvidia-smi", "--id", str(device),
        "--query-gpu=index,name,uuid,driver_version,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,nounits",
    ]
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(), "command": command}
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def child_environment(output: Path, project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(project_root) + (os.pathsep + inherited_path if inherited_path else "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    # Keep the font cache local unless the caller has explicitly chosen a cache.
    environment.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    return environment


def run_case(args, output: Path, row: dict,
             resume: Path, end_time: float) -> None:
    case = output / row["case"]
    case.mkdir(exist_ok=False)
    gpu = row["backend"] == "gpu_surface"
    project_root = ROOT if gpu else args.cpu_project_root
    runner = project_root / ("run_long_evolution_v131_gpu.py" if gpu else "run_long_evolution_v131_cpu.py")
    environment = child_environment(output, project_root)
    command = [str(args.gpu_python if gpu else args.cpu_python), str(runner)]
    if gpu:
        command += ["--gpu-device", str(args.gpu_device), "--gpu-surface"]
    command += [
        "--cpu-workers", str(args.cpu_workers), "--render-workers", str(args.render_workers),
        "--cell-kernels", "--config", str(args.config), "--resume", str(resume),
        "--end-time", str(end_time), "--dt", str(args.dt),
        "--output", str(case), "--checkpoint", str(case / "checkpoint"),
    ]
    if args.frames:
        command += ["--frame-interval", "20"]
    row.update(command=command, project_root=str(project_root), pythonpath=environment["PYTHONPATH"],
               resume=str(resume), end_time_myr=end_time,
               checkpoint=str(case / "checkpoint"), log=str(case / "run.log"),
               telemetry_before=telemetry(args.gpu_device))
    print(f"START {row['case']}: target {end_time:g} Myr", flush=True)
    with (case / "run.log").open("x", encoding="utf-8") as log:
        started = perf_counter()
        try:
            completed = subprocess.run(command, cwd=project_root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            row["returncode"] = completed.returncode
        except BaseException as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["wall_seconds"] = perf_counter() - started
            row["telemetry_after"] = telemetry(args.gpu_device)
    timings = case / "render_timings.json"
    if timings.exists():
        row["execution_report"] = json.loads(timings.read_text(encoding="utf-8"))
    if completed.returncode:
        raise RuntimeError(f"{row['case']} exited {completed.returncode}; see {case / 'run.log'}")
    if gpu:
        gpu_report = row.get("execution_report", {}).get("gpu_execution", {})
        require_surface_execution(gpu_report)
    actual_end = float(json.loads((case / "checkpoint" / "meta.json").read_text(encoding="utf-8"))["time_myr"])
    if not math.isclose(actual_end, end_time, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError(f"{row['case']} stopped at {actual_end:g}, expected {end_time:g} Myr")
    print(f"DONE {row['case']}: {row['wall_seconds']:.3f} s", flush=True)


def require_surface_execution(gpu_report: dict) -> None:
    if (gpu_report.get("backend") != "cuda" or gpu_report.get("surface_pipeline_enabled") is not True
            or (gpu_report.get("surface_pipeline") or {}).get("calls", 0) <= 0):
        raise RuntimeError("GPU candidate did not report an enabled CUDA surface pipeline with positive call count")


def write_summary(output: Path, report: dict) -> None:
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def require_exact(report: dict, name: str, reference: Path, actual: Path) -> None:
    comparison = {"case": name, "reference": str(reference), "actual": str(actual),
                  **compare_checkpoints(reference, actual)}
    report["comparisons"].append(comparison)
    print(f"CHECK {name}: exact={comparison['exact']}; {comparison['differences']}", flush=True)
    if not comparison["exact"]:
        raise RuntimeError(f"GPU candidate failed exact validation: {name}; {comparison['differences']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-python", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path, required=True)
    parser.add_argument("--cpu-project-root", type=Path, default=ROOT,
                        help="Independent CPU reference checkout (default: this checkout)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=1, choices=range(1, 33))
    parser.add_argument("--render-workers", type=int, default=4, choices=(1, 2, 4, 6, 8, 12))
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--frames", action="store_true", help="Also request periodic frames every 20 Myr")
    parser.add_argument("--midpoint", type=float,
                        help="Also run a GPU prefix then resume its checkpoint on CPU to the endpoint")
    parser.add_argument("--output", type=Path,
                        help="New directory beneath results/gpu_surface (default: a unique timestamped directory)")
    args = parser.parse_args(argv)
    args.cpu_project_root = args.cpu_project_root.resolve()
    if not (args.cpu_project_root / "run_long_evolution_v131_cpu.py").is_file():
        parser.error(f"CPU project root has no optimized runner: {args.cpu_project_root}")
    for name in ("cpu_python", "gpu_python", "config"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} is not a file: {path}")
        setattr(args, name, path)
    args.resume = args.resume.resolve()
    for name in ("state.npz", "meta.json"):
        if not (args.resume / name).is_file():
            parser.error(f"input checkpoint is missing {name}: {args.resume}")
    try:
        start = float(json.loads((args.resume / "meta.json").read_text(encoding="utf-8"))["time_myr"])
    except (ValueError, KeyError) as exc:
        parser.error(f"invalid checkpoint time: {exc}")
    if (args.repeat < 1 or args.gpu_device < 0 or not math.isfinite(args.dt) or args.dt <= 0
            or not math.isfinite(args.end_time) or not math.isfinite(start) or args.end_time <= start):
        parser.error("repeat/dt must be positive, GPU index non-negative, and finite end-time later than checkpoint")
    intervals = [args.end_time - start]
    if args.midpoint is not None:
        if not math.isfinite(args.midpoint) or not start < args.midpoint < args.end_time:
            parser.error("midpoint must be between checkpoint time and end-time")
        intervals += [args.midpoint - start, args.end_time - args.midpoint]
    if args.frames:
        intervals.append(20.0)
    if any(not math.isclose(span / args.dt, round(span / args.dt), rel_tol=0.0, abs_tol=1.0e-9)
           for span in intervals):
        parser.error("simulation intervals and optional 20-Myr frame interval must align to dt")
    args.start_time = start
    if args.output:
        args.output = args.output.resolve()
        results = RESULTS.resolve()
        if args.output == results or not args.output.is_relative_to(results):
            parser.error(f"--output must be a new directory beneath {results}")
        if args.output.exists():
            parser.error(f"output already exists: {args.output}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.output:
        output = args.output
        output.mkdir(parents=True, exist_ok=False)
    else:
        prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=RESULTS))
    environment = child_environment(output, ROOT)
    report = {
        "status": "running", "output": str(output), "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"config": str(args.config), "checkpoint": str(args.resume),
                   "start_time_myr": args.start_time, "end_time_myr": args.end_time,
                   "dt_myr": args.dt, "midpoint_myr": args.midpoint},
        "settings": {"repeat": args.repeat, "cpu_workers": args.cpu_workers,
                     "render_workers": args.render_workers, "periodic_frames": args.frames,
                     "gpu_device": args.gpu_device, "cpu_project_root": str(args.cpu_project_root)},
        "environment": {name: environment.get(name) for name in (
            "PYTHONPATH", "CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "CUDA_VISIBLE_DEVICES",
            "MPLCONFIGDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "measurement_notes": [
            "Full subprocess wall times include imports, CUDA initialization, checkpoint I/O and final report rendering.",
            "CPU/GPU order alternates across repeats; first process use may populate inherited caches.",
            "Background load is uncontrolled. Before/after GPU snapshots do not establish isolation during a run.",
            "Separate Python environments may differ in dependencies; their executable paths are recorded in commands.",
            "Exact array bytes and complete metadata/history are required; no tolerance fallback is used.",
        ],
        "runs": [], "comparisons": [],
    }
    failure = None
    try:
        report["input_checkpoint_sha256_before"] = checkpoint_hashes(args.resume)
        report["input_config_sha256_before"] = sha256_file(args.config)
        write_summary(output, report)
        first_cpu = None
        for repeat in range(1, args.repeat + 1):
            pair = {}
            order = ("cpu", "gpu_surface") if repeat % 2 else ("gpu_surface", "cpu")
            for backend in order:
                row = {"case": f"{backend}_{repeat}", "backend": backend, "repeat": repeat,
                       "purpose": "full_segment"}
                report["runs"].append(row)
                write_summary(output, report)
                run_case(args, output, row, args.resume, args.end_time)
                pair[backend] = Path(row["checkpoint"])
                write_summary(output, report)
            require_exact(report, f"pair_{repeat}", pair["cpu"], pair["gpu_surface"])
            if first_cpu is None:
                first_cpu = pair["cpu"]
            else:
                require_exact(report, f"cpu_repeat_{repeat}", first_cpu, pair["cpu"])
            write_summary(output, report)
        if args.midpoint is not None:
            prefix = {"case": "gpu_midpoint", "backend": "gpu_surface", "purpose": "resume_validation"}
            report["runs"].append(prefix)
            run_case(args, output, prefix, args.resume, args.midpoint)
            write_summary(output, report)
            midpoint = Path(prefix["checkpoint"])
            report["midpoint_sha256_before_resume"] = checkpoint_hashes(midpoint)
            resumed = {"case": "cpu_from_gpu_midpoint", "backend": "cpu", "purpose": "resume_validation"}
            report["runs"].append(resumed)
            run_case(args, output, resumed, midpoint, args.end_time)
            report["midpoint_sha256_after_resume"] = checkpoint_hashes(midpoint)
            if report["midpoint_sha256_before_resume"] != report["midpoint_sha256_after_resume"]:
                raise RuntimeError("CPU continuation modified the GPU midpoint checkpoint")
            require_exact(report, "cpu_resume_from_gpu", first_cpu, Path(resumed["checkpoint"]))
        report["status"] = "exact_validation_passed"
    except (Exception, KeyboardInterrupt) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        report["status"] = "candidate_failed"
        report["error"] = failure
    finally:
        try:
            report["input_checkpoint_sha256_after"] = checkpoint_hashes(args.resume)
            report["input_config_sha256_after"] = sha256_file(args.config)
            unchanged = (report.get("input_checkpoint_sha256_before") == report["input_checkpoint_sha256_after"]
                         and report.get("input_config_sha256_before") == report["input_config_sha256_after"])
            report["input_unchanged"] = unchanged
            if not unchanged:
                failure = "Input checkpoint or configuration changed during validation"
                report.update(status="candidate_failed", input_integrity_error=failure)
        except OSError as exc:
            failure = f"Cannot verify input integrity: {exc}"
            report.update(status="candidate_failed", input_integrity_error=failure)
        if report["status"] == "exact_validation_passed":
            medians = {
                mode: statistics.median(row["wall_seconds"] for row in report["runs"]
                                        if row["purpose"] == "full_segment" and row["backend"] == mode)
                for mode in ("cpu", "gpu_surface")
            }
            report["observed_median_wall_seconds"] = medians
            report["observed_cpu_over_gpu_wall_ratio"] = medians["cpu"] / medians["gpu_surface"]
        write_summary(output, report)
    print(f"{report['status']}: {output / 'summary.json'}", flush=True)
    if failure:
        print(failure, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
