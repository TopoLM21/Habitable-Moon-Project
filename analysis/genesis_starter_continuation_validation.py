"""Bounded mature continuations with independent restart and step comparisons.

Every case invokes the public CLI in a separate process. This module never
imports the mature runner or installs its hooks in the validation process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results" / "genesis_runs" / "starter_validation_20260927"
CASES = (
    {"name": "strong_5120_10myr_step1", "source": "strong_50kpa_5120_step1", "duration": 10., "step": 1.},
    {"name": "strong_5120_10myr_step05", "source": "strong_50kpa_5120_step1", "duration": 10., "step": .5},
    {"name": "whole_1280_3myr_step1", "source": "strong_50kpa_1280_step1", "duration": 3., "step": 1.},
    {"name": "first_1280_1myr_step1", "source": "strong_50kpa_1280_step1", "duration": 1., "step": 1.},
    {"name": "resumed_1280_total3myr_step1", "resume": "first_1280_1myr_step1", "duration": 3., "step": 1.},
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def code_hashes():
    paths = set(ROOT.glob("tectonics/*.py")) | set(ROOT.glob("visualization/*.py"))
    paths.update(ROOT.glob("run_long_evolution_v*.py"))
    paths.update((ROOT / "run_genesis_starter_continuation.py", ROOT / "execution_policy.py", Path(__file__).resolve()))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in sorted(paths)}


def differences(a, b, path="", limit=80):
    """List exact JSON differences without discarding histories or counters."""
    out = []

    def visit(x, y, label):
        if len(out) >= limit:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if x.keys() != y.keys():
                out.append({"path": label, "only_left": sorted(x.keys() - y.keys()), "only_right": sorted(y.keys() - x.keys())})
            for key in sorted(x.keys() & y.keys()):
                visit(x[key], y[key], f"{label}.{key}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append({"path": label, "left_length": len(x), "right_length": len(y)})
            for index, (xx, yy) in enumerate(zip(x, y)):
                visit(xx, yy, f"{label}[{index}]")
        elif x != y or type(x) is not type(y):
            out.append({"path": label, "left": x, "right": y})

    visit(a, b, path)
    return out


def compare_npz(left, right):
    result = {"all_exact": True, "fields": {}}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        result["same_keys"] = set(a.files) == set(b.files)
        result["all_exact"] &= result["same_keys"]
        result["only_left"] = sorted(set(a.files) - set(b.files))
        result["only_right"] = sorted(set(b.files) - set(a.files))
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            same = bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))
            field = {"exact": same, "left_shape": list(x.shape), "right_shape": list(y.shape)}
            if key == "metadata":
                try:
                    field["json_differences"] = differences(json.loads(str(x)), json.loads(str(y)), "metadata")
                except (ValueError, TypeError):
                    field["json_differences"] = None
            elif x.shape == y.shape and x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
                delta = y.astype(float) - x.astype(float)
                field["max_abs_difference"] = float(np.max(np.abs(delta))) if delta.size else 0.
                field["rms_difference"] = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.
                scale = float(np.max(np.abs(x))) if x.size else 0.
                field["relative_to_left_max"] = field["max_abs_difference"] / scale if scale else None
            result["fields"][key] = field
            result["all_exact"] &= same
    return result


def run_case(spec, output, source_root, timeout):
    directory = output / spec["name"]
    source = output / spec["resume"] if "resume" in spec else source_root / spec["source"] / "starter_checkpoint.npz"
    flag = "--resume" if "resume" in spec else "--starter-checkpoint"
    command = [sys.executable, "-u", str(ROOT / "run_genesis_starter_continuation.py"), flag, str(source),
               "--output", str(directory), "--duration-myr", str(spec["duration"]), "--step-myr", str(spec["step"])]
    print(f"START {spec['name']}", flush=True)
    started = perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    log = output / f"{spec['name']}.log"
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=timeout, check=False)
    wall = perf_counter() - started
    result = {"specification": spec, "command": command, "wall_seconds_including_runner_outputs": wall,
              "returncode": completed.returncode, "log": str(log), "checks": {}}
    report_path = directory / "continuation.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        verified_hashes = {name: (directory / name).is_file() and digest(directory / name) == expected
                           for name, expected in report.get("checkpoint_sha256", {}).items()}
        result.update(report=report, artifact_hashes=verified_hashes,
                      report_sha256=digest(report_path), plot_sha256=digest(directory / "continuation.png") if (directory / "continuation.png").is_file() else None)
        result["checks"] = {
            "cli_completed": completed.returncode == 0,
            "report_completed": report.get("status") == "completed",
            "all_runtime_checks": bool(report.get("checks")) and all(report["checks"].values()),
            "artifact_provenance": bool(verified_hashes) and all(verified_hashes.values()),
            "elapsed_age_preserved": math.isclose(report["final_time_myr"], report["import"]["origin_time_myr"] + spec["duration"], rel_tol=0, abs_tol=1e-10),
            "no_physical_certification_claim": report.get("physical_handoff_certified") is False,
        }
    else:
        result["checks"] = {"cli_completed": completed.returncode == 0, "report_written": False}
    print(f"DONE {spec['name']} exit={completed.returncode} wall={wall:.2f}s checks={all(result['checks'].values())}", flush=True)
    return result


def restart_comparison(output):
    left = output / "whole_1280_3myr_step1"
    right = output / "resumed_1280_total3myr_step1"
    a = json.loads((left / "continuation.json").read_text(encoding="utf-8"))
    b = json.loads((right / "continuation.json").read_text(encoding="utf-8"))
    meta_a = json.loads((left / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((right / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    mature = compare_npz(left / "mature_checkpoint/state.npz", right / "mature_checkpoint/state.npz")
    young = compare_npz(left / "young_context/starter_checkpoint.npz", right / "young_context/starter_checkpoint.npz")
    meta_diff = differences(meta_a, meta_b, "mature_meta")
    history_diff = differences(a["history"], b["history"], "history")
    totals = ("final_time_myr", "clocks", "cumulative_mantle_material_transfer_kg", "remaining_mantle_material_mass_kg",
              "final_plate_count", "transport_commits", "maximum_residual_rotation_deg", "final_mean_surface_speed_km_myr",
              "final_max_surface_speed_km_myr", "ocean_fraction")
    total_diff = differences({key: a[key] for key in totals}, {key: b[key] for key in totals}, "cumulative_final")
    return {"mature_arrays": mature, "young_arrays_and_metadata": young,
            "mature_metadata_differences": meta_diff, "history_differences": history_diff,
            "cumulative_final_differences": total_diff,
            "all_exact": bool(mature["all_exact"] and young["all_exact"] and not meta_diff and not history_diff and not total_diff),
            "note": "All mature state arrays and metadata, young context arrays/metadata, complete history and cumulative final diagnostics are compared. Segment-local ledgers are separately validated and naturally cover different intervals."}


def step_comparison(output):
    left = output / "strong_5120_10myr_step1"
    right = output / "strong_5120_10myr_step05"
    a = json.loads((left / "continuation.json").read_text(encoding="utf-8"))
    b = json.loads((right / "continuation.json").read_text(encoding="utf-8"))
    fields = ("final_plate_count", "transport_commits", "maximum_residual_rotation_deg", "final_mean_surface_speed_km_myr",
              "final_max_surface_speed_km_myr", "ocean_fraction", "cumulative_mantle_material_transfer_kg")
    diagnostics = {}
    for name in fields:
        x, y = a[name], b[name]
        diagnostics[name] = {"step1": x, "step05": y, "difference": y-x,
                             "relative_difference_vs_step1": (y-x)/abs(x) if x else None}
    return {"same_final_time": a["final_time_myr"] == b["final_time_myr"], "final_diagnostics": diagnostics,
            "mature_field_comparison": compare_npz(left / "mature_checkpoint/state.npz", right / "mature_checkpoint/state.npz"),
            "note": "Measured step sensitivity, not an asserted convergence pass. A change of topology or transported cell ownership is reported rather than suppressed."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.)
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be new or empty")
    if not math.isfinite(args.case_timeout_seconds) or args.case_timeout_seconds <= 0:
        parser.error("Case timeout must be finite and positive")
    sources = {}
    for name in sorted({spec["source"] for spec in CASES if "source" in spec}):
        for filename in ("starter_checkpoint.npz", "parameters.json"):
            path = args.sources / name / filename
            sources[str(path.resolve())] = digest(path)
    before = code_hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"format": "genesis-starter-continuation-validation-0.1", "source_hashes": sources,
              "code_hashes_before": before, "cases": {}, "checks": {}}
    write_json(args.output / "validation.json", result)
    for spec in CASES:
        if "resume" in spec and result["cases"][spec["resume"]]["returncode"] != 0:
            result["cases"][spec["name"]] = {"returncode": None, "skipped": "source continuation failed", "checks": {"source_available": False}}
        else:
            try:
                result["cases"][spec["name"]] = run_case(spec, args.output, args.sources, args.case_timeout_seconds)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                result["cases"][spec["name"]] = {"returncode": None, "error": str(exc), "checks": {"completed": False}}
        write_json(args.output / "validation.json", result)
    all_cases = all(all(case["checks"].values()) for case in result["cases"].values())
    result["checks"]["all_case_checks"] = all_cases
    result["checks"]["source_files_unchanged"] = all(digest(Path(path)) == value for path, value in sources.items())
    result["code_hashes_after"] = code_hashes()
    result["checks"]["code_unchanged_during_matrix"] = before == result["code_hashes_after"]
    if all_cases:
        result["restart_comparison"] = restart_comparison(args.output)
        result["checks"]["restart_exact"] = result["restart_comparison"]["all_exact"]
        result["step_comparison"] = step_comparison(args.output)
        result["checks"]["same_step_comparison_endpoint"] = result["step_comparison"]["same_final_time"]
    write_json(args.output / "validation.json", result)
    print(json.dumps(result["checks"], indent=2), flush=True)
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
