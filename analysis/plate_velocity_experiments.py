"""Reproducible, traced Genesis velocity comparisons in isolated CLI processes.

The source world and physical coefficients are retained. The two historical
projection arms explicitly retain legacy young mechanics; corrected_mechanics
selects the versioned source/material/SI closure. Durations are
elapsed times since the archived Starter partition, as in the continuation CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=_jsonable,
                               allow_nan=False) + "\n", encoding="utf-8")


def _stats(values, weights=None):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {key: 0.0 for key in ("mean", "median", "p90", "max", "rms")}
    if weights is None:
        weights = np.ones(len(values))
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return {"mean": float(np.average(values, weights=weights)),
            "median": float(np.interp(.5, cumulative, values[order])),
            "p90": float(np.interp(.9, cumulative, values[order])),
            "max": float(values.max()),
            "rms": float(np.sqrt(np.average(values**2, weights=weights)))}


def _boundary_metrics(mesh, boundaries, radius):
    from tectonics.kinematics import BoundaryType
    lengths = np.array([np.arccos(np.clip(np.dot(mesh.vertices[b.vertex_u],
        mesh.vertices[b.vertex_v]), -1., 1.)) * radius for b in boundaries])
    kinds = np.array([int(b.boundary_type) for b in boundaries])
    normal = np.array([b.normal_rate_km_per_myr for b in boundaries])
    return {"count": len(boundaries), "length_km": float(lengths.sum()),
        "types": {kind.name: {"count": int(np.count_nonzero(kinds == int(kind))),
            "length_km": float(lengths[kinds == int(kind)].sum())} for kind in BoundaryType},
        "relative_speed_km_myr": _stats([b.relative_speed_km_per_myr for b in boundaries], lengths),
        "absolute_normal_speed_km_myr": _stats(np.abs(normal), lengths),
        "max_convergent_normal_speed_km_myr": float(max(0., -normal.min())) if len(normal) else 0.,
        "max_divergent_normal_speed_km_myr": float(max(0., normal.max())) if len(normal) else 0.}


def _legacy_projection(mesh, cell_plate, plate_count, radius_km, flow):
    """Archived area-average operation, independent of later default changes."""
    areas = mesh.physical_cell_areas_km2(radius_km)
    count = np.bincount(cell_plate, weights=areas, minlength=plate_count)
    result = np.zeros((plate_count, 3))
    for axis in range(3):
        result[:, axis] = np.bincount(cell_plate,
            weights=areas * flow.cell_omega_rad_per_myr[:, axis], minlength=plate_count)
    result[count > 0] /= count[count > 0, None]
    return result


def run_segment(args):
    import tectonics.dynamics as dynamics
    from tectonics.kinematics import classify_boundaries

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output.with_suffix(".dynamics.jsonl")
    if trace_path.exists():
        raise ValueError(f"Trace already exists: {trace_path}")
    if args.mode == "legacy":
        dynamics.plate_mean_mantle_omega = _legacy_projection
    original = dynamics.update_plate_dynamics
    signature = inspect.signature(original)
    trace_rows = 0

    def traced(*positional, **keywords):
        nonlocal trace_rows
        arguments = signature.bind(*positional, **keywords).arguments
        parameters = arguments["params"]
        if hasattr(parameters, "mantle_projection"):
            parameters.mantle_projection = ("legacy_area_mean" if args.mode == "legacy"
                else "velocity_least_squares")
        trace = keywords.setdefault("trace", {})
        result = original(*positional, **keywords)
        mesh, state, radius = arguments["mesh"], arguments["state"], arguments["radius_km"]
        areas = mesh.physical_cell_areas_km2(radius)
        row = {"state_time_myr": float(state.time_myr), "dt_myr": float(arguments["dt_myr"]),
            "plate_count": len(result[0].plates), "trace": trace,
            "dynamics_diagnostics": asdict(result[1]),
            "boundary_before": _boundary_metrics(mesh, result[2], radius),
            "boundary_after": _boundary_metrics(mesh, classify_boundaries(mesh, result[0], radius,
                arguments["normal_threshold_km_per_myr"], arguments["inactive_speed_km_per_myr"]), radius)}
        for name in ("current_omega", "mantle_omega", "common_mantle", "target_omega",
                     "relaxed_omega", "post_gauge_omega", "final_omega"):
            speed = np.linalg.norm(np.cross(trace[name][state.cell_plate], mesh.centroids), axis=1) * radius
            row[name + "_surface_speed_km_myr"] = _stats(speed, areas)
        flow = arguments.get("mantle_flow")
        if flow is not None:
            from tectonics.mantle import plate_rigid_mantle_fit
            local = np.linalg.norm(np.cross(flow.cell_omega_rad_per_myr, mesh.centroids), axis=1) * radius
            row["local_mantle_speed_km_myr"] = _stats(local, areas)
            row["mantle_formation_rms_rad_myr"] = float(flow.formation_rms_rad_per_myr)
            row["mantle_rms_rad_myr"] = float(np.sqrt(np.mean(np.sum(flow.cell_omega_rad_per_myr**2, axis=1))))
            row["mantle_realised_amplitude_fraction"] = row["mantle_rms_rad_myr"] / max(
                row["mantle_formation_rms_rad_myr"], 1e-30)
            fitted = plate_rigid_mantle_fit(mesh, state.cell_plate, len(result[0].plates), radius, flow)
            row["rigid_fit"] = asdict(fitted)
            best_speed = np.linalg.norm(np.cross(fitted.omega_rad_per_myr[state.cell_plate],
                mesh.centroids), axis=1) * radius
            row["best_rigid_fit_surface_speed_km_myr"] = _stats(best_speed, areas)
            from tectonics.plate_velocity_diagnostics import velocity_budget
            subp = arguments.get("subduction_memory_params")
            row["velocity_budget"] = velocity_budget(mesh, state, arguments["current_system"],
                flow, radius, trace, result[2], subduction_memory=arguments.get("subduction_memory"),
                slab_reference_length_km=1800. if subp is None else subp.slab_length_cap_km)
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=_jsonable, allow_nan=False) + "\n")
        trace_rows += 1
        return result

    # Install before the runner imports its own reference to this function.
    dynamics.update_plate_dynamics = traced
    import tectonics.genesis_starter_continuation as continuation
    build = continuation.build_starter_continuation
    load_config = continuation.load_config

    def select_mode(cfg):
        from tectonics.genesis_young_mechanics import BASAL_RIDGE_MECHANICS, LEGACY_MECHANICS, mechanics_version
        desired = (
            BASAL_RIDGE_MECHANICS if args.mode == "corrected_mechanics" else LEGACY_MECHANICS)
        if args.resume and mechanics_version(cfg) != desired:
            raise ValueError("Mechanics comparison must rebuild from Starter; saved source meanings cannot be relabelled")
        cfg.setdefault("young_shell", {})["mechanics_model_version"] = desired
        if "mantle_projection" in dynamics.DynamicsParameters.__dataclass_fields__:
            cfg["plate_dynamics"]["mantle_projection"] = (
                "legacy_area_mean" if args.mode == "legacy" else "velocity_least_squares")
        elif args.mode != "legacy":
            raise RuntimeError("Corrected mantle projection is not available yet")
        return cfg

    def load_config_with_mode(*positional, **keywords):
        return select_mode(load_config(*positional, **keywords))

    def build_with_mode(*positional, **keywords):
        bundle, cfg, provenance = build(*positional, **keywords)
        return bundle, select_mode(cfg), provenance

    continuation.build_starter_continuation = build_with_mode
    continuation.load_config = load_config_with_mode
    report = continuation.run_starter_continuation(args.source, output,
        duration_myr=args.duration_myr, step_myr=1., resume=args.resume,
        cpu_workers=2, render_workers=1, cell_kernels=True,
        boundary_forces=False, frame_interval_myr=1e6, surface_only_frames=True,
        finalize=False)
    selected = {key: report[key] for key in ("status", "final_time_myr", "final_plate_count",
        "final_mean_surface_speed_km_myr", "final_max_surface_speed_km_myr", "transport_commits",
        "checks", "material_ledger", "history", "young_fracture", "young_fracture_events")}
    selected.update(mode=args.mode, source=str(args.source.resolve()), output=str(output),
        trace_file=str(trace_path), traced_steps=trace_rows)
    _write_json(output.with_suffix(".metrics.json"), selected)
    print(json.dumps({key: selected[key] for key in ("mode", "final_time_myr", "final_plate_count",
        "final_mean_surface_speed_km_myr", "final_max_surface_speed_km_myr", "transport_commits",
        "traced_steps", "checks")}))


def run_series(args):
    previous = args.source
    for duration in args.ages:
        output = args.output / args.mode / f"elapsed_{duration:04g}"
        command = [sys.executable, str(Path(__file__).resolve()), "segment", "--source", str(previous),
            "--output", str(output), "--duration-myr", str(duration), "--mode", args.mode]
        if previous != args.source or args.resume:
            command.append("--resume")
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.with_suffix(".log")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Experiment segment failed: {log}")
        metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        print(f"{args.mode} elapsed {duration:g}: "
              f"mean={metrics['final_mean_surface_speed_km_myr']:.12g}, "
              f"plates={metrics['final_plate_count']}; {log}", flush=True)
        previous = output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("segment", "series"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "velocity_least_squares", "corrected_mechanics"), default="legacy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--duration-myr", type=float)
    parser.add_argument("--ages", nargs="+", type=float, default=[5., 50., 100., 200., 400.])
    args = parser.parse_args()
    if args.action == "segment":
        if args.duration_myr is None:
            parser.error("segment requires --duration-myr")
        run_segment(args)
    else:
        run_series(args)


if __name__ == "__main__":
    main()
