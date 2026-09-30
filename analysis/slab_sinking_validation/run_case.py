"""Isolated, traced slab-closure experiments; never rewrite source checkpoints.

``series`` runs fresh at the archived Starter by default; an explicit
``--resume`` continues a validated arm without changing its physical model.
``frozen`` explicitly probes counterfactual coefficients on a saved state and
does not claim an evolved geological result or a changed checkpoint model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/genesis_runs/starter_20260928_192306_547551/starter_checkpoint.npz"
SECTIONS = {"plate_dynamics", "subduction_memory"}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=jsonable, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    """Record all mechanics code, including new modules added during development."""
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
            for path in sorted((ROOT / "tectonics").glob("*.py"))}


def source_hashes(source):
    if source.is_file():
        result = {str(source): digest(source)}
        metadata = source.parent / "parameters.json"
        if metadata.exists():
            result[str(metadata)] = digest(metadata)
        return result
    report = json.loads((source / "continuation.json").read_text(encoding="utf-8"))
    hashes = {name: digest(source / name) for name in report["checkpoint_sha256"]}
    if hashes != report["checkpoint_sha256"]:
        raise ValueError("Saved source integrity mismatch")
    hashes["continuation.json"] = digest(source / "continuation.json")
    return hashes


def overrides(args):
    result = {}
    if args.slab_model is not None:
        result["plate_dynamics.young_slab_force_model"] = args.slab_model
    for item in args.parameter:
        key, separator, raw = item.partition("=")
        if not separator or len(key.split(".")) != 2 or key.split(".")[0] not in SECTIONS:
            raise ValueError("Use --parameter plate_dynamics.NAME=JSON or subduction_memory.NAME=JSON")
        if key in result:
            raise ValueError(f"Duplicate override: {key}")
        result[key] = json.loads(raw)
    return result


def configured(cfg, selected, *, resume=False):
    from tectonics.dynamics import DynamicsParameters
    from tectonics.subduction_memory import SubductionMemoryParameters
    from tectonics.genesis_young_mechanics import corrected_mechanics
    if not corrected_mechanics(cfg):
        raise ValueError("Slab experiments require existing versioned SI mechanics; legacy sources cannot be relabelled")
    types = {"plate_dynamics": DynamicsParameters, "subduction_memory": SubductionMemoryParameters}
    result = deepcopy(cfg)
    for qualified, value in selected.items():
        section, key = qualified.split(".")
        if key not in types[section].__dataclass_fields__:
            raise ValueError(f"Unknown physical parameter: {qualified}")
        existing = getattr(types[section](**result[section]), key)
        if resume and existing != value:
            raise ValueError(f"Resume cannot change {qualified}: {existing!r} -> {value!r}; restart an independent arm from Starter")
        result[section][key] = value
    return result


def speed_stats(mesh, owner, radius, omega):
    from tectonics.plate_velocity_diagnostics import weighted_stats
    values = np.linalg.norm(np.cross(np.asarray(omega)[owner], mesh.centroids), axis=1) * radius
    return weighted_stats(values, mesh.physical_cell_areas_km2(radius))


def force_metrics(trace):
    """Raw trace is retained; these common metrics also work with old disabled arms."""
    result = {key: float(value) for key, value in trace.items()
              if key.endswith("_w") and np.isscalar(value)}
    names = ("basal_driving_torque_nm", "ridge_torque_nm", "slab_torque_nm")
    if all(name in trace for name in names) and "target_torque_residual_nm" in trace:
        scale = sum(float(np.linalg.norm(trace[name])) for name in names)
        scale += float(np.linalg.norm(trace.get("slab_constraint_reaction_torque_nm", 0.)))
        result["target_torque_relative_residual"] = float(
            np.linalg.norm(trace["target_torque_residual_nm"]) / max(scale, 1.))
        result["target_torque_residual_scale"] = "sum_of_source_and_reaction_torque_norms"
    components = ("mantle_omega", "ridge_drive_raw", "slab_drive_raw")
    if all(name in trace for name in components) and "target_omega" in trace:
        reconstructed = sum(np.asarray(trace[name]) for name in components)
        reconstructed += np.asarray(trace.get("slab_constraint_omega", 0.))
        result["target_velocity_component_relative_error"] = float(np.linalg.norm(
            reconstructed-np.asarray(trace["target_omega"]))/max(np.linalg.norm(trace["target_omega"]), 1e-30))
    # Old disabled traces contain basal dissipation only. New traces count
    # gross slab gravity work and expose the complete dissipation explicitly.
    dissipation_name = "total_dissipation_w" if "total_dissipation_w" in trace else "basal_drag_dissipation_w"
    if all(name in trace for name in ("total_source_power_w", dissipation_name, "transient_net_power_w")):
        result["net_power_balance_residual_w"] = float(trace["total_source_power_w"]
            + trace.get("slab_constraint_power_w", 0.)
            - trace[dissipation_name] - trace["transient_net_power_w"])
        scale = max(abs(float(trace["total_source_power_w"])), abs(float(trace[dissipation_name])), 1.)
        result["net_power_balance_relative_residual"] = result["net_power_balance_residual_w"] / scale
        result["transient_net_power_fraction"] = float(trace["transient_net_power_w"]) / scale
        result["slab_constraint_power_fraction"] = float(trace.get("slab_constraint_power_w", 0.)) / scale
    if "total_drag_tensor_nm_s" in trace:
        matrix = np.asarray(trace["total_drag_tensor_nm_s"])
        scale = max(float(np.linalg.norm(matrix)), 1.)
        result["total_drag_relative_asymmetry"] = float(np.linalg.norm(matrix-matrix.T) / scale)
        spectrum = np.linalg.eigvalsh(.5*(matrix+matrix.T) / scale)
        result["total_drag_min_normalized_eigenvalue"] = float(spectrum.min())
        result["total_drag_condition_number"] = float(np.linalg.cond(matrix / scale))
    for key in ("basal_drag_dissipation_w", "slab_bending_dissipation_w", "slab_mantle_dissipation_w", "total_dissipation_w"):
        if key in trace:
            result[key + "_nonnegative"] = bool(trace[key] >= -1e-9)
    if "slab_sinking_diagnostics" in trace:
        result["slab_sinking_diagnostics"] = trace["slab_sinking_diagnostics"]
    if trace.get("slab_sections"):
        ratios = [float(section["neck_tension_n"])/max(float(section["neck_capacity_n"]), 1.)
            for section in trace["slab_sections"] if "neck_tension_n" in section]
        result["maximum_attached_neck_tension_to_capacity"] = max(ratios, default=0.)
    return result


def run_segment(args):
    import tectonics.dynamics as dynamics
    import tectonics.genesis_starter_continuation as continuation
    from tectonics.young_boundary import accepted_slab_inventory_diagnostics
    selected = overrides(args)
    source, output = args.source.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    siblings = [output.with_suffix(suffix) for suffix in (".dynamics.jsonl", ".metrics.json", ".request.json")]
    if output.exists() or any(path.exists() for path in siblings):
        raise ValueError("Experiment output and trace must be new; nothing is overwritten")
    hashes, source_before = code_hashes(), source_hashes(source)
    write_json(siblings[2], dict(action="segment", source=source, output=output,
        process_id=os.getpid(), parent_process_id=os.getppid(),
        source_sha256=source_before, parameter_overrides=selected, resume=args.resume,
        requested_duration_myr=args.duration_myr, step_myr=args.step_myr,
        requested_mechanics_version=args.mechanics_version,
        fresh_default_mechanics_version=args.default_mechanics_version,
        requested_subdivisions=args.subdivisions, production_sha256=hashes))
    original, original_build, original_load = dynamics.update_plate_dynamics, continuation.build_starter_continuation, continuation.load_config
    signature = inspect.signature(original)
    count, last = 0, None

    def traced(*positional, **keywords):
        nonlocal count, last
        arguments = signature.bind(*positional, **keywords).arguments
        trace = keywords.setdefault("trace", {})
        result = original(*positional, **keywords)
        mesh, state = arguments["mesh"], arguments["state"]
        radius = arguments["radius_km"]
        last = dict(state_time_myr=float(state.time_myr), dt_myr=float(arguments["dt_myr"]),
            plate_count=len(result[0].plates), dynamics_diagnostics=asdict(result[1]),
            metrics=force_metrics(trace), trace=trace,
            accepted_slab_inventory=accepted_slab_inventory_diagnostics(arguments.get("subduction_memory")))
        for field in ("current_omega", "mantle_omega", "target_omega", "final_omega"):
            last[field + "_surface_speed_mm_yr"] = speed_stats(mesh, state.cell_plate, radius, trace[field])
        with siblings[0].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(last, default=jsonable, allow_nan=False) + "\n")
        count += 1
        return result

    def build(model, state, mature_config):
        # Version and inventory-affecting choices must be applied before the
        # import creates its accepted-material inventory. Keep this historical
        # harness pinned to 0.4 even when production gains a new default.
        cfg = deepcopy(mature_config)
        cfg.setdefault("young_shell", {})["mechanics_model_version"] = (
            args.mechanics_version or args.default_mechanics_version)
        return original_build(model, state, configured(cfg, selected))

    def load(*positional, **keywords):
        cfg = original_load(*positional, **keywords)
        if args.resume and args.mechanics_version is not None:
            from tectonics.genesis_young_mechanics import mechanics_version
            if mechanics_version(cfg) != args.mechanics_version:
                raise ValueError("Resume cannot change saved mechanics version")
        return configured(cfg, selected, resume=True) if args.resume else cfg

    dynamics.update_plate_dynamics = traced
    continuation.build_starter_continuation = build
    continuation.load_config = load
    # Preserve exact small solver inputs if a long geological run finds a
    # numerical corner case; no production function or source state is edited.
    try:
        import tectonics.young_slab_constraints as constraints
    except ImportError:
        constraints = None
    if constraints is not None:
        solve_constraints = constraints.solve_no_eduction
        def capture_solver_failure(drag, torque, feed, **keywords):
            try:
                return solve_constraints(drag, torque, feed, **keywords)
            except Exception:
                arrays = dict(drag_nm_s=drag, driving_torque_nm=torque, feed_matrix_m=feed)
                if keywords.get("reaction_weights") is not None:
                    arrays["reaction_weights"] = keywords["reaction_weights"]
                np.savez_compressed(output.with_suffix(".solver_failure.npz"), **arrays)
                raise
        constraints.solve_no_eduction = capture_solver_failure
    started = time.perf_counter()
    report = continuation.run_starter_continuation(source, output,
        duration_myr=args.duration_myr, step_myr=args.step_myr, resume=args.resume,
        subdivisions=args.subdivisions, cpu_workers=args.cpu_workers, render_workers=1,
        cell_kernels=True, boundary_forces=False, frame_interval_myr=1e6,
        surface_only_frames=True, finalize=False)
    after = code_hashes()
    unchanged_source = source_before == source_hashes(source)
    summary = {key: report.get(key) for key in ("status", "final_time_myr", "duration_myr", "step_myr",
        "mechanics_model_version", "young_slab_force_model", "final_plate_count",
        "final_mean_surface_speed_km_myr", "final_max_surface_speed_km_myr", "transport_commits",
        "checks", "accepted_slab_inventory", "material_ledger", "limitations")}
    summary.update(source=str(source), output=str(output), parameter_overrides=selected,
        source_unchanged=unchanged_source, source_sha256=source_before,
        elapsed_seconds=time.perf_counter()-started, trace_steps=count,
        production_sha256=hashes, production_files_changed_during_run=[name for name in hashes.keys() | after.keys() if hashes.get(name) != after.get(name)],
        last_dynamics_metrics=None if last is None else last["metrics"],
        thermal_energy_relative_residual=report["history"][-1]["thermal_energy_relative_residual"])
    write_json(siblings[1], summary)
    print(json.dumps(summary, default=jsonable), flush=True)
    if not unchanged_source or report["status"] != "completed":
        raise SystemExit("Experiment source integrity or continuation checks failed")


def run_frozen(args):
    import tectonics.simulation as simulation
    from tectonics.plate_velocity_diagnostics import diagnose_checkpoint
    source, output = args.source.resolve(), args.output.resolve()
    if source.name == "mature_checkpoint":
        source = source.parent
    if output.exists():
        raise ValueError("Frozen result must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected, hashes, source_before = overrides(args), code_hashes(), source_hashes(source)
    load = simulation.load_config
    def frozen_config(*positional, **keywords):
        cfg = load(*positional, **keywords)
        if args.mechanics_version is not None:
            from tectonics.genesis_young_mechanics import mechanics_version
            if mechanics_version(cfg) != args.mechanics_version:
                raise ValueError("A frozen checkpoint retains its saved mechanics version")
        return configured(cfg, selected)
    simulation.load_config = frozen_config
    try:
        report = diagnose_checkpoint(source, step_myr=args.step_myr)
    finally:
        simulation.load_config = load
    report.update(evaluation="frozen_counterfactual_force_probe_no_evolution",
        parameter_overrides=selected, production_sha256=hashes, source_sha256=source_before,
        source_unchanged=source_before == source_hashes(source),
        metrics=force_metrics(report["trace"]))
    metadata = json.loads((source / "mature_checkpoint/meta.json").read_text(encoding="utf-8"))
    from tectonics.subduction_memory import memory_from_json
    from tectonics.young_boundary import accepted_slab_inventory_diagnostics
    report["accepted_slab_inventory"] = accepted_slab_inventory_diagnostics(memory_from_json(metadata["subduction_memory"]))
    write_json(output, report)
    print(json.dumps(dict(output=str(output), parameter_overrides=selected,
        source_unchanged=report["source_unchanged"], metrics=report["metrics"])), flush=True)
    if not report["source_unchanged"]:
        raise SystemExit("Frozen probe changed source files")


def run_series(args):
    if args.output.exists():
        raise ValueError("Series output must be new")
    if args.ages != sorted(set(args.ages)) or any(age <= 0 for age in args.ages):
        raise ValueError("Series ages must be positive and strictly increasing")
    args.output.mkdir(parents=True)
    write_json(args.output / "series.json", dict(process_id=os.getpid(),
        source=str(args.source.resolve()), resume=args.resume, ages=args.ages,
        stop_after_segment_file=str((args.output / "STOP_AFTER_SEGMENT").resolve())))
    source = args.source.resolve()
    for index, age in enumerate(args.ages):
        output = args.output / ("elapsed_" + f"{age:04g}".replace(".", "p"))
        command = [sys.executable, str(Path(__file__).resolve()), "segment", "--source", str(source),
            "--output", str(output), "--duration-myr", str(age), "--step-myr", str(args.step_myr),
            "--cpu-workers", str(args.cpu_workers)]
        if index or args.resume:
            command.append("--resume")
        if args.mechanics_version is not None or not args.resume:
            command += ["--mechanics-version", args.mechanics_version or args.default_mechanics_version]
        if args.slab_model is not None:
            command += ["--slab-model", args.slab_model]
        if args.subdivisions is not None:
            command += ["--subdivisions", str(args.subdivisions)]
        for item in args.parameter:
            command += ["--parameter", item]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
            "CUPY_CACHE_DIR": str(ROOT / ".venv/cupy-cache"), "QT_QPA_PLATFORM": "offscreen"}
        with output.with_suffix(".log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Experiment failed: {output.with_suffix('.log')}")
        metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        print(json.dumps({key: metrics[key] for key in ("duration_myr", "final_mean_surface_speed_km_myr",
            "final_max_surface_speed_km_myr", "final_plate_count", "transport_commits", "checks")}), flush=True)
        if (args.output / "STOP_AFTER_SEGMENT").exists():
            print("Requested stop after completed segment; checkpoint preserved", flush=True)
            break
        source = output


def main(*, default_mechanics_version="young-mechanics-0.4"):
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("series", "segment", "frozen"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slab-model", help="Exact production closure name; omitted retains saved/default choice")
    parser.add_argument("--mechanics-version", help=("Explicit version for a fresh import; resume/frozen must match the saved version. "
        f"Fresh default: {default_mechanics_version}; saved versions are preserved."))
    parser.set_defaults(default_mechanics_version=default_mechanics_version)
    parser.add_argument("--parameter", action="append", default=[], metavar="SECTION.KEY=JSON")
    parser.add_argument("--step-myr", type=float, default=1.)
    parser.add_argument("--duration-myr", type=float)
    parser.add_argument("--ages", type=float, nargs="+", default=[50., 100., 200., 400.])
    parser.add_argument("--subdivisions", type=int)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "segment" and args.duration_myr is None:
        parser.error("segment requires --duration-myr")
    if args.action == "frozen" and (args.subdivisions is not None or args.resume):
        parser.error("frozen uses the saved mesh and material; refinement requires a fresh series")
    {"series": run_series, "segment": run_segment, "frozen": run_frozen}[args.action](args)


if __name__ == "__main__":
    main()
