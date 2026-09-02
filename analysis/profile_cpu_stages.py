"""Paired initialization comparison and current CPU-stage profiling.

Each case is a fresh process; all input files are read only. Only the diagnostic
child's loaded functions are instrumented. Render-worker times overlap the
coordinator and must NOT be added to its exclusive wall-time buckets.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
import functools
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Budget:
    def __init__(self):
        self.current = "imports_and_instrumentation"
        self.last = perf_counter()
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.functions = defaultdict(float)
        self.initialization_calls = []
        self.coverage = []

    def switch(self, category):
        now = perf_counter()
        self.seconds[self.current] += now - self.last
        self.last, self.current = now, category

    @contextmanager
    def scope(self, category):
        previous = self.current
        self.switch(category)
        self.calls[category] += 1
        try:
            yield
        finally:
            self.switch(previous)

    def wrap(self, function, category, *, physics_only=False, initialization_only=False):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            if physics_only and not self.current.startswith("physics/"):
                return function(*args, **kwargs)
            if initialization_only and not self.current.startswith("initialization/"):
                return function(*args, **kwargs)
            actual = category
            if category == "rendering":
                actual += "/during_steps" if self.current.startswith("physics/") else "/reports"
            began = perf_counter()
            previous = self.current
            name = f"{function.__module__}.{function.__name__}"
            try:
                with self.scope(actual):
                    result = function(*args, **kwargs)
                if function.__name__ == "build_transport_map":
                    with self.scope("diagnostic/coverage"):
                        import numpy as np
                        multiplicity = np.sum(result.covered, axis=0)
                        self.coverage.append({"step": len(self.coverage) + 1,
                                              "cells": int(multiplicity.size),
                                              "single_source": int(np.count_nonzero(multiplicity == 1)),
                                              "overlap": int(np.count_nonzero(multiplicity > 1)),
                                              "gap": int(np.count_nonzero(multiplicity == 0))})
                return result
            finally:
                elapsed = perf_counter() - began
                self.functions[name] += elapsed
                if actual.startswith("initialization/"):
                    self.initialization_calls.append({"function": name, "parent": previous,
                                                      "inclusive_seconds": elapsed})
        return measured


def instrument_main(base, budget):
    """Keep every existing statement and its order; add scopes around blocks."""
    tree = ast.parse(inspect.getsource(base.main))
    function = tree.body[0]
    matches = [i for i, node in enumerate(function.body) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "dti"]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one main time-step loop")
    index = matches[0]
    groups = [("initialization/other", function.body[:index]),
              ("physics/other", [function.body[index]]),
              ("checkpoint_and_output", function.body[index + 1:])]
    function.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id="_cpu_stage_scope", ctx=ast.Load()),
        args=[ast.Constant(value=name)], keywords=[]))], body=body) for name, body in groups]
    ast.fix_missing_locations(tree)
    namespace = dict(base.__dict__, _cpu_stage_scope=budget.scope)
    exec(compile(tree, "<cpu-stage-main>", "exec"), namespace)
    base.__dict__["_cpu_stage_scope"] = budget.scope
    base.main = types.FunctionType(namespace["main"].__code__, base.__dict__, "main")


def install_timers(runner, budget):
    import tectonics.lithosphere_kernels  # locally imported by the integrator
    modules = [module for name, module in list(sys.modules.items()) if module and
               name.startswith(("tectonics.", "visualization.", "run_long_evolution_v"))]
    # Isolate the actual owner-selection loop, without rewriting its arithmetic.
    import tectonics.lithosphere as lithosphere
    original = lithosphere.advance_lithosphere
    tree = ast.parse(inspect.getsource(original))
    definition = tree.body[0]
    matches = [i for i, node in enumerate(definition.body) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "target"]
    if len(matches) != 1:
        raise RuntimeError("Expected one covered-cell owner-selection loop")
    index = matches[0]
    definition.body[index] = ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id="_cpu_stage_scope", ctx=ast.Load()),
        args=[ast.Constant(value="physics/cell_owner_loop")], keywords=[]))],
        body=[definition.body[index]])
    ast.fix_missing_locations(tree)
    namespace = dict(lithosphere.__dict__, _cpu_stage_scope=budget.scope)
    exec(compile(tree, "<cpu-stage-lithosphere>", "exec"), namespace)
    lithosphere.__dict__["_cpu_stage_scope"] = budget.scope
    measured = types.FunctionType(namespace[original.__name__].__code__, lithosphere.__dict__,
                                  original.__name__, original.__defaults__)
    measured.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(measured, original)
    for module in modules:
        for alias, value in list(vars(module).items()):
            if value is original:
                setattr(module, alias, measured)
    initialization = {"build_prototype": "prototype_other", "build_initial_mesh": "mesh_request",
                      "build_icosphere": "mesh", "random_plate_system": "initial_plates",
                      "rigid_motion_residual": "rigid_residual"}
    physics = {
        "build_transport_map": "transport", "advance_lithosphere": "lithosphere_other",
        "advance_sediments": "sediments", "advance_mantle_flow": "mantle",
        "update_plate_dynamics": "plate_dynamics", "advance_continental_cycle": "continental_cycle",
        "advance_topography": "topography", "advance_hydrosphere": "hydrosphere",
        "boundary_records_for_state": "boundaries", "classify_boundaries": "classify_boundaries",
        "compute_volcanic_arc_forcing": "volcanic_arcs",
        "_neighbor_root_contrast": "plume_neighbor_contrast",
        "fill_single_source_cells": "single_source_cells",
        "_redistribute_collision_overflow": "collision_overflow",
        "_redistribute_continental_footprint_overflow": "footprint_overflow",
    }
    replacements = {}
    for module in modules:
        for function in list(vars(module).values()):
            if not inspect.isfunction(function) or function in replacements:
                continue
            name, origin = function.__name__, function.__module__
            if origin.startswith("visualization.") and (name.startswith("save_") or "gif" in name):
                replacements[function] = budget.wrap(function, "rendering")
            elif origin.startswith("run_long_evolution_v") and name.startswith("_write_v"):
                replacements[function] = budget.wrap(function, "checkpoint_and_output")
            elif origin.startswith("tectonics.") and name in initialization:
                replacements[function] = budget.wrap(function, "initialization/" + initialization[name],
                                                      initialization_only=True)
            elif origin.startswith("tectonics.") and (name in physics or name.startswith("advance_")
                                                        or name == "refresh_mechanical_lithosphere"):
                replacements[function] = budget.wrap(function, "physics/" + physics.get(name, name),
                                                      physics_only=True)
    for module in modules:
        for alias, value in list(vars(module).items()):
            if inspect.isfunction(value) and value in replacements:
                setattr(module, alias, replacements[value])
    from tectonics.topology import PlateTopologyManager
    PlateTopologyManager.update = budget.wrap(PlateTopologyManager.update, "physics/topology", physics_only=True)
    instrument_main(runner.base, budget)


def child(args):
    budget = Budget()
    from execution_policy import apply_process_priority
    apply_process_priority(args.process_priority)
    from tectonics.cpu_runtime import CpuExecution
    from visualization.render_runtime import RenderExecution
    import run_long_evolution_v131 as runner

    config_path = args.config
    if args.subdivisions is not None:
        import yaml
        from tectonics.simulation import load_config
        config = load_config(config_path)
        config["mesh"]["subdivisions"] = args.subdivisions
        history = config.get("tides", {}).get("eccentricity_history_csv")
        if history:
            # The runner resolves orbital input relative to config.parent.parent.
            # Relocating our diagnostic copy must not relocate that input.
            config["tides"]["eccentricity_history_csv"] = str((config_path.parent.parent / str(history)).resolve())
        config_path = args.output / "diagnostic_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    sys.argv = ["run_long_evolution_v131.py", "--config", str(config_path),
                "--end-time", str(args.end_time), "--dt", str(args.dt),
                "--frame-interval", str(args.frame_interval), "--output", str(args.output),
                "--checkpoint", str(args.output / "checkpoint")]
    if args.resume:
        sys.argv += ["--resume", str(args.resume)]
    if args.finalize:
        sys.argv.append("--finalize")
    with CpuExecution(1, cell_kernels=True, reuse_initial_mesh=args.mode == "after",
                      numeric_kernels=args.numeric_mode != "legacy",
                      single_source_cells=args.numeric_mode.startswith("cells"),
                      cell_workers=int(args.numeric_mode[5:]) if args.numeric_mode.startswith("cells") else 1
                      ) as execution, RenderExecution(
            args.render_workers, process_priority=args.process_priority) as rendering:
        rendering.install_runner_hooks()
        if not args.plain:
            install_timers(runner, budget)
        profiler = None
        if args.cprofile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        with budget.scope("runner_other"):
            runner.main()
        # Account explicitly for the barrier and worker shutdown, outside physics.
        budget.switch("rendering/final_drain_and_shutdown")
    if profiler is not None:
        profiler.disable()
        budget.switch("diagnostic/profile_report")
        profiler.dump_stats(str(args.output / "profile.prof"))
        import pstats
        with (args.output / "profile_top.txt").open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats("tottime").print_stats(55)
            stats.sort_stats("cumulative").print_stats("tectonics", 65)
    budget.switch("finished")
    data = {"mode": args.mode, "numeric_mode": args.numeric_mode,
            "instrumented": not args.plain, "cprofile": args.cprofile,
            "exclusive_wall_seconds": dict(budget.seconds), "bucket_calls": dict(budget.calls),
            "inclusive_function_seconds_do_not_sum": dict(budget.functions),
            "initialization_calls_inclusive_do_not_sum": budget.initialization_calls,
            "transport_coverage": budget.coverage,
            "numerical_execution": execution.numerical_report(),
            "instrumented_seconds": sum(budget.seconds.values()), "render_report": rendering.report()}
    (args.output / "stage_timings.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--dt", type=float, default=4)
    parser.add_argument("--frame-interval", type=float, default=20)
    parser.add_argument("--subdivisions", type=int, help="Fresh runs only; never rewrite an input config")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["before", "after"], default=["before", "after"])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-pngs", type=Path)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--process-priority", choices=["normal", "below_normal"], default="below_normal")
    parser.add_argument("--plain", action="store_true", help="Uninstrumented A/B wall times")
    parser.add_argument("--cprofile", action="store_true", help="Attribution only, not speed benchmarking")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=["before", "after"], default="after")
    numeric_modes = ["legacy", "kernels", "cells1", "cells2", "cells4", "cells8"]
    parser.add_argument("--numeric-modes", nargs="+", choices=numeric_modes, default=["legacy"])
    parser.add_argument("--numeric-mode", choices=numeric_modes, default="legacy")
    args = parser.parse_args()
    if args.repeat < 1 or args.dt <= 0 or args.end_time <= 0 or args.frame_interval <= 0:
        parser.error("repeat and time settings must be positive")
    if args.resume and args.subdivisions is not None:
        parser.error("Cannot change a checkpoint's mesh resolution")
    for name in ("config", "resume", "output", "reference", "reference_pngs"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not args.output.is_relative_to(ROOT / "results/cpu_performance"):
        parser.error("Output must be inside results/cpu_performance")
    if args.child:
        if (args.output / "checkpoint").exists() or (args.output / "stage_timings.json").exists():
            parser.error("Diagnostic child output already contains a result")
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    from analysis.benchmark_cpu_modes import compare_checkpoints
    from analysis.benchmark_render_modes import compare_pngs
    reference, reference_pngs = args.reference, args.reference_pngs
    rows = []
    for repeat in range(args.repeat):
        modes = [(mode, numeric) for mode in args.modes for numeric in args.numeric_modes]
        if repeat % 2:
            modes.reverse()
        for mode, numeric in modes:
            label = mode if args.numeric_modes == ["legacy"] else f"{mode}_{numeric}"
            case = args.output / f"{label}_{repeat + 1}"
            case.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode,
                       "--numeric-mode", numeric,
                       "--config", str(args.config), "--output", str(case), "--end-time", str(args.end_time),
                       "--dt", str(args.dt), "--frame-interval", str(args.frame_interval),
                       "--render-workers", str(args.render_workers), "--process-priority", args.process_priority]
            if args.resume:
                command += ["--resume", str(args.resume)]
            if args.subdivisions is not None:
                command += ["--subdivisions", str(args.subdivisions)]
            command += ["--" + flag for flag in ("plain", "cprofile", "finalize") if getattr(args, flag)]
            print(f"START {label} repeat {repeat + 1}", flush=True)
            began = perf_counter()
            environment = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
            with (case / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            wall = perf_counter() - began
            data = json.loads((case / "stage_timings.json").read_text(encoding="utf-8"))
            reference = reference or case / "checkpoint"
            reference_pngs = reference_pngs or case
            row = {"case": case.name, "wall_seconds": wall, **data,
                   "checkpoint_comparison": compare_checkpoints(reference, case / "checkpoint"),
                   "png_comparison": compare_pngs(reference_pngs, case)}
            rows.append(row)
            (args.output / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            exact = row["checkpoint_comparison"]["exact"] and row["png_comparison"]["png_exact"]
            print(f"DONE {label}: {wall:.3f}s; state+PNG exact={exact}", flush=True)
            if not exact:
                raise RuntimeError("Diagnostic changed the reference result")


if __name__ == "__main__":
    main()
