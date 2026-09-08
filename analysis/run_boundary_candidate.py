"""Whole-model experiment for the exact prepared-CPU boundary-force candidate.

Only this process replaces the original boundary loop. The production dynamics
module on disk, GUI defaults, checkpoints and physical parameters are unchanged.
The ordinary exact GPU surface runner handles the rest of the computation.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import AbstractContextManager
import functools
import inspect
import json
from pathlib import Path
import sys
from time import perf_counter
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.probe_gpu_boundary_forces import BoundaryGeometry, NAMES, prepared_cpu
import tectonics.dynamics as dynamics


class BoundaryCandidateContext(AbstractContextManager):
    """One current mesh, all dynamic fields refreshed by prepared_cpu each call."""
    def __init__(self):
        self.geometry = None
        self.calls = 0
        self.seconds = 0.0
        self.replacements = []
        self._installed = False

    def calculate(self, mesh, state, boundaries, radius_km, pcount, params,
                  mantle_flow, thermal_lithosphere_thickness_km, subduction_memory):
        if not self._installed:
            raise RuntimeError("Boundary candidate context is not active")
        started = perf_counter()
        if self.geometry is None or self.geometry.mesh is not mesh:
            self.geometry = BoundaryGeometry(mesh)
        result = prepared_cpu(
            self.geometry, state, boundaries, radius_km, pcount, params,
            mantle_flow=mantle_flow,
            thermal_lithosphere_thickness_km=thermal_lithosphere_thickness_km,
            subduction_memory=subduction_memory,
        )
        self.calls += 1
        self.seconds += perf_counter() - started
        return result

    def __enter__(self):
        if self._installed:
            raise RuntimeError("Boundary candidate context is already active")
        original = dynamics.update_plate_dynamics
        if getattr(original, "_stage4_boundary_candidate", False):
            raise RuntimeError("Another boundary candidate is active")
        tree = ast.parse(inspect.getsource(original))
        definition = tree.body[0]
        # Production now has an explicit opt-in dispatch. The historical
        # research wrapper still replaces this one block inside its process,
        # independently of CpuExecution's default-disabled production option.
        matches = [
            i for i, node in enumerate(definition.body)
            if isinstance(node, ast.If) and any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_boundary_force_terms_reference"
                for call in ast.walk(node)
            )
        ]
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one production boundary dispatch")
        arguments = ("mesh", "state", "boundaries", "radius_km", "pcount", "params",
                     "mantle_flow", "thermal_lithosphere_thickness_km", "subduction_memory")
        definition.body[matches[0]] = ast.Assign(
            targets=[ast.Tuple([ast.Name(name, ast.Store()) for name in NAMES], ast.Store())],
            value=ast.Call(func=ast.Name("_stage4_boundary_calculate", ast.Load()),
                           args=[ast.Name(name, ast.Load()) for name in arguments], keywords=[]))
        ast.fix_missing_locations(tree)
        namespace = dict(vars(dynamics), _stage4_boundary_calculate=self.calculate)
        exec(compile(tree, "<stage4-boundary-candidate>", "exec"), namespace)
        replacement = types.FunctionType(namespace[original.__name__].__code__, namespace,
                                         original.__name__, original.__defaults__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        functools.update_wrapper(replacement, original)
        replacement._stage4_boundary_candidate = True
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith(("tectonics.", "run_long_evolution_v")):
                continue
            for alias, value in list(vars(module).items()):
                if value is original:
                    self.replacements.append((module, alias, original, replacement))
                    setattr(module, alias, replacement)
        self._installed = True
        return self

    def report(self):
        return {
            "backend": "prepared_cpu_boundary_forces",
            "scope": "isolated process experiment; production defaults unchanged",
            "calls": self.calls,
            "inclusive_seconds": self.seconds,
            "timing_scope": "packing, lazy exact geometry, dynamics, ordered CPU sums",
            "cached_edges": 0 if self.geometry is None else len(self.geometry.edges),
            "cached_geometry_numeric_bytes": 0 if self.geometry is None else self.geometry.numeric_bytes,
            "cache_note": "one current mesh; byte count excludes Python object overhead",
        }

    def __exit__(self, *exc):
        for module, alias, original, replacement in reversed(self.replacements):
            if getattr(module, alias) is replacement:
                setattr(module, alias, original)
        self.replacements.clear()
        self._installed = False
        self.geometry = None


def main():
    # Load every runner layer before replacing its imported function aliases.
    import run_long_evolution_v131
    import run_long_evolution_v131_gpu as gpu_runner

    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        gpu_runner.main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    options, _ = parser.parse_known_args()
    with BoundaryCandidateContext() as candidate:
        gpu_runner.main()
        report = candidate.report()
        if report["calls"] < 1:
            raise RuntimeError("Candidate did not execute any boundary-force calls")
    path = Path(options.output) / "render_timings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["boundary_candidate"] = report
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
