"""Small shell convergence/control matrix, with reproducible boundary histories."""
from dataclasses import replace
import json
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tectonics.genesis import parameters_from_config, initial_state, advance, temperatures
from tectonics.genesis_shell import (build_icosphere, Membrane, shell_parameters_from_config,
                                    initialize_shell, advance_shell, diagnose_shell)
from tectonics.simulation import load_config


def simulate(p, thermal, dt, duration=1.5):
    mesh = build_icosphere(p.subdivisions)
    membrane = Membrane(mesh, p.poisson_ratio)
    shell = initialize_shell(mesh, p, thermal)
    global_state = initial_state(thermal)
    tm, ts = temperatures(np.array(global_state.energy), thermal)
    maximum = 0.
    max_residual = 0.
    for i in range(1, round(duration/dt)+1):
        old_tm, old_ts = tm, ts
        global_state, _ = advance(global_state, thermal, i*dt)
        tm, ts = temperatures(np.array(global_state.energy), thermal)
        shell = advance_shell(shell, mesh, membrane, p, thermal, i*dt, old_ts, ts, old_tm, tm)
        maximum = max(maximum, float(np.average(shell.damage>=p.damage_threshold, weights=mesh.areas_unit_sphere)))
        max_residual = max(max_residual, shell.equilibrium_residual)
        if shell.stopped_reason:
            break
    result = diagnose_shell(shell, mesh, p, thermal, ts, tm)
    result["max_damaged_area_fraction"] = maximum
    result["max_equilibrium_residual"] = max_residual
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    config = load_config(ROOT/"configs"/"genesis_moon.yaml")
    thermal = parameters_from_config(config)
    p = replace(shell_parameters_from_config(config), subdivisions=2)
    cases = [
        ("uniform_free_control", replace(p, initial_temperature_anomaly_k=0., convective_traction_pa=0.), .002),
        ("thermal_only", replace(p, convective_traction_pa=0.), .002),
        ("traction_20kpa", p, .002),
        ("traction_50kpa", replace(p, convective_traction_pa=50000.), .002),
        ("half_step_50kpa", replace(p, convective_traction_pa=50000.), .001),
        ("sub3_50kpa", replace(p, convective_traction_pa=50000., subdivisions=3), .002),
        ("layers64_50kpa", replace(p, convective_traction_pa=50000., column_layers=64), .002),
    ]
    result = {}
    for name, parameters, dt in cases:
        result[name] = simulate(parameters, thermal, dt)
        print(name, json.dumps(result[name]), flush=True)
    args.output.mkdir(parents=True)
    (args.output/"validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
