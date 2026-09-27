"""Controls and common-time comparisons for orbit-assisted shell onset."""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import parameters_from_config
from tectonics.genesis_shell import shell_parameters_from_config
from tectonics.genesis_onset import OnsetModel, onset_parameters_from_config
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config


def simulate(model, dt, end):
    shell, thermal, onset, orbit = model.initial()
    history = []
    for i in range(1, round(end/dt)+1):
        shell, thermal, onset, orbit, _ = model.step(shell, thermal, onset, orbit, i*dt)
        if i % max(1, round(.02/dt)) == 0 or shell.stopped_reason or thermal.stopped_reason:
            g, s, o, orbital = model.diagnostics(shell, thermal, onset, orbit)
            history.append({**s, **o, "ocean_fraction": g["ocean_fraction"]})
        if shell.stopped_reason or thermal.stopped_reason:
            break
    return {"shell_parameters": asdict(model.p), "onset_parameters": asdict(model.onset_p),
            "tides_parameters": asdict(model.tides_p), "dt_myr": dt,
            "requested_end_myr": end, "final": history[-1], "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    config = load_config(ROOT/"configs"/"genesis_moon.yaml")
    thermal = parameters_from_config(config)
    shell = replace(shell_parameters_from_config(config), subdivisions=2)
    onset = onset_parameters_from_config(config)
    tides = tidal_parameters_from_config(config, thermal)
    cases = [
        ("uniform_free_control", replace(shell, initial_temperature_anomaly_k=0., convective_traction_pa=0.), onset, replace(tides, enabled=False), .002, 1.5),
        ("reference", shell, onset, tides, .002, 3.),
        ("without_tides", shell, onset, replace(tides, enabled=False), .002, 3.),
        ("without_water_weakening", shell, replace(onset, water_weakening=False), tides, .002, 3.),
        ("strong_traction", replace(shell, convective_traction_pa=50000.), onset, tides, .002, .96),
        ("strong_half_step", replace(shell, convective_traction_pa=50000.), onset, tides, .001, .96),
        ("strong_fine_mesh", replace(shell, convective_traction_pa=50000., subdivisions=3), onset, tides, .002, .96),
    ]
    results = {}
    for name, p, o, t, dt, end in cases:
        results[name] = simulate(OnsetModel(p, thermal, o, t), dt, end)
        print(name, json.dumps(results[name]["final"]), flush=True)
    args.output.mkdir(parents=True)
    (args.output/"validation.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
