"""Small reproducible genesis sensitivity/convergence study, no mature runs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tectonics.genesis import parameters_from_config, run_genesis
from tectonics.simulation import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    p = parameters_from_config(load_config(ROOT / "configs" / "genesis_moon.yaml"))
    runs = {}
    cases = [
        ("reference", p, 0.01),
        ("half_step", p, 0.005),
        ("brighter_star", replace(p, stellar_flux_w_m2=2000), 0.01),
        ("lower_water", replace(p, water_volume_km3=0.5*p.water_volume_km3), 0.01),
        ("critical_transition_2K", replace(p, critical_transition_width_k=2), 0.01),
        ("critical_transition_10K", replace(p, critical_transition_width_k=10), 0.01),
    ]
    report = {}
    for name, params, step in cases:
        state, rows = run_genesis(params, max_step_myr=step)
        runs[name] = rows
        report[name] = {"events_myr": state.events, "final": rows[-1],
                        "max_relative_energy_residual": max(abs(r["relative_energy_residual"]) for r in rows)}
    report["convergence"] = {
        "event_difference_years": {name: abs(time-report["half_step"]["events_myr"][name])*1e6
                                   for name, time in report["reference"]["events_myr"].items()},
        "final_surface_difference_k": abs(report["reference"]["final"]["surface_temperature_k"] - report["half_step"]["final"]["surface_temperature_k"]),
        "final_mantle_difference_k": abs(report["reference"]["final"]["mantle_temperature_k"] - report["half_step"]["final"]["mantle_temperature_k"]),
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    labels = {"reference": "Звезда 1284 Вт/м² · базовый запас воды",
              "brighter_star": "Звезда 2000 Вт/м² · тот же запас",
              "lower_water": "Звезда 1284 Вт/м² · половина воды"}
    for name, label in labels.items():
        rows = runs[name]
        t = [r["time_myr"] for r in rows]
        axes[0].plot(t, [r["surface_temperature_k"] for r in rows], label=label)
        axes[1].plot(t, [r["ocean_fraction"] for r in rows], label=label)
    axes[0].set(ylabel="Температура поверхности, K", title="Сильнее облучение — медленнее остывание")
    axes[1].set(ylabel="Доля запаса воды в океане", title="Океан появляется по условиям, а не по таймеру")
    for ax in axes:
        ax.set_xlabel("Время после расплавленного старта, млн лет")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Генезис · чувствительность упрощённой тепловой модели", fontsize=14)
    fig.savefig(args.output / "stellar_comparison.png", dpi=155)
    plt.close(fig)
    print(json.dumps(report["convergence"], indent=2))


if __name__ == "__main__":
    main()
