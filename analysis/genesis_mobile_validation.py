"""Common-age step/mesh comparisons and a uniform unloaded mobile-shell control."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tectonics.genesis import ENERGY_SCALE, parameters_from_config
from tectonics.genesis_shell import shell_parameters_from_config
from tectonics.genesis_onset import onset_parameters_from_config
from tectonics.genesis_mobile import MOBILE_VERSION, MobileModel, mobile_parameters_from_config
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config


def _sample(model, state, thermal, orbit):
    global_row, shell, onset, orbit_row = model.diagnostics(state, thermal, orbit)
    return {"time_myr": state.time_myr, "global": global_row, "shell": shell,
            "onset": onset, "orbit": orbit_row}


def simulate(model, dt, end, *, sample_interval=.02, name="case"):
    state, thermal, orbit = model.initial()
    history = [_sample(model, state, thermal, orbit)]
    for index in range(1, round(end/dt)+1):
        state, thermal, orbit, _ = model.step(state, thermal, orbit, index*dt)
        stopped = state.stopped_reason or thermal.stopped_reason
        if index % round(sample_interval/dt) == 0 or stopped or index == round(end/dt):
            history.append(_sample(model, state, thermal, orbit))
        if index % round(.2/dt) == 0 or stopped:
            print(f"{name}: t={state.time_myr:.6f} Myr, "
                  f"accepted={state.accepted_steps}, rejected={state.rejected_steps}, "
                  f"status={stopped or 'running'}", flush=True)
        if stopped:
            break
    final = history[-1]
    scale = ENERGY_SCALE*model.thermal.area_m2
    thermal_energy = np.asarray(thermal.energy)*scale
    shell = final["shell"]
    global_row = final["global"]
    budgets = {
        "material_mass_initial_kg": state.initial_mass_kg,
        "material_mass_final_kg": float(state.layer_mass_kg.sum()),
        "material_mass_relative_residual": shell["relative_material_mass_residual"],
        "column_energy_initial_j": state.initial_column_energy_j,
        "column_energy_final_j": float(np.sum(state.layer_mass_kg*state.column_enthalpy)),
        "column_boundary_input_j": state.boundary_energy_j,
        "column_energy_relative_residual": shell["relative_column_energy_residual"],
        "global_energy_initial_j": thermal.initial_total_energy*scale,
        "global_mantle_energy_j": float(thermal_energy[0]),
        "global_surface_energy_j": float(thermal_energy[1]),
        "global_input_energy_j": float(thermal_energy[2]),
        "global_radiated_energy_j": float(thermal_energy[3]),
        "global_energy_relative_residual": global_row["relative_energy_residual"],
        "orbital_dissipated_energy_j": orbit.dissipated_energy_j,
        "tidal_heat_received_j": state.tidal_heat_received_j,
        "orbit_heat_transfer_relative_residual": final["onset"]["orbit_heat_transfer_relative_residual"],
        "water_inventory_kg": global_row["total_water_mass_kg"],
        "water_vapor_kg": global_row["vapor_mass_kg"],
        "water_ocean_kg": global_row["ocean_mass_kg"],
        "water_mass_relative_residual": (global_row["vapor_mass_kg"]+global_row["ocean_mass_kg"]
                                          -global_row["total_water_mass_kg"])/max(global_row["total_water_mass_kg"], 1.),
    }
    maxima = {
        "global_energy": max(abs(row["global"]["relative_energy_residual"]) for row in history),
        "column_energy": max(abs(row["shell"]["relative_column_energy_residual"]) for row in history),
        "material_mass": max(abs(row["shell"]["relative_material_mass_residual"]) for row in history),
        "orbit_heat_transfer": max(abs(row["onset"]["orbit_heat_transfer_relative_residual"]) for row in history),
        "mechanical_equilibrium": max(abs(row["shell"]["mechanical_equilibrium_residual"]) for row in history),
    }
    return {"model_version": MOBILE_VERSION, "geometry": "material",
            "parameters": {"thermal": asdict(model.thermal), "shell": asdict(model.p),
                           "onset": asdict(model.onset_p), "tides": asdict(model.tides_p),
                           "mobile": asdict(model.mobile_p)},
            "controls": {"requested_step_myr": dt, "sample_interval_myr": sample_interval,
                         "max_thermal_step_myr": .01, "requested_end_myr": end},
            "status": state.stopped_reason or thermal.stopped_reason or "completed",
            "events_myr": thermal.events, "final": final, "budgets": budgets,
            "maximum_sampled_absolute_residuals": maxima, "history": history}


def common_age_comparison(results):
    names = ("strong_traction", "strong_half_step", "strong_fine_mesh")
    indexed = {name: {round(row["time_myr"], 12): row for row in results[name]["history"]} for name in names}
    common = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    metrics = (("shell", "max_total_membrane_strain"), ("shell", "mean_damage"),
               ("shell", "damaged_area_fraction"), ("shell", "mean_lid_thickness_km"),
               ("shell", "mechanical_radius_km"), ("shell", "min_face_quality"),
               ("onset", "mean_speed_cm_yr"), ("onset", "max_displacement_km"))
    age = common[-1]
    values = {name: {key: indexed[name][age][group][key] for group, key in metrics} for name in names}
    baseline = values["strong_traction"]
    differences = {name: {key: {"signed_difference": value-baseline[key],
                              "relative_difference": ((value-baseline[key])/abs(baseline[key])
                                                      if abs(baseline[key]) > 1e-14 else None)}
                          for key, value in values[name].items()}
                   for name in names[1:]}
    return {"cases": names, "common_sample_ages_myr": common, "latest_common_age_myr": age,
            "requested_comparison_age_myr": 1.2, "reached_requested_common_age": abs(age-1.2) < 1e-12,
            "latest_common_values": values, "differences_from_strong_traction": differences,
            "method": "Compare saved samples at identical ages without interpolation. Terminal states at different ages are not compared."}


def save_comparison(results, path, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"strong_traction": "80 ячеек · шаг 2 тыс. лет · 50 кПа",
              "strong_half_step": "80 ячеек · шаг 1 тыс. лет · 50 кПа",
              "strong_fine_mesh": "320 ячеек · шаг 2 тыс. лет · 50 кПа",
              "uniform_free_control": "80 ячеек · равномерное остывание без нагрузок"}
    colors = {"strong_traction": "#3b6cc2", "strong_half_step": "#dd792f",
              "strong_fine_mesh": "#8b50ae", "uniform_free_control": "#238672"}
    panels = (("max_total_membrane_strain", "Максимальная накопленная деформация", "Деформация Генки"),
              ("damaged_area_fraction", "Площадь с повреждением выше порога", "Доля площади"),
              ("relative_material_mass_residual", "Баланс массы движущегося материала", "Относительная невязка"),
              ("relative_column_energy_residual", "Баланс тепла материальных колонок", "Относительная невязка"))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "#f7f9fc"}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        try:
            for ax, (metric, title, ylabel) in zip(axes.flat, panels):
                for name, case in results.items():
                    age = [row["time_myr"] for row in case["history"]]
                    value = [row["shell"][metric] for row in case["history"]]
                    ax.plot(age, value, label=labels[name], color=colors[name],
                            linestyle="--" if name == "uniform_free_control" else "-", linewidth=1.7)
                ax.set(title=title, xlabel="Возраст, млн лет", ylabel=ylabel, xlim=(0, 1.2))
                ax.grid(alpha=.22)
                if metric.startswith("relative_"):
                    peak = max(abs(row["shell"][metric]) for case in results.values() for row in case["history"])
                    extent = max(5e-16, 1.2*peak)
                    ax.set_ylim(-extent, extent)
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                else:
                    ax.set_ylim(bottom=0)
            axes[0, 0].axhline(.05, color="#697585", linewidth=.9, linestyle=":")
            axes[0, 0].text(.035, .051, "Предел прежней модели: 5%", color="#697585", fontsize=9)
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .922), ncol=2, frameon=False)
            fig.suptitle("Движущаяся оболочка · проверка шага, сетки и сохранения", fontsize=15, y=.98)
            fig.text(.5, .027, "Последний общий возраст сравнения: "
                     f"{comparison['latest_common_age_myr']:.3f} млн лет. Число ячеек и шаг указаны до адаптивного деления.\n"
                     "Балансы тепловых колонок и глобального остывания учитываются раздельно.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.085, right=.975, bottom=.12, top=.82, hspace=.4, wspace=.27)
            fig.savefig(path, dpi=140)
        finally:
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new directory")
    path = ROOT/"configs"/"genesis_moon.yaml"
    config = load_config(path)
    thermal = parameters_from_config(config)
    shell = replace(shell_parameters_from_config(config), subdivisions=1, convective_traction_pa=50000.)
    onset = onset_parameters_from_config(config)
    tides = tidal_parameters_from_config(config, thermal)
    mobile = mobile_parameters_from_config(config)
    cases = (
        ("strong_traction", shell, tides, .002),
        ("strong_half_step", shell, tides, .001),
        ("strong_fine_mesh", replace(shell, subdivisions=2), tides, .002),
        ("uniform_free_control", replace(shell, initial_temperature_anomaly_k=0., convective_traction_pa=0.),
         replace(tides, enabled=False), .002),
    )
    args.output.mkdir(parents=True)
    provenance = {"config_path": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "mobile_module_sha256": hashlib.sha256((ROOT/"tectonics"/"genesis_mobile.py").read_bytes()).hexdigest()}
    results = {}
    for name, p, t, dt in cases:
        results[name] = simulate(MobileModel(p, thermal, onset, t, mobile), dt, 1.2, name=name)
        (args.output/f"{name}.json").write_text(json.dumps(results[name], indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(name, json.dumps(results[name]["final"]["shell"]), flush=True)
    comparison = common_age_comparison(results)
    report = {"provenance": provenance, "cases": results, "comparison": comparison}
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save_comparison(results, args.output/"comparison.png", comparison)
    print("MOBILE_VALIDATION_COMPLETE", args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
