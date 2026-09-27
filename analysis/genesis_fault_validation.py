"""Common-age fault-band refinement, slip-disabled and unloaded controls."""
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
from tectonics.genesis_mobile import mobile_parameters_from_config
from tectonics.genesis_faults import FAULT_VERSION, FaultModel
from tectonics.genesis_fault_law import weak_plane_parameters_from_config
from tectonics.genesis_tides import tidal_parameters_from_config
from tectonics.simulation import load_config


def _sample(model, state, thermal, orbit):
    global_row, shell, onset, orbit_row = model.diagnostics(state, thermal, orbit)
    friction_sum = float(state.friction_work_cell_j.sum())
    viscous_sum = float(state.viscous_work_cell_j.sum())
    return {"time_myr": state.time_myr, "global": global_row, "shell": shell,
            "onset": onset, "orbit": orbit_row,
            "fault_work_ledger_relative_residuals": {
                "friction": (friction_sum-state.friction_work_j)/max(state.friction_work_j, 1.),
                "viscous": (viscous_sum-state.viscous_fault_work_j)/max(state.viscous_fault_work_j, 1.)}}


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
    budgets["fault_mechanical_work"] = {
        "friction_work_j": state.friction_work_j,
        "viscous_fault_work_j": state.viscous_fault_work_j,
        "fault_dissipation_j": state.friction_work_j+state.viscous_fault_work_j,
        "friction_cell_work_sum_j": float(state.friction_work_cell_j.sum()),
        "viscous_cell_work_sum_j": float(state.viscous_work_cell_j.sum()),
        "minimum_friction_cell_work_j": float(state.friction_work_cell_j.min()),
        "minimum_viscous_cell_work_j": float(state.viscous_work_cell_j.min()),
        "minimum_sampled_friction_increment_j": float(np.diff([row["onset"]["friction_work_j"] for row in history]).min()),
        "minimum_sampled_viscous_increment_j": float(np.diff([row["onset"]["viscous_fault_work_j"] for row in history]).min()),
        "cell_sum_relative_residuals": final["fault_work_ledger_relative_residuals"],
        "included_in_global_heat": False,
        "included_in_column_heat": False,
        "interpretation": "Backward-Euler frictional and viscous shear-work estimates; not a closed total mechanical-energy budget.",
    }
    budgets["fault_shear"] = {
        "band_width_km": model.fault_p.band_width_km,
        "maximum_accumulated_shear": float(state.cumulative_shear.max()),
        "minimum_accumulated_shear": float(state.cumulative_shear.min()),
        "maximum_absolute_signed_shear": float(np.abs(state.signed_shear).max()),
        "max_equivalent_slip_km": shell["max_equivalent_slip_km"],
        "mean_equivalent_slip_km": shell["mean_equivalent_slip_km"],
        "active_material_cell_count": int(state.fault_active.sum()),
        "definition": "Equivalent slip = explicit band width times accumulated absolute shear; no displacement discontinuity.",
    }
    maxima = {
        "global_energy": max(abs(row["global"]["relative_energy_residual"]) for row in history),
        "column_energy": max(abs(row["shell"]["relative_column_energy_residual"]) for row in history),
        "material_mass": max(abs(row["shell"]["relative_material_mass_residual"]) for row in history),
        "orbit_heat_transfer": max(abs(row["onset"]["orbit_heat_transfer_relative_residual"]) for row in history),
        "mechanical_equilibrium": max(abs(row["shell"]["mechanical_equilibrium_residual"]) for row in history),
        "friction_cell_sum": max(abs(row["fault_work_ledger_relative_residuals"]["friction"]) for row in history),
        "viscous_cell_sum": max(abs(row["fault_work_ledger_relative_residuals"]["viscous"]) for row in history),
    }
    return {"model_version": FAULT_VERSION, "geometry": "material",
            "parameters": {"thermal": asdict(model.thermal), "shell": asdict(model.p),
                           "onset": asdict(model.onset_p), "tides": asdict(model.tides_p),
                           "mobile": asdict(model.mobile_p), "faults": asdict(model.fault_p)},
            "controls": {"requested_step_myr": dt, "sample_interval_myr": sample_interval,
                         "max_thermal_step_myr": .01, "requested_end_myr": end},
            "status": state.stopped_reason or thermal.stopped_reason or "completed",
            "events_myr": thermal.events, "final": final, "budgets": budgets,
            "maximum_sampled_absolute_residuals": maxima, "history": history}


def common_age_comparison(results, names=("strong_traction", "strong_half_step", "strong_fine_mesh")):
    """Compare saved samples at exactly matching ages, never terminal aliases."""
    indexed = {name: {round(row["time_myr"], 12): row for row in results[name]["history"]}
               for name in names}
    common = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    if not common:
        raise ValueError("Comparison cases have no common saved age")
    metrics = (("shell", "max_total_membrane_strain"), ("shell", "mean_damage"),
               ("shell", "damaged_area_fraction"), ("shell", "fault_active_area_fraction"),
               ("shell", "fault_slipping_area_fraction"), ("shell", "max_equivalent_slip_km"),
               ("shell", "mean_equivalent_slip_km"), ("shell", "mean_lid_thickness_km"),
               ("shell", "mechanical_radius_km"), ("shell", "min_face_quality"),
               ("onset", "mean_speed_cm_yr"), ("onset", "max_displacement_km"),
               ("onset", "friction_work_j"), ("onset", "viscous_fault_work_j"),
               ("onset", "fault_dissipation_j"))
    age = common[-1]
    values = {name: {key: indexed[name][age][group][key] for group, key in metrics} for name in names}
    baseline = values[names[0]]
    differences = {name: {key: {"signed_difference": value-baseline[key],
                               "relative_difference": ((value-baseline[key])/abs(baseline[key])
                                                       if abs(baseline[key]) > 1e-14 else None)}
                          for key, value in values[name].items()}
                   for name in names[1:]}
    return {"cases": list(names), "baseline_case": names[0],
            "common_sample_ages_myr": common, "latest_common_age_myr": age,
            "requested_comparison_age_myr": 1.2, "reached_requested_common_age": abs(age-1.2) < 1e-12,
            "latest_common_values": values, "differences_from_baseline": differences,
            "method": "Compare saved samples at identical ages without interpolation. Terminal states at different ages are not compared."}


def save_comparison(results, path, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"strong_traction": "80 ячеек · 2 тыс. лет · сдвиг включён",
              "strong_half_step": "80 ячеек · 1 тыс. лет · сдвиг включён",
              "strong_fine_mesh": "320 ячеек · 2 тыс. лет · сдвиг включён",
              "strong_without_slip": "80 ячеек · 2 тыс. лет · сдвиг выключен",
              "uniform_free_control": "80 ячеек · равномерное остывание без нагрузок"}
    colors = {"strong_traction": "#3b6cc2", "strong_half_step": "#dd792f",
              "strong_fine_mesh": "#8b50ae", "strong_without_slip": "#a44d4b",
              "uniform_free_control": "#238672"}
    panels = (("shell", "max_equivalent_slip_km", "Максимальный эквивалентный сдвиг", "км"),
              ("shell", "fault_active_area_fraction", "Площадь с закреплёнными слабыми плоскостями", "Доля площади"),
              ("onset", "friction_work_j", "Работа сопротивления сдвигу", "Дж · отдельно от теплового баланса"),
              ("onset", "viscous_fault_work_j", "Вязкая работа в разломных зонах", "Дж · отдельно от теплового баланса"),
              ("shell", "relative_material_mass_residual", "Баланс массы материала", "Относительная невязка"),
              ("shell", "relative_column_energy_residual", "Баланс тепла материальных колонок", "Относительная невязка"))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "#f7f9fc"}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11.5))
        try:
            for ax, (group, metric, title, ylabel) in zip(axes.flat, panels):
                for name, case in results.items():
                    age = [row["time_myr"] for row in case["history"]]
                    value = [row[group][metric] for row in case["history"]]
                    style = ":" if name == "uniform_free_control" else "--" if name == "strong_without_slip" else "-"
                    ax.plot(age, value, label=labels[name], color=colors[name], linestyle=style, linewidth=1.6)
                ax.set(title=title, xlabel="Возраст, млн лет", ylabel=ylabel, xlim=(0, 1.2))
                ax.grid(alpha=.22)
                if metric.startswith("relative_"):
                    peak = max(abs(row[group][metric]) for case in results.values() for row in case["history"])
                    extent = max(5e-16, 1.2*peak)
                    ax.set_ylim(-extent, extent)
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                else:
                    ax.set_ylim(bottom=0)
                    if metric.endswith("_j"):
                        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .925),
                       ncol=2, frameon=False, fontsize=9)
            fig.suptitle("Разломные зоны · шаг, сетка, отключение сдвига и сохранение", fontsize=15, y=.978)
            fig.text(.5, .036,
                     f"Общий возраст сравнения шага и сетки: {comparison['latest_common_age_myr']:.3f} млн лет. "
                     "Нагрузка сильных опытов: 50 кПа.\n"
                     "Эквивалентный сдвиг = ширина зоны × накопленная сдвиговая деформация; оболочка остаётся связной.\n"
                     "Работа трения и вязкости записана отдельно: она не добавлена в тепло колонок или глобальное тепло.",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=.085, right=.975, bottom=.125, top=.81, hspace=.48, wspace=.27)
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
    fault = weak_plane_parameters_from_config(config)
    cases = (
        ("strong_traction", shell, tides, fault, .002),
        ("strong_half_step", shell, tides, fault, .001),
        ("strong_fine_mesh", replace(shell, subdivisions=2), tides, fault, .002),
        ("strong_without_slip", shell, tides, replace(fault, enabled=False), .002),
        ("uniform_free_control", replace(shell, initial_temperature_anomaly_k=0., convective_traction_pa=0.),
         replace(tides, enabled=False), fault, .002),
    )
    args.output.mkdir(parents=True)
    source_files = ("tectonics/genesis_faults.py", "tectonics/genesis_fault_law.py",
                    "tectonics/genesis_mobile.py", "analysis/genesis_fault_validation.py")
    provenance = {"config_path": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "module_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}}
    results = {}
    for name, p, t, f, dt in cases:
        results[name] = simulate(FaultModel(p, thermal, onset, t, mobile, f), dt, 1.2, name=name)
        (args.output/f"{name}.json").write_text(json.dumps(results[name], indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(name, json.dumps(results[name]["final"]["shell"]), flush=True)
    comparison = common_age_comparison(results)
    slip_comparison = common_age_comparison(results, ("strong_traction", "strong_without_slip"))
    all_comparison = common_age_comparison(results, tuple(results))
    report = {"model_version": FAULT_VERSION, "geometry": "material", "provenance": provenance,
              "cases": results, "comparison": comparison, "slip_toggle_comparison": slip_comparison,
              "all_cases_comparison": all_comparison,
              "scope": "Fixed-connectivity continuous material shell with finite-width irreversible shear; no detached plate contact.",
              "mechanical_work_accounting": "Fault frictional and viscous work is diagnostic only, excluded from global and column heat. This report does not assert a closed total mechanical-energy balance."}
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save_comparison(results, args.output/"comparison.png", comparison)
    print("FAULT_VALIDATION_COMPLETE", args.output.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
