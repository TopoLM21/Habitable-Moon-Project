"""Summarize actual coupled cases and plot numerical sensitivity, not full speeds."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports = {name: json.loads((HERE/"runs"/name/"report.json").read_text(encoding="utf-8"))
               for name in ("starter_dt0p25", "late_dt1", "late_dt0p5", "late_dt0p25", "late_restart_first", "late_restart_second")}
    entries = {}
    for name, report in reports.items():
        final = report["history"][-1]
        entries[name] = dict(
            start_mean_speed_mm_per_year=report["initial_diagnostics"]["dynamics"]["mean_speed_mm_per_year"],
            end_mean_speed_mm_per_year=final["endpoint_dynamics"]["mean_speed_mm_per_year"],
            end_max_speed_mm_per_year=final["endpoint_dynamics"]["max_speed_mm_per_year"],
            cumulative_losses=report["cumulative_losses"], cumulative_births=report["cumulative_births"],
            cumulative_thermal_sources=report["cumulative_thermal_sources"],
            remaining_reference_mantle_mass_kg=report["remaining_reference_mantle_mass_kg"],
            final_parcel_count=final["parcel_count"], wall_seconds=report["wall_seconds"],
            maximum_material_relative_residual=max(abs(value) for row in report["history"]
                                                  for value in row["relative_residuals"].values()),
            maximum_energy_relative_residual=max(abs(row["heat"]["relative_energy_residual"]) for row in report["history"]),
            maximum_torque_relative_residual=max(row["endpoint_dynamics"]["torque_relative_residual"] for row in report["history"]),
            maximum_power_relative_residual=max(row["endpoint_dynamics"]["power_relative_residual"] for row in report["history"]),
            source_unchanged=report["source_unchanged"], report_sha256=digest(HERE/"runs"/name/"report.json"),
            checkpoint_sha256=digest(HERE/"runs"/name/"fractional_checkpoint.json"))
        assert entries[name]["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert entries[name]["source_unchanged"]
        assert all(digest(Path(path)) == expected for path, expected in report["experiment"]["source_sha256"].items())
        assert all(digest(ROOT/path) == expected for path, expected in report["production_sha256"].items())
    direct, resumed = reports["late_dt1"], reports["late_restart_second"]
    restart_equal = direct["checkpoint_sha256"] == resumed["checkpoint_sha256"]
    assert restart_equal and direct["history"] == resumed["history"]
    sensitivity = {}
    for first, second in (("late_dt1", "late_dt0p5"), ("late_dt0p5", "late_dt0p25")):
        coarse, fine = entries[first], entries[second]
        changes = {key: (fine[key]/coarse[key]-1.)*100. for key in
                   ("end_mean_speed_mm_per_year", "end_max_speed_mm_per_year")}
        for key, group, quantity in (("removed_basalt_volume", "cumulative_losses", "oceanic_volume_km3"),
                ("created_basalt_volume", "cumulative_births", "oceanic_volume_km3"),
                ("thermal_cold_volume_source", "cumulative_thermal_sources", "cold_mantle_volume_km3"),
                ("thermal_excess_mass_source", "cumulative_thermal_sources", "density_excess_mass_kg")):
            changes[key] = (fine[group][quantity]/coarse[group][quantity]-1.)*100.
        sensitivity[f"{first}_to_{second}"] = changes
    previous = json.loads((ROOT/"analysis/slab_sinking_followup/validation.json").read_text(encoding="utf-8"))
    changed = [path for path, expected in previous["current_production_sha256"].items()
               if digest(ROOT/path) != expected]
    assert not changed, changed
    transport = json.loads((ROOT/"analysis/fractional_transport_validation/validation_summary.json").read_text(encoding="utf-8"))
    changed_transport = [path for path, expected in transport["production_sha256"].items() if digest(ROOT/path) != expected]
    assert not changed_transport, changed_transport
    test_log = HERE/"final_tests.log"
    tests = re.findall(r"(\d+) passed in ([0-9.]+)s", test_log.read_text(encoding="utf-8"))
    if len(tests) != 1:
        raise ValueError("Expected one complete successful final test run")
    summary = dict(scope="Basal-only coupled surface with Genesis heat; no slab/ridge forces or fracture/topology evolution",
        cases=entries, stepsize_sensitivity_percent=sensitivity, checkpoint_restart_bitwise_equal=restart_equal,
        previous_05_production_files_changed=changed, previous_fractional_transport_files_changed=changed_transport,
        tests=dict(passed=int(tests[0][0]), seconds=float(tests[0][1]), sha256=digest(test_log)),
        limitations=["Three stepsizes over 2 Myr test short temporal sensitivity, not certified long-term convergence",
            "No grid convergence established for this coupled model",
            "These speeds exclude slab and ridge forces and cannot replace the full 0.5 speed history"])
    (HERE/"validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for name, label, marker in (("late_dt1", "Шаг 1 млн лет", "o"), ("late_dt0p5", "Шаг 0,5 млн лет", "s"),
                                ("late_dt0p25", "Шаг 0,25 млн лет", "^")):
        report = reports[name]
        start = report["experiment"]["source_time_myr"]-report["experiment"]["origin_time_myr"]
        x = [start]+[row["elapsed_since_source_myr"]+start for row in report["history"]]
        y = [report["initial_diagnostics"]["dynamics"]["mean_speed_mm_per_year"]]+[
            row["endpoint_dynamics"]["mean_speed_mm_per_year"] for row in report["history"]]
        axes[0].plot(x, y, marker=marker, label=label)
        removed = [0.]+[row["cumulative_losses"]["oceanic_volume_km3"]/1000. for row in report["history"]]
        axes[1].plot(x, removed, marker=marker, label=label)
        cold_source = [report["initial_diagnostics"]["cooling"]["thermal_source_delta"]["cold_mantle_volume_km3"]/1e6]+[
            row["cumulative_thermal_sources"]["cold_mantle_volume_km3"]/1e6 for row in report["history"]]
        axes[2].plot(x, cold_source, marker=marker, label=label)
    axes[0].set_title("Базальное равновесие пересчитывается")
    axes[0].set_ylabel("Средняя скорость, мм/год")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].set_title("Удалённый материал сохранён в архиве")
    axes[1].set_ylabel("Океаническая кора, тыс. км³")
    axes[2].set_title("Охлаждение ещё чувствительно к шагу")
    axes[2].set_ylabel("Прирост холодного объёма, млн км³")
    for ax in axes:
        ax.set_xlabel("Время после первого разделения, млн лет")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Дробный перенос + локальное охлаждение + базальные силы\nБез тяги погружающихся плит, хребтов и нового разрушения", fontsize=12)
    fig.savefig(HERE/"comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
