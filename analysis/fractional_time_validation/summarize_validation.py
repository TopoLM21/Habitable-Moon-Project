"""Audit case hashes/restart and compare timing sensitivity against the baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = ("starter_dt0p25", "late_dt1", "late_dt0p5", "late_dt0p25",
         "late_restart_first", "late_restart_second")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(folder):
    reports = {name: json.loads((folder/name/"report.json").read_text(encoding="utf-8")) for name in NAMES}
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
            source_unchanged=report["source_unchanged"], report_sha256=digest(folder/name/"report.json"),
            checkpoint_sha256=digest(folder/name/"fractional_checkpoint.json"))
        assert entries[name]["checkpoint_sha256"] == report["checkpoint_sha256"], name
        assert entries[name]["source_unchanged"], name
        assert all(digest(Path(path)) == expected for path, expected in report["experiment"]["source_sha256"].items()), name
        assert all(digest(ROOT/path) == expected for path, expected in report["production_sha256"].items()), name
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
    baseline = json.loads((HERE/"baseline.json").read_text(encoding="utf-8"))
    old = baseline["old_validation"]
    legacy = json.loads((HERE/"legacy_compatibility.json").read_text(encoding="utf-8"))
    assert legacy["checkpoint_bitwise_equal"]
    assert digest(ROOT/legacy["previous_checkpoint"]) == legacy["previous_sha256"]
    assert digest(ROOT/legacy["reproduced_checkpoint"]) == legacy["reproduced_sha256"]
    assert legacy["previous_sha256"] == legacy["reproduced_sha256"]
    test_log = HERE/"final_tests.log"
    tests = None
    if test_log.exists():
        matches = re.findall(r"(\d+) passed in ([0-9.]+)s", test_log.read_text(encoding="utf-8"))
        if len(matches) != 1:
            raise ValueError("Expected one complete successful final test run")
        tests = dict(passed=int(matches[0][0]), seconds=float(matches[0][1]), sha256=digest(test_log))
    summary = dict(scope="Basal-only coupled surface; timing correction only, with no slab/ridge forces or evolving fracture/topology",
        cases=entries, stepsize_sensitivity_percent=sensitivity,
        previous_stepsize_sensitivity_percent=old["stepsize_sensitivity_percent"],
        checkpoint_restart_bitwise_equal=restart_equal,
        legacy_endpoint_checkpoint_bitwise_equal=True,
        legacy_compatibility=legacy,
        analytic_birth_cooling=json.loads((HERE/"analytic_birth_cooling.json").read_text(encoding="utf-8")),
        previous_05_production_files_changed=changed,
        changed_fractional_files=[path for path, expected in baseline["production_sha256"].items()
                                 if digest(ROOT/path) != expected],
        tests=tests,
        limitations=["A 2 Myr temporal sensitivity experiment is not a certified long-term convergence study",
            "No grid convergence established for this coupled model",
            "These speeds exclude ridge/slab forces and cannot replace the complete 0.5 speed history"])
    (HERE/"validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return summary, old


def plot(summary, old):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    names, steps = ("late_dt0p25", "late_dt0p5", "late_dt1"), (.25, .5, 1.)
    for result, label, marker in ((old, "До исправления", "o"), (summary, "После исправления", "s")):
        axes[0].plot(steps, [result["cases"][name]["cumulative_thermal_sources"]["cold_mantle_volume_km3"]/1e6 for name in names],
                     marker=marker, label=label)
        axes[1].plot(steps, [result["cases"][name]["end_mean_speed_mm_per_year"] for name in names],
                     marker=marker, label=label)
    axes[0].set_title("Холодный объём: чувствительность к шагу")
    axes[0].set_ylabel("Тепловой прирост за 2 млн лет, млн км³")
    axes[1].set_title("Скорость в том же базальном эксперименте")
    axes[1].set_ylabel("Средняя скорость в конце, мм/год")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes:
        ax.set_xlabel("Шаг времени, млн лет")
        ax.set_xticks(steps)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Дробная поверхность: проверка времени рождения и охлаждения\nБез тяги погружающихся плит, хребтов и нового разрушения", fontsize=12)
    fig.savefig(HERE/"comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE/"runs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary, old = summarize(args.runs)
    if args.plot:
        plot(summary, old)
    print(json.dumps(dict(stepsize_sensitivity_percent=summary["stepsize_sensitivity_percent"],
        checkpoint_restart_bitwise_equal=summary["checkpoint_restart_bitwise_equal"],
        previous_05_production_files_changed=summary["previous_05_production_files_changed"]), indent=2))


if __name__ == "__main__":
    main()
